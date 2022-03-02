# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from collections import OrderedDict

from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.utils import check_lang_groups, csv_str_list, list_of_csv_str_lists

from ..data import FairseqDataset, data_utils, iterators
from ..data.multilingual.grouped_sampled_multilingual_dataset import (
    GroupedSampledMultilingualDataset,
)
from ..data.multilingual.sampled_multi_dataset import CollateFormat
from ..data.multilingual.sampled_multilingual_dataset import SampledMultilingualDataset
from ..data.multilingual.sampling_method import SamplingMethod
from .translation_multi_simple_epoch import get_time_gap

logger = logging.getLogger(__name__)


@register_task("multilingual_translation_sampled")
class SampledMultilingualTranslationTask(MultilingualTranslationTask):
    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        SamplingMethod.add_arguments(parser)
        parser.add_argument(
            "--model-lang-pairs",
            default=None,
            type=csv_str_list,
            help="language pairs that will be used for building the model. --lang-pairs are used by default.",
        )
        parser.add_argument(
            "--eval-lang-pairs",
            default=None,
            type=csv_str_list,
            help="language pairs that will be used for evaluating the model. --lang-pairs are used by default.",
        )
        parser.add_argument(
            "--src-data-lang-groups",
            default=None,
            type=list_of_csv_str_lists,
        )
        parser.add_argument(
            "--tgt-data-lang-groups",
            default=None,
            type=list_of_csv_str_lists,
        )

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        if training:
            self.model_lang_pairs = (
                self.lang_pairs
                if args.model_lang_pairs is None
                else args.model_lang_pairs
            )
            self.eval_lang_pairs = (
                self.lang_pairs
                if args.eval_lang_pairs is None
                else args.eval_lang_pairs
            )

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a dataset split."""
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split("-")
            langpair_dataset = load_langpair_dataset(
                data_path,
                split,
                src,
                self.dicts[src],
                tgt,
                self.dicts[tgt],
                combine=True,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[src].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
            )

        datasets = OrderedDict(
            [
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.lang_pairs
            ]
        )

        sampling_method = SamplingMethod(self.args, self).sampling_method_selector()
        ratios = (
            None
            if sampling_method is None
            else sampling_method([len(dataset) for dataset in datasets.values()])
        )

        if (
            self.args.src_data_lang_groups is not None
            or self.args.tgt_data_lang_groups is not None
        ):
            check_lang_groups(self.args.src_data_lang_groups)
            check_lang_groups(self.args.tgt_data_lang_groups)
            self.datasets[split] = GroupedSampledMultilingualDataset(
                datasets,
                epoch=epoch,
                sampling_ratios=ratios,
                seed=self.args.seed,
                collate_format=CollateFormat.ordered_dict_simple,
                eval_key=None
                if self.training
                else f"{self.args.source_lang}-{self.args.target_lang}",
                src_lang_groups=self.args.src_data_lang_groups,
                tgt_lang_groups=self.args.tgt_data_lang_groups,
            )
        else:
            self.datasets[split] = SampledMultilingualDataset(
                datasets,
                epoch=epoch,
                sampling_ratios=ratios,
                seed=self.args.seed,
                collate_format=CollateFormat.ordered_dict,
                eval_key=None
                if self.training
                else f"{self.args.source_lang}-{self.args.target_lang}",
            )

    # needs to be overridden to work with SampledMultiDataset
    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {
                f"{self.args.source_lang}-{self.args.target_lang}": (
                    self.args.max_source_positions,
                    self.args.max_target_positions,
                )
            }
        return OrderedDict(
            [
                (key, (self.args.max_source_positions, self.args.max_target_positions))
                for split in self.datasets.keys()
                for key in self.datasets[split].keys
            ]
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        # each sample contains one language-pair
        assert len(sample) == 1
        model.train()

        lang_pair = next(iter(sample.keys()))
        loss, sample_size, logging_output = criterion(
            model.models[lang_pair], sample[lang_pair]
        )
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    # from translation_multi_simple_epoch
    def create_batch_sampler_func(
        self,
        max_positions,
        ignore_invalid_inputs,
        max_tokens,
        max_sentences,
        required_batch_size_multiple=1,
        seed=1,
    ):
        def construct_batch_sampler(dataset, epoch):
            splits = [
                s for s, _ in self.datasets.items() if self.datasets[s] == dataset
            ]
            split = splits[0] if len(splits) > 0 else None
            # NEW implementation
            if epoch is not None:
                # initialize the dataset with the correct starting epoch
                dataset.set_epoch(epoch)

            # get indices ordered by example size
            start_time = time.time()
            logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
            logger.info(
                f"[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # filter examples that are too large
            if max_positions is not None:
                my_time = time.time()
                indices = self.filter_indices_by_size(
                    indices, dataset, max_positions, ignore_invalid_inputs
                )
                logger.info(
                    f"[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
                )
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # create mini-batches with given size constraints
            my_time = time.time()
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

            logger.info(
                f"[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}"
            )
            logger.info(
                f"[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            return batch_sampler

        return construct_batch_sampler

    # from translation_multi_simple_epoch
    # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # initialize the dataset with the correct starting epoch
        assert isinstance(dataset, FairseqDataset)
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]
        if self.args.sampling_method == "RoundRobin":
            batch_iter = super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                skip_remainder_batch=False,
                grouped_shuffling=False,
                update_epoch_batch_itr=False,
            )
            self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter

        construct_batch_sampler = self.create_batch_sampler_func(
            max_positions,
            ignore_invalid_inputs,
            max_tokens,
            max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
        )

        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=construct_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        return epoch_iter
