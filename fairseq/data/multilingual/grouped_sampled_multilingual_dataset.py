# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.data.multilingual.sampled_multi_dataset import (
    CollateFormat,
    default_virtual_size_func,
)
from fairseq.data.multilingual.sampled_multilingual_dataset import (
    SampledMultilingualDataset,
)

logger = logging.getLogger(__name__)


class GroupedSampledMultilingualDataset(SampledMultilingualDataset):
    """
    Ensures that each batch contains samples of only one group pair.
    Otherwise functions like SampledMultiDataset.
    """

    def __init__(
        self,
        datasets,
        sampling_ratios=None,
        seed=2,
        epoch=1,
        eval_key=None,
        collate_format=CollateFormat.single,
        virtual_size=default_virtual_size_func,
        split="",
        shared_collater=False,
        shuffle=True,
        src_lang_groups=None,
        tgt_lang_groups=None,
    ):
        super().__init__(
            datasets,
            sampling_ratios,
            seed,
            epoch,
            eval_key,
            collate_format,
            virtual_size,
            split,
            shared_collater,
            shuffle,
        )

        def _groups_to_dict(groups):
            return {lang: frozenset(group) for group in groups for lang in group}

        src_groups = (
            _groups_to_dict(src_lang_groups) if src_lang_groups is not None else {}
        )
        tgt_groups = (
            _groups_to_dict(tgt_lang_groups) if tgt_lang_groups is not None else {}
        )

        self.key_to_group = {}

        for key in self.keys:
            src, tgt = key.split("-")
            self.key_to_group[
                key
            ] = f"{src_groups.get(src, src)}-{tgt_groups.get(tgt, tgt)}"

        self.groups = set(self.key_to_group.values())

    def _group_indices(self, indices):
        grouped_indices = {k: [] for k in self.groups}

        for i in indices:
            ds_idx, _ = self._get_dataset_and_index(i)
            grouped_indices[self.key_to_group[self.keys[ds_idx]]].append(i)

        return list(grouped_indices.items())
