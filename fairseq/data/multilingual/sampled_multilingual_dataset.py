# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np

from fairseq.data import SampledMultiDataset

logger = logging.getLogger(__name__)


class SampledMultilingualDataset(SampledMultiDataset):
    """
    Ensures that each batch contains samples only from one dataset.
    Otherwise functions like SampledMultiDataset.
    """

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        dataset_indices = [[] for _ in range(len(self.datasets))]
        for i in indices:
            ds_idx, _ = self._get_dataset_and_index(i)
            dataset_indices[ds_idx].append(i)

        batches = []
        for ds_idx, indices in enumerate(dataset_indices):
            cur_batches = super().batch_by_size(
                np.array(indices, dtype=np.int64),
                max_tokens,
                max_sentences,
                required_batch_size_multiple,
            )
            logger.info(
                f"Created {len(cur_batches)} batches for dataset {self.keys[ds_idx]}"
            )
            batches += cur_batches

        return batches

    def filter_indices_by_size(self, indices, max_sizes):
        if isinstance(max_sizes, dict):
            max_sizes = next(iter(max_sizes.values()))
        return super().filter_indices_by_size(indices, max_sizes)
