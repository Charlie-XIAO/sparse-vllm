from typing import Dict, List

import numpy as np
import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.kv_cache_sp.base import (KVCacheSparsifierBase,
                                   KVCacheSparsifierStepOutput)
from vllm.outputs import RequestOutput


class H2OKVCacheSparsifier(KVCacheSparsifierBase):
    """The H2O KV cache sparsifier.

    H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
    Language Models.
    https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf
    """

    def __init__(self, budget: int, num_per_evict: int) -> None:
        super().__init__(budget, num_per_evict)

        self.seq_ids_to_cum_attn_scores: Dict[int, torch.Tensor] = {}

    def step(self, block_manager: BlockSpaceManagerV1, seq_id: int,
             attn_scores: torch.Tensor) -> KVCacheSparsifierStepOutput:
        num_slots = attn_scores.size(2)

        # Accumulate the attention scores
        agg_attn_scores = attn_scores.numpy().mean(axis=(0, 1))
        if seq_id in self.seq_ids_to_cum_attn_scores:
            self.seq_ids_to_cum_attn_scores[seq_id].resize(num_slots)
            self.seq_ids_to_cum_attn_scores[seq_id] += agg_attn_scores
        else:
            self.seq_ids_to_cum_attn_scores[seq_id] = agg_attn_scores

        # Flatten block mask and get indices of active slots
        block_masks = block_manager.block_tables[seq_id].masks()
        block_size = len(block_masks[0])
        total_block_mask = np.concatenate(block_masks)[:num_slots]
        active_slots = np.where(total_block_mask)[0]
        num_active_slots = len(active_slots)

        if num_active_slots <= self.budget:
            # We have not exceeded the budget so no need for eviction
            return KVCacheSparsifierStepOutput(
                do_evict=False,
                num_active_slots=num_active_slots,
                num_total_slots=len(block_masks) * block_size,
                num_removed_blocks=0)

        # We should keep the k last tokens and the k tokens from the rest with
        # the highest attention scores
        num_keep = self.budget - self.num_per_evict + 1
        k = num_keep // 2
        k_last = (num_keep + 1) // 2

        # - `active_slots[:-k_last]` is because the k last tokens will always be
        #   preserved and should not take part in the top-k selection
        # - `self.seq_ids_to_cum_attn_scores[seq_id][...]` is retrieving the
        #   values of the candidate slots
        # - `np.argpartition(..., -k)[-k:]` is obtaining the indices (relative
        #   to the candidate slots)
        # - `active_slots[...]` is converting the indices to be relative to the
        #   whole array
        topk_slots = active_slots[np.argpartition(
            self.seq_ids_to_cum_attn_scores[seq_id][active_slots[:-k_last]],
            -k)[-k:]]

        # We know the indices we want to keep; by flipping boolean mask we can
        # get the indices we want to evict; note that we avoid evicting masked
        # indices again though it is no-op, so we perform a boolean and with the
        # original mask
        evict_mask = np.ones(num_slots, dtype=np.bool_)
        evict_mask[topk_slots] = False
        evict_mask[active_slots[-k_last:]] = False
        slots_to_evict = np.where(evict_mask & total_block_mask)[0]

        # Deactivate the slots, then free fully deactivated blocks and slice
        # the attention scores accordingly
        block_manager.deactivate_slots(seq_id, slots_to_evict)
        removed_blocks = block_manager.free_fully_deactivated_blocks(seq_id)
        for i in removed_blocks:
            self.seq_ids_to_cum_attn_scores[seq_id] = np.delete(
                self.seq_ids_to_cum_attn_scores[seq_id],
                np.s_[i * block_size:(i + 1) * block_size])

        # The block masks have been changed in the previous step; the stats need
        # to be based on the updated version
        block_masks = block_manager.block_tables[seq_id].masks()
        return KVCacheSparsifierStepOutput(
            do_evict=True,
            num_active_slots=num_active_slots,
            num_total_slots=len(block_masks) * block_size,
            num_removed_blocks=len(removed_blocks))

    def clean_self(self, outputs: List[RequestOutput]) -> None:
        for output in outputs:
            for seq_id in output.seq_ids:
                self.seq_ids_to_cum_attn_scores.pop(seq_id, None)
