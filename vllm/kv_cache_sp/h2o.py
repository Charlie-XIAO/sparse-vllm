import math
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

    def __init__(self, budget: int, num_per_evict: int, internal: str) -> None:
        super().__init__(budget, num_per_evict, internal)

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
                num_evicted_tokens=0,
                num_migrate_dst_blocks=0,
                slots_to_migrate=[])

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

        num_evicted_tokens = 0
        num_migrate_dst_blocks = 0
        slots_to_migrate = []

        if self.internal == "no-op":
            block_manager.deactivate_slots(seq_id, slots_to_evict)

        elif self.internal == "free-block":
            block_manager.deactivate_slots(seq_id, slots_to_evict)
            # Free fully deactivate blocks and slice the attention scores
            # accordingly to keep consistency
            removed_blocks = block_manager.free_fully_deactivated_blocks(
                seq_id)
            for i in removed_blocks:
                self.seq_ids_to_cum_attn_scores[seq_id] = np.delete(
                    self.seq_ids_to_cum_attn_scores[seq_id],
                    np.s_[i * block_size:(i + 1) * block_size])
            # Update for the returned step output
            block_masks = block_manager.block_tables[seq_id].masks()
            num_evicted_tokens = len(removed_blocks) * block_size

        elif self.internal == "copy":
            self.seq_ids_to_cum_attn_scores[seq_id] = np.delete(
                self.seq_ids_to_cum_attn_scores[seq_id], slots_to_evict)
            num_evicted_tokens = len(slots_to_evict)
            num_migrate_dst_blocks = math.ceil(
                (num_slots - num_evicted_tokens) / block_size)
            slots_to_migrate = np.setdiff1d(np.arange(num_slots),
                                            slots_to_evict).tolist()

        elif self.internal == "spvllm":
            raise NotImplementedError  # TODO(Charlie-XIAO)

        else:
            raise ValueError(
                "Unrecognized KV cache internal memory management "
                f"strategy: {self.internal}")

        return KVCacheSparsifierStepOutput(
            do_evict=True,
            num_active_slots=num_active_slots,
            num_total_slots=len(block_masks) * block_size,
            num_evicted_tokens=num_evicted_tokens,
            num_migrate_dst_blocks=num_migrate_dst_blocks,
            slots_to_migrate=slots_to_migrate)

    def clean_self(self, outputs: List[RequestOutput]) -> None:
        for output in outputs:
            for seq_id in output.seq_ids:
                self.seq_ids_to_cum_attn_scores.pop(seq_id, None)
