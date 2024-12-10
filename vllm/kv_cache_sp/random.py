from typing import List

import numpy as np
import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.kv_cache_sp.base import (KVCacheSparsifierBase,
                                   KVCacheSparsifierStepOutput)
from vllm.outputs import RequestOutput


class RandomKVCacheSparsifier(KVCacheSparsifierBase):
    """A random KV cache sparsifier.

    This is completely for experimental purposes. It randomly evicts tokens
    when exceeding KV cache budget which makes no sense.
    """

    def step(self, block_manager: BlockSpaceManagerV1, seq_id: int,
             attn_scores: torch.Tensor) -> KVCacheSparsifierStepOutput:
        num_slots = attn_scores.size(2)

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

        # Randomly choose slots to evict from the active slots
        slots_to_evict = np.random.choice(active_slots,
                                          self.num_per_evict,
                                          replace=False)

        num_removed_blocks = 0

        if self.internal == "no-op":
            block_manager.deactivate_slots(seq_id, slots_to_evict)

        elif self.internal == "free-block":
            block_manager.deactivate_slots(seq_id, slots_to_evict)
            # Free fully deactivate blocks and slice the attention scores
            # accordingly to keep consistency
            removed_blocks = block_manager.free_fully_deactivated_blocks(
                seq_id)
            # Update for the returned step output
            block_masks = block_manager.block_tables[seq_id].masks()
            num_removed_blocks = len(removed_blocks)

        elif self.internal == "copy":
            raise NotImplementedError  # TODO(Charlie-XIAO)!

        elif self.internal == "spvllm":
            raise NotImplementedError  # TODO(Charlie-XIAO)

        else:
            raise ValueError(
                "Unrecognized KV cache internal memory management "
                f"strategy: {self.internal}")

        # The block masks have been changed in the previous step; the stats need
        # to be based on the updated version
        return KVCacheSparsifierStepOutput(
            do_evict=True,
            num_active_slots=num_active_slots,
            num_total_slots=len(block_masks) * block_size,
            num_removed_blocks=num_removed_blocks)

    def clean_self(self, outputs: List[RequestOutput]) -> None:
        pass  # No-op
