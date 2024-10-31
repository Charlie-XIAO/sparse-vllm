from typing import List

import numpy as np
import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.kv_cache_sp.base import KVCacheSparsifierBase
from vllm.outputs import RequestOutput


class RandomKVCacheSparsifier(KVCacheSparsifierBase):
    """A random KV cache sparsifier.

    This is completely for experimental purposes. It randomly evicts tokens
    when exceeding KV cache budget which makes no sense.
    """

    def step(self, block_manager: BlockSpaceManagerV1, seq_id: int,
             attn_scores: torch.Tensor) -> None:
        num_slots = attn_scores.size(2)
        mask = np.concatenate(
            block_manager.block_tables[seq_id].masks())[:num_slots]
        active_slots = np.where(mask)[0]
        num_active_slots = len(active_slots)

        if num_active_slots <= self.num_tokens_budget:
            # We have not exceeded the budget so no need for eviction
            return

        slots_to_evict = np.random.choice(active_slots,
                                          self.num_tokens_per_eviction,
                                          replace=False)
        block_manager.deactivate_slots(seq_id, slots_to_evict)

    def clean_self(self, outputs: List[RequestOutput]) -> None:
        pass  # No-op
