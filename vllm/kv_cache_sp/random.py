from typing import List, Tuple

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
             attn_scores: torch.Tensor) -> Tuple[bool, int, int]:
        num_slots = attn_scores.size(2)
        (_, active_slots, num_total_slots,
         num_active_slots) = self._get_blocks_info(block_manager, seq_id,
                                                   num_slots)

        if num_active_slots <= self.num_tokens_budget:
            # We have not exceeded the budget so no need for eviction
            return (False, num_active_slots, num_total_slots)

        slots_to_evict = np.random.choice(active_slots,
                                          self.num_tokens_per_eviction,
                                          replace=False)
        block_manager.deactivate_slots(seq_id, slots_to_evict)

        return (True, num_active_slots, num_total_slots)

    def clean_self(self, outputs: List[RequestOutput]) -> None:
        pass  # No-op
