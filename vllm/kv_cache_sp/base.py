from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.outputs import RequestOutput


@dataclass
class KVCacheSparsifierStepOutput:
    # Whether an eviction is performed in this step
    do_evict: bool

    # The number of active slots and the total number of slots; their difference
    # is the internal fragmentation (deactivated slots and unoccupied slots)
    num_active_slots: int
    num_total_slots: int

    # The number of tokens that are evicted. This does not include slots that
    # are masked, but that are actually removed from the physical block table
    # layout.
    num_evicted_tokens: int

    # The number of blocks that we will migrate to, and the slots that should be
    # copied during the migration. This happens when using the copy internal
    # strategy, where all current blocks will be freed and new blocks will be
    # used.
    num_migrate_dst_blocks: int
    slots_to_migrate: List[int]


class KVCacheSparsifierBase(ABC):
    """Base class for KV cache sparsifiers."""

    def __init__(self, budget: int, num_per_evict: int, internal: str) -> None:
        self.budget = budget
        self.num_per_evict = num_per_evict
        self.internal = internal

    @abstractmethod
    def step(self, block_manager: BlockSpaceManagerV1, seq_id: int,
             attn_scores: torch.Tensor) -> KVCacheSparsifierStepOutput:
        """Proceed by one iteration.

        This will instruct the block manager to deactivate specific blocks if we
        are running out of KV cache budget. Subclasses must implement this
        abstract method.

        Parameters
        ----------
        block_manager : BlockSpaceManagerV1
            The block manager for managing the currently KV cache blocks. Only
            v1 block manager is supported for now.
        seq_id : int
            The ID of the sequence.
        attn_scores : torch.Tensor
            The attention scores of shape (num_layers, num_heads, num_tokens).
        """
        raise NotImplementedError

    @abstractmethod
    def clean_self(self, outputs: List[RequestOutput]) -> None:
        """Clean up the sparsifier itself.

        Some KV cache sparsifiers store additional information internally, among
        which some may never be used again. This function should thus manage the
        cleanup when possible. Subclasses must implement this abstract method,
        even if it is no-op.

        Parameters
        ----------
        outputs : List[RequestOutput]
            The request outputs, which carry information about the requests that
            have been completed.
        """
        raise NotImplementedError
