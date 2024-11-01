from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.outputs import RequestOutput


class KVCacheSparsifierBase(ABC):
    """Base class for KV cache sparsifiers."""

    def __init__(self, budget: int, num_per_evict: int) -> None:
        self.budget = budget
        self.num_per_evict = num_per_evict

        if self.num_per_evict >= self.budget:
            raise ValueError("The number of tokens per KV cache eviction must "
                             "be strictly less than the KV cache budget")

    @abstractmethod
    def step(self, block_manager: BlockSpaceManagerV1, seq_id: int,
             attn_scores: torch.Tensor) -> Tuple[bool, int, int]:
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

        Returns
        -------
        0 : bool
            Whether an eviction is performed.
        1 : int
            The number of active slots (in particular, the slots that are
            currently occupied and not deactivated). This is useful for
            computing internal fragmentation.
        2 : int
            The total number of slots (in particular, this not only includes
            deactivated slots but also slots that are not yet occupied). This is
            useful for computing internal fragmentation, i.e., the total number
            minus the active number.
        """
        pass

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
        pass

    def _get_blocks_info(
            self, block_manager: BlockSpaceManagerV1, seq_id: int,
            num_slots: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Get information of the blocks.

        Parameters
        ----------
        block_manager : BlockSpaceManagerV1
        seq_id : int
        num_slots : int
            The number of slots that are occupied.

        Returns
        -------
        0 : np.ndarray
            The block masks flattened/concatenated together and stripped to the
            number of slots that are occupied.
        1 : np.ndarray
            The indices of the active slots.
        2 : int
            The total number of slots.
        3 : int
            The number of active slots.
        """
        # Compute the total number of slots; this excludes blocks that are fully
        # deactivated (i.e., all False) which would have been freed
        block_masks = block_manager.block_tables[seq_id].masks()
        num_total_slots = sum(len(mask) for mask in block_masks if mask.any())

        # Flatten the masks and strip to the actual number of slots (i.e.,
        # excluding slots that are not yet occupied)
        total_block_mask = np.concatenate(block_masks)[:num_slots]
        active_slots = np.where(total_block_mask)[0]
        num_active_slots = len(active_slots)

        return total_block_mask, active_slots, num_total_slots, num_active_slots
