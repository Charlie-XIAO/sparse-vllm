from abc import ABC, abstractmethod
from typing import List

import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.outputs import RequestOutput


class KVCacheSparsifierBase(ABC):
    """Base class for KV cache sparsifiers."""

    def __init__(self, num_tokens_budget: int,
                 num_tokens_per_eviction: int) -> None:
        self.num_tokens_budget = num_tokens_budget
        self.num_tokens_per_eviction = num_tokens_per_eviction

        if self.num_tokens_per_eviction >= self.num_tokens_budget:
            raise ValueError("The number of tokens per KV cache eviction must "
                             "be strictly less than the KV cache budget")

    @abstractmethod
    def step(self, block_manager: BlockSpaceManagerV1, seq_id: int,
             attn_scores: torch.Tensor) -> None:
        """Proceed by one iteration.

        This will instruct the block manager to deactivate specific blocks if we
        are running out of KV cache budget. Subclasses must implement this
        abstract method.

        Note: `attn_scores` has shape (num_layers, num_heads, num_tokens).
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
