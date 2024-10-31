"""Token blocks."""
from typing import TYPE_CHECKING, Iterator, List, Optional

import numpy as np

from vllm.utils import Device

DEFAULT_LAST_ACCESSED_TIME: float = -1


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens

        self.ref_count = 0
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME

        self.computed = False

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed})')


class BlockTable:
    """Holds a list of blocks with caching of their associated block_ids.

    Each slot in the block mask is True if that slot is active and False if that
    slot is inactive. Note that all slots are initially marked as active even if
    they are not yet occupied.
    """

    def __init__(self, blocks: Optional[List[PhysicalTokenBlock]] = None):
        self._blocks: List[PhysicalTokenBlock] = []
        self._block_ids: List[int] = []
        self._block_masks: List[np.ndarray] = []

        if blocks is not None:
            for block in blocks:
                self.append(block)

    def append(self, block: PhysicalTokenBlock):
        self._blocks.append(block)
        self._block_ids.append(block.block_number)
        self._block_masks.append(self._get_fresh_mask(block))

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, key):
        return self._blocks[key]

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[PhysicalTokenBlock]:
            raise RuntimeError("Method should be automatically generated")

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            blocks = value
            self._blocks[key] = blocks
            self._block_ids[key] = [b.block_number for b in blocks]
            self._block_masks[key] = [self._get_fresh_mask(b) for b in blocks]
        else:
            block = value
            self._blocks[key] = block
            self._block_ids[key] = block.block_number
            self._block_masks[key] = self._get_fresh_mask(block)

    def _get_fresh_mask(self, block: PhysicalTokenBlock) -> np.ndarray:
        return np.ones(block.block_size, dtype=np.bool_)

    def reset(self):
        self._blocks = []
        self._block_ids = []
        self._block_masks = []

    def copy(self) -> "BlockTable":
        return BlockTable(self._blocks)

    def list(self) -> List[PhysicalTokenBlock]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_ids

    def masks(self) -> List[np.ndarray]:
        return self._block_masks

    def _set_slots_status(self, slots: List[int], status: bool):
        # NOTE(Charlie-XIAO): This is assuming that the block sizes of all
        # physical blocks in this block table are the same. This also assumes
        # that the physical blocks are logically contiguous.
        block_size = self._blocks[0].block_size
        for slot in slots:
            i, j = divmod(slot, block_size)
            self._block_masks[i][j] = status

    def deactivate_slots(self, slots: List[int]):
        self._set_slots_status(slots, False)

    def activate_slots(self, slots: List[int]):
        self._set_slots_status(slots, True)
