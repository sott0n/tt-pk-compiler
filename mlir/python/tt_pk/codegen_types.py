"""Shared dataclasses used by both schedule building and backend emission.

Lives outside codegen.py so resident.py can reference the types without a
circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CBSpec:
    """One CB that lives on a set of cores."""
    buffer_index: int
    dtype: str
    page_size: int
    tile_count: int
    cores: Tuple[Tuple[int, int], ...] = ()

    def total_size(self) -> int:
        return self.tile_count * self.page_size


@dataclass(frozen=True)
class SemSpec:
    """One tt-metal semaphore the runtime creates in the program."""
    id: int
    producer: str = ""
    consumer: str = ""
    initial_value: int = 0
    cores: Tuple[Tuple[int, int], ...] = ()
