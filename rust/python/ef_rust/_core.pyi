"""Stub for the private native module ``ef_rust._core`` (hand-written; checked with stubtest)."""

from collections.abc import Sequence
from typing import final

import numpy as np
import numpy.typing as npt

__all__ = ["DEBUG_BUILD", "HashingKernel"]

DEBUG_BUILD: bool

@final
class HashingKernel:
    def __new__(
        cls,
        word_ranges: Sequence[tuple[int, int]],
        /,
        *,
        dim: int,
        ngram_min: int,
        ngram_max: int,
        sublinear_tf: bool,
    ) -> HashingKernel: ...
    def accumulate_into(
        self, texts: list[object], out: npt.NDArray[np.float32], /
    ) -> list[int]: ...
    @property
    def dim(self) -> int: ...
