from collections.abc import Iterable

import numpy as np
from cs336_basics.types import PackedPair, pack_pair

def to_encoded_pairs(bb: list[tuple[int, int]]) -> list[PackedPair]:
    return  [pack_pair(a, b) for (a, b) in bb]

def from_encoded_pairs(bb: np.ndarray) -> list[tuple[int, int]]:
    return [(int(b >> 16), int(b & 0xFFFF)) for b in bb]

# def
