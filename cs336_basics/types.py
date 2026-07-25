import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Self

Pair = tuple[bytes, bytes]
IdPair = tuple[int, int]
PairCount = Counter[Pair]
IdPairCount = Counter[IdPair]

PackedPair = int

PackedPairCount = Counter[PackedPair]

TieBreak = Callable[[PackedPair, PackedPair], bool]


def pack_pair(a: int, b: int) -> PackedPair:
    return (a << 16) | b


def unpack_pair(enc_pair: PackedPair) -> tuple[int, int]:
    return enc_pair >> 16, enc_pair & 0xFFFF


class PackedPairLoc(defaultdict[PackedPair, set[int]]):
    def __init__(self):
        super().__init__(set)


@dataclass(frozen=True)
class SpecialToken:
    text: str


Pretoken = SpecialToken | str
