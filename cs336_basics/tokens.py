from collections.abc import Iterable
from dataclasses import dataclass
from typing import final

import numpy as np

from .model_types import PackedPair, PackedPairCount, pack_pair, unpack_pair


# Converts string to tuple of bytes
def to_bytes(string: str, offset: int = 0) -> np.ndarray:
    return np.array([b + offset for b in bytes(string, encoding="utf-8")], dtype="uint16")


@final
class Word:
    __slots__ = ("tokens", "freq")

    def __init__(self, tokens: Iterable[int], freq: int = 1) -> None:
        self.tokens: np.ndarray = np.array(tokens, dtype="uint16")
        self.freq = freq

    # Checks if pair found at pos
    def found_at(self, pos: int, a: int, b: int) -> bool:
        if pos < self.tokens.size - 1 and self.tokens[pos] == a and self.tokens[pos + 1] == b:
            return True
        return False

    def pairs(self) -> list[PackedPair]:
        return [(int(a) << 16) | int(b) for a, b in zip(self.tokens, self.tokens[1:])]  # pyright: ignore[reportAny]

    # Merge the pair if found
    def merge(self, pair: PackedPair, ab: int) -> PackedPairCount:
        a, b = unpack_pair(pair)
        i, updates = 0, PackedPairCount()
        new_tokens: list[int] = []
        while i < self.tokens.size:
            if self.found_at(i, a, b):
                if new_tokens:
                    prev = int(self.tokens[i - 1])
                    updates[pack_pair(prev, a)] -= self.freq
                    updates[pack_pair(prev, ab)] += self.freq
                updates[pair] -= self.freq
                new_tokens.append(ab)

                while self.found_at(i + 2, a, b):
                    updates[pack_pair(ab, ab)] += self.freq
                    updates[pack_pair(a, b)] -= self.freq
                    updates[pack_pair(b, a)] -= self.freq
                    new_tokens.append(ab)
                    i += 2

                look = i + 2
                if look < self.tokens.size:
                    nxt = int(self.tokens[look])
                    updates[pack_pair(b, nxt)] -= self.freq
                    updates[pack_pair(ab, nxt)] += self.freq
                i += 2
            else:
                new_tokens.append(int(self.tokens[i]))
                i += 1

        self.tokens = np.array(new_tokens, dtype="uint16")
        return updates
