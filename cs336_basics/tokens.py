from dataclasses import dataclass

import numpy as np
from typing import final
from collections.abc import Iterable
from .types import PackedPair, PackedPairCount, pack_pair, unpack_pair



# Converts string to tuple of bytes
def to_bytes(string: str, offset: int = 0) -> np.ndarray:
    return np.array([b + offset for b in bytes(string, encoding="utf-8")], dtype='uint16')


@final
class NewWord:
    # Word contains id tokens and frequency of its appearance,
    # tokens offsetted by number special_tokens
    __slots__ = ('tokens', 'freq')

    def __init__(self, tokens: Iterable[int], freq: int = 1) -> None:
        self.tokens: np.ndarray = np.array(tokens, dtype='uint16')
        self.freq = freq

    # Checks if pair found at pos
    def found_at(self, pos: int, a: int, b: int) -> bool:
        if pos < self.tokens.size - 1 and self.tokens[pos] == a and self.tokens[pos + 1] == b:
            return True
        return False

    def pairs(self) -> np.ndarray:
        return self.tokens.astype(dtype='uint32')[:-1] << 16 | self.tokens[1:]


    # Merge the pair if found
    def merge(self, pair: PackedPair, ab: int) -> PackedPairCount:
        a, b = unpack_pair(pair)
        i, j, updates = 0, 0, PackedPairCount()
        new_tokens = np.empty_like(self.tokens)
        while i < self.tokens.size:
            if self.found_at(i, a, b):
                if j > 0: #new_tokens:
                    prev = int(self.tokens[i - 1])
                    updates[pack_pair(prev, a)] -= self.freq
                    updates[pack_pair(prev, ab)] += self.freq
                updates[pair] -= self.freq
                new_tokens[j] = ab
                j += 1

                while self.found_at(i + 2, a, b):
                    updates[pack_pair(ab, ab)] += self.freq
                    updates[pack_pair(a, b)] -= self.freq
                    updates[pack_pair(b, a)] -= self.freq
                    new_tokens[j] = ab
                    j += 1
                    i += 2

                look = i + 2
                if look < self.tokens.size:
                    nxt = int(self.tokens[look])
                    updates[pack_pair(b, nxt)] -= self.freq
                    updates[pack_pair(ab, nxt)] += self.freq
                i += 2
            else:
                new_tokens[j] = self.tokens[i]
                j += 1
                i += 1

        self.tokens = new_tokens[:j].copy()
        return updates
