from dataclasses import dataclass

import numpy as np
from typing import final
from collections.abc import Iterable
from .types import IdPairCount, MergeResult, Vocab, Merges, Pair, PairCount, IdPair


# Converts string to tuple of bytes
def to_bytes(string: str, offset: int = 0) -> np.ndarray:
    return np.array([b + offset for b in bytes(string, encoding="utf-8")], dtype='uint16')

def found_at(tokens: np.ndarray, pos: int, pair: IdPair) -> bool:
    return pos < tokens.size - 1 and tokens[pos] == pair[0] and tokens[pos + 1] == pair[1]

class NewTokens:
  def __init__(self, length: int) -> None:
    self.pos : int = 0
    self.tokens: np.ndarray = np.empty(length, dtype="uint16")

  def is_empty(self) -> bool:
    return self.pos == 0

  def add(self, token: int) -> None:
    self.tokens[self.pos] = token
    self.pos += 1

  def copy(self) -> np.ndarray:
    return self.tokens[:self.pos]

@final
class NewWord:
    # Word contains id tokens and frequency of its appearance,
    # tokens offsetted by number special_tokens
    def __init__(self, tokens: Iterable[int], freq: int = 1) -> None:
        self.tokens: np.ndarray = np.array(tokens, dtype='uint16')
        self.freq = freq

    # Checks if pair found at pos
    def found_at(self, pos: int, pair: IdPair) -> bool:
        return found_at(self.tokens, pos, pair)

    def pairs(self) -> list[IdPair]:
        return list(zip(self.tokens, self.tokens[1:]))

    # Merge the pair if found
    def merge(self, pair: IdPair, ab: int) -> IdPairCount:
        a, b = pair
        i, j, updates = 0, 0, IdPairCount()
        new_tokens = np.empty(len(self.tokens), dtype='uint16')
        while i < len(self.tokens):
            if self.found_at(i, pair):
                if j > 0: #new_tokens:
                    prev = int(self.tokens[i - 1])
                    updates[(prev, a)] -= self.freq
                    updates[(prev, ab)] += self.freq
                updates[pair] -= self.freq
                new_tokens[j] = ab
                j += 1

                while self.found_at(i + 2, pair):
                    updates[(ab, ab)] += self.freq
                    updates[(a, b)] -= self.freq
                    updates[(b, a)] -= self.freq
                    new_tokens[j] = ab
                    j += 1
                    i += 2

                look = i + 2
                if look < len(self.tokens):
                    nxt = int(self.tokens[look])
                    updates[(b, nxt)] -= self.freq
                    updates[(ab, nxt)] += self.freq
                i += 2
            else:
                new_tokens[j] = self.tokens[i]
                j += 1
                i += 1

        self.tokens = new_tokens[:j].copy()
        return updates
