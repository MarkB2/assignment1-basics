import numpy as np
from typing import final
from collections.abc import Iterable
from .types import IdPairCount, MergeResult, Vocab, Merges, Pair, PairCount, IdPair


# Converts string to tuple of bytes
def to_bytes(string: str, offset: int = 0) -> np.ndarray:
    return np.array([b + offset for b in bytes(string, encoding="utf-8")], dtype='uint16')

def found_at(tokens: np.ndarray, pos: int, pair: IdPair) -> bool:
    return pos < tokens.size - 1 and tokens[pos] == pair[0] and tokens[pos + 1] == pair[1]


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
        i, new_tokens, updates = 0, [], IdPairCount()
        while i < len(self.tokens):
            if self.found_at(i, pair):
                if new_tokens:
                    prev = int(self.tokens[i - 1])
                    updates[(prev, a)] -= self.freq
                    updates[(prev, ab)] += self.freq
                updates[pair] -= self.freq
                new_tokens.append(ab)

                while self.found_at(i + 2, pair):
                    updates[(ab, ab)] += self.freq
                    updates[(a, b)] -= self.freq
                    updates[(b, a)] -= self.freq
                    new_tokens.append(ab)
                    i += 2

                look = i + 2
                if look < len(self.tokens):
                    nxt = int(self.tokens[look])
                    updates[(b, nxt)] -= self.freq
                    updates[(ab, nxt)] += self.freq
                i += 2
            else:
                new_tokens.append(self.tokens[i])
                i += 1

        self.tokens = np.array(new_tokens, dtype='uint16')
        return updates
