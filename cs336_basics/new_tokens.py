import numpy as np
from typing import final
from .types import Vocab, Merges, Pair, PairCount


# Converts string to tuple of bytes
def to_bytes(string: str) -> np.ndarray:
    return np.array([b for b in bytes(string, encoding="utf-8")], dtype='uint16')


@final
class NewWord:
    # Word contains bytes tokens and frequency of its appearance
    def __init__(self, string: str, freq: int = 1) -> None:
        self.tokens: np.ndarray = to_bytes(string)
        self.freq = freq

    # Checks if pair found at pos
    def found_at(self, pos: int, pair: Pair) -> bool:
        return self.tokens[pos : pos + 2] == pair

    def pairs(self) -> list[Pair]:
        return list(zip(self.tokens, self.tokens[1:]))

    # Merge the pair if found
    def merge(self, pair: Pair) -> PairCount:
        a, b = pair
        ab = a + b
        i, new_tokens, updates = 0, [], PairCount()
        while i < len(self.tokens):
            if self.found_at(i, pair):
                if new_tokens:
                    prev = self.tokens[i - 1]
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
                    nxt = self.tokens[look]
                    updates[(b, nxt)] -= self.freq
                    updates[(ab, nxt)] += self.freq
                i += 2
            else:
                new_tokens.append(self.tokens[i])
                i += 1

        self.tokens = tuple(new_tokens)
        return updates
