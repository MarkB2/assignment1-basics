from collections.abc import Iterable
import json
from pathlib import Path
from typing import Self

from .types import IdPair


def save(path: Path | str, obj: dict[int, str] | list[list[str]]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load(path: Path | str) -> dict[int, str] | list[list[str]]:
    with open(path) as f:
        return json.load(f)  # pyright: ignore[reportAny]


class Vocab:
    def __init__(self, special_tokens: list[str] | None = None) -> None:
        self._forward: dict[int, bytes] = {}
        self._reverse: dict[bytes, int] = {}
        self._next_id: int = 0
        for tok in special_tokens or []:
            _ = self.add(tok.encode("utf-8"))
        for i in range(256):
            _ = self.add(bytes([i]))

    def add(self, token: bytes):
        token_id = self._next_id
        self._forward[token_id] = token
        self._reverse[token] = token_id
        self._next_id += 1
        return token_id

    def add_special_token(self, token: str) -> int:
        tok_bytes = token.encode("utf-8")
        if tok_bytes in self._reverse:
            return self._reverse[tok_bytes]
        return self.add(tok_bytes)

    def add_merge(self, pair: IdPair) -> int:
        ab = self.to_bytes(pair)
        return self.add(ab)

    def bytes_for(self, token_id: int) -> bytes:
        return self._forward[token_id]

    def id_for(self, token: bytes) -> int:
        return self._reverse[token]

    def to_tokens(self, string: str) -> Iterable[int]:
        return tuple([self.id_for(bytes([b])) for b in string.encode("utf-8")])

    def to_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join([self.bytes_for(b) for b in ids])

    def __len__(self) -> int:
        return len(self._forward)

    def save(self, path: Path | str) -> None:
        save(path, {k: v.decode("latin1") for k, v in self._forward.items()})

    @classmethod
    def load(cls, path: Path | str) -> Self:
        return cls.from_dict({int(k): v.encode("latin1") for k, v in load(path).items()})  # pyright: ignore

    @classmethod
    def from_dict(cls, initial: dict[int, bytes]) -> Self:
        vocab = cls.__new__(cls)
        vocab._forward = initial.copy()
        vocab._reverse = {v: k for k, v in vocab._forward.items()}
        vocab._next_id = max(vocab._forward) + 1
        return vocab


class Merges(list[IdPair]):
    def save(self, path: Path | str, vocab: Vocab | None = None) -> None:
        vocab = vocab or Vocab()
        save(path, [[s.decode("latin1") for s in pair] for pair in self])

    @classmethod
    def load(cls, path: Path | str):
        return Merges([Pair([s.encode("latin1") for s in pair]) for pair in load(path)])  # pyright: ignore
