import json
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, Self, final, cast

from .types import EncodedPair, encode_pair, decode_pair


class EncodedVocab(NamedTuple):
    vocab: dict[str, str]
    merges: list[list[str]]


def save(path: Path | str, vocab: EncodedVocab) -> None:
    with open(path, "w") as f:
        json.dump(vocab, f, indent=4)


def load(path: Path | str) -> EncodedVocab:
    with open(path) as f:
        return EncodedVocab(*json.load(f))  # pyright: ignore[reportAny]


def decode_latin(b: bytes) -> str:
    return b.decode("latin1")


def encode_latin(s: str) -> bytes:
    return s.encode("latin1")


class Vocab:
    def __init__(self, special_tokens: list[str] | None = None) -> None:
        self._forward: dict[int, bytes] = {}
        self._reverse: dict[bytes, int] = {}
        self._merges: list[EncodedPair] = []
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

    def add_merge(self, pair: EncodedPair) -> int:
        ab = self.to_bytes(decode_pair(pair))
        self._merges.append(pair)
        return self.add(ab)

    def bytes_for(self, token_id: int) -> bytes:
        return self._forward[token_id]

    def id_for(self, token: bytes) -> int:
        return self._reverse[token]

    def to_ids(self, string: str) -> Iterable[int]:
        return tuple([self.id_for(bytes([b])) for b in string.encode("utf-8")])

    def to_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join([self.bytes_for(b) for b in ids])

    def merges(self) -> list[EncodedPair]:
        return self._merges

    def __len__(self) -> int:
        return len(self._forward)

    def save(self, path: Path | str) -> None:
        save(
            path,
            EncodedVocab(
                {str(k): decode_latin(v) for k, v in self._forward.items()},
                [[decode_latin(self.bytes_for(b)) for b in decode_pair(pair)] for pair in self._merges],
            ),
        )

    @classmethod
    def load(cls, path: Path | str) -> Self:
        return cls.from_dict(load(path))  # pyright: ignore

    @classmethod
    def from_dict(cls, source: EncodedVocab) -> Self:
        vocab = cls.__new__(cls)
        vocab._forward = {int(k): encode_latin(v) for k, v in source.vocab.items()}
        vocab._reverse = {v: k for k, v in vocab._forward.items()}
        vocab._merges = [
            encode_pair(tuple(vocab.id_for(encode_latin(pair[0])), vocab.id_for(encode_latin(pair[0]))))
                for pair in source.merges]
        vocab._next_id = max(vocab._forward) + 1
        return vocab

@final
class VocabCodec:
    def __init__(self, vocab: Vocab) -> None:
        self._vocab = vocab

    def encode(self, bytez: bytes) -> list[int]:
        return [self._vocab.id_for(bytes([b])) for b in bytez]

    def decode(self, ids: Iterable[int]) -> bytes:
        return b"".join([self._vocab.bytes_for(b) for b in ids])

@final
class TieBreaker:
    def __init__(self, vocab: Vocab) -> None:
        self._vocab = vocab

    def tuple_for(self, pair: tuple[int, int]) -> tuple[bytes, ...]:
        return tuple([self._vocab.bytes_for(b) for b in pair])

    def lex_greater(self, first: EncodedPair, second: EncodedPair) -> bool:
        return self.tuple_for(decode_pair(first)) > self.tuple_for(decode_pair(second))
