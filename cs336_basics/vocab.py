import json
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, Self, final

from .max_heap import MaxHeap
from .types import PackedPair, PackedPairCount, TieBreak, pack_pair, unpack_pair, BytesVocab, StringVocab



def decode_latin(b: bytes) -> str:
    return b.decode("latin1")


def encode_latin(s: str) -> bytes:
    return s.encode("latin1")


class Vocab:
    def __init__(self, special_tokens: list[str] | None = None) -> None:
        self._forward: dict[int, bytes] = {}
        self._reverse: dict[bytes, int] = {}
        self._merges: list[PackedPair] = []
        self._next_id: int = 0
        for tok in special_tokens or []:
            _ = self.add_special_token(tok)
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

    def add_merge(self, pair: PackedPair) -> int:
        ab = self.to_bytes(unpack_pair(pair))
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

    def merges(self) -> list[PackedPair]:
        return self._merges

    def __len__(self) -> int:
        return len(self._forward)


def vocab_to_bytes_vocab(vocab: Vocab) -> BytesVocab:
    return BytesVocab(
        vocab._forward,  # pyright: ignore[reportPrivateUsage]
        [tuple([vocab.bytes_for(b) for b in unpack_pair(pair)]) for pair in vocab._merges],  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    )


def vocab_to_string_vocab(vocab: Vocab) -> StringVocab:
    bytes_vocab = vocab_to_bytes_vocab(vocab)
    return StringVocab(
        {str(k): decode_latin(v) for k, v in bytes_vocab.vocab.items()},
        [[decode_latin(b) for b in pair] for pair in bytes_vocab.merges],
    )


def string_vocab_to_vocab(source: StringVocab) -> Vocab:
    vocab = Vocab.__new__(Vocab)
    vocab._forward = {int(k): encode_latin(v) for k, v in source.vocab.items()}
    vocab._reverse = {v: k for k, v in vocab._forward.items()}
    vocab._merges = [pack_pair(*[vocab.id_for(encode_latin(a)) for a in pair]) for pair in source.merges]
    vocab._next_id = max(vocab._forward) + 1
    return vocab


def save_vocab(path: Path | str, vocab: Vocab) -> None:
    with open(path, "w") as f:
        json.dump(vocab_to_string_vocab(vocab), f, indent=4)


def load_vocab(path: Path | str) -> Vocab:
    with open(path) as f:
        return string_vocab_to_vocab(StringVocab(*json.load(f)))  # pyright: ignore[reportAny]


@final
class VocabCodec:
    def __init__(self, vocab: Vocab) -> None:
        self._vocab = vocab

    def encode(self, bytez: bytes) -> list[int]:
        return [self._vocab.id_for(bytes([b])) for b in bytez]

    def decode(self, ids: Iterable[int]) -> bytes:
        return b"".join([self._vocab.bytes_for(b) for b in ids])


class TieBrakeCache:
    def __init__(self) -> None:
        self._cache: dict[int, bool] = {}

    @staticmethod
    def _pack(a: PackedPair, b: PackedPair) -> int:
        return (a << 32) | b

    def _canonical_key(self, a: PackedPair, b: PackedPair) -> tuple[int, bool]:
        swapped = b > a
        return self._pack(a, b) if not swapped else self._pack(b, a), swapped

    def get(self, a: PackedPair, b: PackedPair, tie_break: TieBreak) -> bool:
        key, swapped = self._canonical_key(a, b)
        if key not in self._cache:
            self._cache[key] = tie_break(a, b) if not swapped else tie_break(b, a)
        return self._cache[key] if not swapped else not self._cache[key]

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key: int) -> bool:
        return key in self._cache


@final
class TieBreaker:
    def __init__(self, vocab: Vocab) -> None:
        self._vocab = vocab
        self._cache = TieBrakeCache()

    def greater(self, a: PackedPair, b: PackedPair) -> bool:
        return self._cache.get(a, b, self.lex_greater)

    def tuple_for(self, pair: tuple[int, int]) -> tuple[bytes, ...]:
        return tuple([self._vocab.bytes_for(b) for b in pair])

    def lex_greater(self, first: PackedPair, second: PackedPair) -> bool:
        return self.tuple_for(unpack_pair(first)) > self.tuple_for(unpack_pair(second))


def make_heap(pair_counts: PackedPairCount, vocab: Vocab) -> MaxHeap:
    return MaxHeap(pair_counts, TieBreaker(vocab).greater)
