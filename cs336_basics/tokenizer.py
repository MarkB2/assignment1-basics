from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Self, final

from .pretokenizer import Pretokenizer

from .tokens import Word
from .model_types import BytesVocab, PackedPair, Pretoken, SpecialToken, pack_pair, unpack_pair
from .vocab import Vocab, bytes_vocab_to_vocab, load_vocab


@final
class Tokenizer:
    def __init__(self, vocab: Vocab, special_tokens: list[str] | None = None):
        self.vocab: Vocab = vocab
        for special_token in special_tokens or []:
            _ = self.vocab.add_special_token(special_token)
        self.pair_index: dict[PackedPair, int] = {pair: index for index, pair in enumerate(self.vocab.merges())}
        self.pretokenizer = Pretokenizer(special_tokens=special_tokens, keep_special_tokens=True)

    @classmethod
    def from_file(cls, vocab_filepath: str | Path, special_tokens: list[str] | None = None) -> Self:
        return cls(load_vocab(vocab_filepath), special_tokens)

    @classmethod
    def from_dicts(
        cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> Self:
        return cls(bytes_vocab_to_vocab(BytesVocab(vocab, merges)), special_tokens)

    def best_pair(self, new_tokens: list[int]) -> tuple[int | None, int]:
        pos, best, best_rank = None, None, float("inf")
        for i, pair in enumerate([pack_pair(a, b) for a, b in zip(new_tokens, new_tokens[1:])]):
            rank = self.pair_index.get(pair)
            if rank is not None and rank < best_rank:
                pos, best, best_rank = i, pair, rank
        if best is None:
            return None, -1
        a, b = [self.vocab.bytes_for(x) for x in unpack_pair(best)]
        return pos, self.vocab.id_for(a + b)

    def pretokenize(self, pretoken: Pretoken) -> list[int]:
        if isinstance(pretoken, SpecialToken):
            return [self.vocab.id_for(pretoken.text.encode("utf8"))]
        return list(self.vocab.to_ids(pretoken))

    def _encode(self, new_tokens: list[int]) -> list[int]:
        while True:
            pos, replacement = self.best_pair(new_tokens)
            if pos is None:
                break
            new_tokens = new_tokens[:pos] + [replacement] + new_tokens[pos + 2 :]
        return new_tokens

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for pretoken in self.pretokenizer.iter_tokens(text):
            ids.extend(self._encode(self.pretokenize(pretoken)))
        return ids


    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)

    def decode(self, tokens: list[int]) -> str:
      return b"".join(self.vocab.bytes_for(id) for id in tokens).decode("utf-8", errors='replace')
