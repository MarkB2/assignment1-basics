from __future__ import annotations
from pathlib import Path
from collections.abc import Iterable, Iterator
from typing import Self, final
from cs336_basics.tokens import Word
from cs336_basics.types import Pair, Vocab, Merges

@final
class Tokenizer:
  def __init__(self, vocab: Vocab, merges: Merges, special_tokens: list[str] | None = None):
    if special_tokens is not None:
      for special_token in special_tokens:
        encoded_special_token = special_token.encode("utf-8")
        if encoded_special_token not in set(vocab.values()):
          vocab[len(vocab)] = encoded_special_token
    self.vocab = vocab
    self.ids = {v:k for k,v in self.vocab.items()}
    self.merges = merges
    self.pairs = set(self.merges)
    self.special_tokens = special_tokens

  @classmethod
  def from_files(cls, vocab_filepath: str | Path, merges_filepath: str | Path, special_tokens: list[str] | None = None) -> Self:
    return cls(Vocab.load(vocab_filepath), Merges.load(merges_filepath), special_tokens)

  def _lookup(self, tokens:tuple[bytes, ...], prev:bytes, pos:int) -> tuple[int, int]:
    if pos < len(tokens) and Pair([prev, tokens[pos]]) in self.pairs:
      return self._lookup(tokens, prev + tokens[pos], pos+1)
    return self.ids[prev], pos

  def _encode(self, word:Word) -> list[int]:
    ids:list[int] = []
    pos, tokens = 0, word.tokens
    while pos < len(tokens) - 1:
      prev = tokens[pos]
      id, pos = self._lookup(tokens, prev, pos+1)
      ids.append(id)
    return ids

  def encode(self, text: str) -> list[int]:
    return self._encode(Word(text))

  def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
    ...

  def decode(self, tokens: list[int]) -> str:
    return b"".join(self.vocab[id] for id in tokens).decode("utf-8", errors='replace')
