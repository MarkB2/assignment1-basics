from __future__ import annotations
from pathlib import Path
from collections.abc import Iterable, Iterator
from typing import Self, final
from cs336_basics.tokens import Word
from cs336_basics.types import BytesVocab
from cs336_basics.vocab import Vocab, bytes_vocab_to_vocab, load_vocab
# from cs336_basics.types import Pair, Vocab, Merges

@final
class Tokenizer:
  def __init__(self, vocab: Vocab, special_tokens: list[str] | None = None):
      self.vocab: Vocab = vocab
      for special_token in special_tokens or []:
          _ = self.vocab.add_special_token(special_token)

  @classmethod
  def from_file(cls, vocab_filepath: str | Path, special_tokens: list[str] | None = None) -> Self:
      return cls(load_vocab(vocab_filepath), special_tokens)

  @classmethod
  def from_dicts(cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> Self:
      return cls(bytes_vocab_to_vocab(BytesVocab(vocab, merges)), special_tokens)

  # def _encode(self, word:Word) -> list[int]:
  #   ids:list[int] = []
  #   pos, tokens = 0, word.tokens
  #   while pos < len(tokens):
  #     prev = tokens[pos]
  #     id, pos = self._lookup(tokens, prev, pos+1)
  #     ids.append(id)
  #   return ids

  # def encode(self, text: str) -> list[int]:
  #   return self._encode(Word(text))

  # def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
  #   ...

  # def decode(self, tokens: list[int]) -> str:
  #   return b"".join(self.vocab[id] for id in tokens).decode("utf-8", errors='replace')
