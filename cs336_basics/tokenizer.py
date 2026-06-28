from __future__ import annotations
from pathlib import Path
from collections.abc import Iterable, Iterator
import json
from cs336_basics.types import Pair

def save(path: Path | str, obj: dict[int, str] | list[list[str]]) -> None:
  with open(path, 'w') as f:
    json.dump(obj, f)

def load(path: Path | str) -> dict[int, str] | list[list[str]]:
  with open(path) as f:
    return json.load(f) # pyright: ignore[reportAny]

class Vocab(dict[int, bytes]):
  def save(self, path: Path | str) -> None:
    save(path, {k:v.decode('latin1') for k, v in self.items()})

  @classmethod
  def load(cls, path: Path | str):
    return Vocab({int(k):v.encode('latin1') for k, v in load(path).items()}) # pyright: ignore

class Merges(list[Pair]):
  def save(self, path: Path | str) -> None:
    save(path, [[s.decode('latin1') for s in pair] for pair in self])

  @classmethod
  def load(cls, path: Path | str):
    return Merges([Pair([s.encode('latin1') for s in pair]) for pair in load(path)]) # pyright: ignore

class Tokenizer:
  def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
    pass

  @classmethod
  def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
    ...

  def encode(self, text: str) -> list[int]:
    ...

  def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
    ...

  def decode(self, tokens: list[int]) -> str:
    ...
