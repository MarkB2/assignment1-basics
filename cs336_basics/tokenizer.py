from pathlib import Path
from collections.abc import Iterable, Iterator
import json

def deserialize(obj):
  if isinstance(obj, Path):
    with open(obj) as f:
      return deserialize(json.load(f))
  if isinstance(obj, tuple):
    return tuple(deserialize(item) for item in obj)
  if isinstance(obj, list):
    return tuple([deserialize(item) for item in obj])
  if isinstance(obj, dict):
    return {int(k): deserialize(v) for k, v in obj.items()}
  if isinstance(obj, str):
    return bytes(obj, encoding='latin-1')
  return obj


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
