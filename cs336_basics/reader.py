import regex as re
from dataclasses import dataclass, field
from collections import Counter
from collections.abc import Iterator
from typing import Self
from .tokens import Word

GPT4_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass
class Pattern:
  pat : str = GPT4_PAT
  special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
  compiled : re.Pattern = field(init=False, repr=False)

  def __post_init__(self):
    self.special_tokens.sort(key=lambda x: len(x), reverse=True)
    exclude = "|".join([re.escape(b) for b in self.special_tokens])
    self.compiled = re.compile(rf"({exclude})|({self.pat})")

def string_reader(string: str, pat: Pattern) -> Iterator[str]:
  for  match in pat.compiled.finditer(string):
    if match.group(2):
      yield match.group(2)

def file_reader(path: str, pat: Pattern) -> Iterator[str]:
  with open(path, encoding='utf-8') as file:
    for line in file:
      yield from string_reader(line, pat)

class Reader:
  def __init__(self, corpus: Counter[str]) -> None:
      self.corpus = corpus

  @classmethod
  def from_string(cls, string: str, pat: Pattern) -> Self:
    return cls(Counter(string_reader(string, pat)))

  @classmethod
  def from_file(cls, path: str, pat: Pattern) -> Self:
    return cls(Counter(file_reader(path, pat)))

  def get_corpus(self) -> list[Word]:
    return [Word(string, freq) for (string, freq) in  self.corpus.items()]
