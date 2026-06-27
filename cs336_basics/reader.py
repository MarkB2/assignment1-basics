from pathlib import Path
import regex as re
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from collections.abc import Iterator

from .types import PairCount, PairLoc
from .tokens import Pair, Word

GPT4_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass
class Pattern:
  pat : str = GPT4_PAT
  special_tokens: list[str] = field(default_factory=lambda: [])
  compiled : re.Pattern = field(init=False, repr=False)

  def __post_init__(self):
    if not self.special_tokens:
      self.special_tokens = ["<|endoftext|>"]
    self.special_tokens.sort(key=lambda x: len(x), reverse=True)
    exclude = "|".join([re.escape(b) for b in self.special_tokens])
    self.compiled = re.compile(rf"({exclude})|({self.pat})")

def reader(source: str | Path, pat: Pattern) -> Iterator[str]:
  if isinstance(source, Path):
    with open(source, encoding='utf-8') as file:
      for line in file:
        yield from reader(line, pat)
  else:
    for  match in pat.compiled.finditer(source):
      if match.group(2):
        yield match.group(2)

class Reader:
  def __init__(self, source: str | Path, pat: Pattern) -> None:
    self.source = source
    self.pat = pat

  def build(self) -> tuple[list[Word], PairCount, PairLoc]:
    corpus = Counter(reader(self.source, self.pat))
    words, pair_counts, pair_locs = [], Counter(), defaultdict(set)
    for loc, (name, freq) in enumerate(corpus.items()):
      word = Word(name, freq)
      words.append(word)
      for pair in word.pairs():
        pair_counts[pair] += freq
        pair_locs[pair].add(loc)
    return words, pair_counts, pair_locs
