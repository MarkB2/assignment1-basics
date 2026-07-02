from pathlib import Path
import regex as re
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from collections.abc import Iterator

from .types import PairCount, PairLoc, Pretoken, SpecialToken
from .tokens import to_bytes, Word

GPT4_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass
class Pattern:
  pat : str = GPT4_PAT
  special_tokens: list[str] = field(default_factory=lambda: [])
  p: str = ""
  compiled : re.Pattern[str] = field(init=False, repr=False)

  def __post_init__(self):
    if not self.special_tokens:
      self.special_tokens = ["<|endoftext|>"]
    self.special_tokens.sort(key=lambda x: len(x), reverse=True)
    exclude = "|".join([re.escape(b) for b in self.special_tokens])
    self.p = rf"({exclude})|({self.pat})"
    self.compiled = re.compile(rf"({exclude})|({self.pat})")

def choose_num_chunks(file_size: int, max_chunk_size: int = 500 * 1024 * 1024) -> int:
  return max(1, -(-file_size // max_chunk_size))  # ceil div, no cpu cap needed

def find_chunk_boundaries(file, special_token: bytes = b"<|endoftext|>") -> list[int]:
    file.seek(0, 2)
    file_size = file.tell()
    num_chunks = choose_num_chunks(file_size)
    chunk_size = file_size // num_chunks

    boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    boundaries[-1] = file_size

    for i in range(1, len(boundaries) - 1):
        pos = boundaries[i]
        file.seek(pos)
        # read forward in a small window until we find the special token
        window = 4096
        while True:
            chunk = file.read(window)
            if not chunk:
                boundaries[i] = file_size
                break
            found = chunk.find(special_token)
            if found != -1:
                boundaries[i] = pos + found
                break
            pos += window
            file.seek(pos)

    return sorted(set(boundaries))

def reader(source: str | Path, pat: Pattern) -> Iterator[str]:
  if isinstance(source, Path):
    with open(source, 'rb') as file:
      boundaries = find_chunk_boundaries(file)
      # boundaries = [0, boundaries[-1]]
      for start, end in zip(boundaries, boundaries[1:]):
        print(f"start, end {start, end}")
        file.seek(start)
        chunk_bytes = file.read(end - start)
        yield from reader(chunk_bytes.decode('utf-8', errors='ignore'), pat)
  else:
    for  match in pat.compiled.finditer(source):
      if match.group(2):
        yield match.group(2)

class Reader:
  def __init__(self, source: str | Path, pat: Pattern) -> None:
    self.source = source
    self.pat = pat

  @classmethod
  def get_counts(cls, words: list[Word]) -> tuple[PairCount, PairLoc]:
    pair_counts, pair_locs = PairCount(), PairLoc()
    for loc, word in enumerate(words):
      for pair in word.pairs():
        pair_counts[pair] += word.freq
        pair_locs[pair].add(loc)
    return pair_counts, pair_locs

  def build(self) -> tuple[list[Word], PairCount, PairLoc]:
    corpus = Counter(reader(self.source, self.pat))
    words = [Word(word, freq) for word, freq in corpus.items()]
    pair_counts, pair_locs = self.get_counts(words)
    return words, pair_counts, pair_locs

class Pretokenizer:
  def __init__(self, source: str | Path, pat: str = GPT4_PAT, special_tokens: list[str] | None = None, keep_special_tokens: bool = False) -> None:
    self.source: str | Path = source
    self.pat: re.Pattern[str] = re.compile(pat)
    special_tokens = special_tokens or ["<|endoftext|>"]
    self.special_tokens: set[str] = set(special_tokens)
    self.split_pat: re.Pattern[str] = self.split_pattern(special_tokens)
    self.keep_special_tokens: bool = keep_special_tokens

  @classmethod
  def split_pattern(cls, special_tokens: list[str]) -> re.Pattern[str]:
    special_tokens.sort(key=lambda x: len(x), reverse=True)
    split_pat = "|".join([re.escape(b) for b in special_tokens])
    return re.compile(rf"({split_pat})")

  def _read(self, source: str) -> Iterator[Pretoken]:
    for part in re.split(self.split_pat, source):
      if part in self.special_tokens:
       if self.keep_special_tokens:
        yield SpecialToken(part)
      else:
        for match in re.finditer(self.pat, part):
            yield match.group()

  def read(self) -> Iterator[Pretoken]:
    if isinstance(self.source, Path):
      with open(self.source, 'rb') as file:
        boundaries = find_chunk_boundaries(file)
        # boundaries = [0, boundaries[-1]]
        for start, end in zip(boundaries, boundaries[1:]):
          print(f"start, end {start, end}")
          file.seek(start)
          chunk_bytes = file.read(end - start)
          yield from self._read(chunk_bytes.decode('utf-8', errors='ignore'))
    else:
      yield from self._read(self.source)

  @classmethod
  def get_counts(cls, words: list[Word]) -> tuple[PairCount, PairLoc]:
    pair_counts, pair_locs = PairCount(), PairLoc()
    for loc, word in enumerate(words):
      for pair in word.pairs():
        pair_counts[pair] += word.freq
        pair_locs[pair].add(loc)
    return pair_counts, pair_locs

  def build(self) -> tuple[list[Word], PairCount, PairLoc]:
    corpus: Counter[Pretoken] = Counter(self.read())
    words = [Word(word, freq) for word, freq in corpus.items()]
    pair_counts, pair_locs = self.get_counts(words)
    return words, pair_counts, pair_locs
