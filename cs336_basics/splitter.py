import regex as re
from collections import Counter, defaultdict
from typing import Iterator
from .pretokenization_example import pre_file_reader, pre_string_reader
from .maxheap import PairMaxHeap
from .merge import to_pairs, merge

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def build_pattern(base: str, special_tokens: list[str]) -> re.Pattern:
    # base = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if not special_tokens:
        return re.compile(f"($^)|" + base)  # ($^) never matches anything
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    special_part = "|".join(re.escape(tok) for tok in sorted_specials)
    return re.compile(f"({special_part})|" + base)

def string_iter(text: str, pat: re.Pattern[str]):
  for match in pat.finditer(text):
    yield match.group()

def file_iter(input_path: str, pat: re.Pattern[str]):
  with open(input_path, encoding='utf-8') as file:
    for line in file:
      for match in pat.finditer(line):
        if match.group(1):
          continue
        yield match.group(0)


def to_bytes(string: str | list) ->tuple[bytes]:
  if isinstance(string, list):
    return [to_bytes(elm) for elm in string]
  return tuple(bytes([b]) for b in string.encode('utf-8'))

class PreTokenizer:
  def __init__(self, pre_tokens: Counter):
    tokens, self.counts = zip(*pre_tokens.items())
    self.tokens = [to_bytes(token) for token in tokens]
    self.pair_freq, self.pair_loc = Counter(), defaultdict(set)
    for i, (token, count) in enumerate(zip(self.tokens, self.counts)):
      for pair in to_pairs(token):
        self.pair_freq[pair] += count
        self.pair_loc[pair].add(i)
    
  def get_counts(self):
    return self.tokens, self.counts, self.pair_freq, self.pair_loc
  
  @classmethod
  def from_file(cls, input_path: str, pat: str = PAT, 
                  special_tokens: list[str] = []):
    pre_tokens = Counter()
    for string in pre_file_reader(input_path, special_tokens):
      next = Counter(string_iter(string, pat))
      pre_tokens += next
    return cls(pre_tokens)

  @classmethod
  def from_string(cls, text: str, pat: re.Pattern = PAT, 
                  special_tokens: list[str] = []):
    pre_tokens = Counter()
    for string in pre_string_reader(str, special_tokens):
      next = Counter(string_iter(string, pat))
      pre_tokens += next
    return cls(pre_tokens)

class BPETrainer:
  def __init__(self, pre_tokenizer: PreTokenizer, vocab_size=32_000, special_tokens=[]):
    self.pre_tokenizer = pre_tokenizer
    self.vocab_size = vocab_size
    self.merges = []
    self.vocab = ['<|endoftext|>'] + [
      chr(a).encode('utf-8') for a in range(256)]


  def train(self):
    tokens, counts, pair_freq, pair_loc = self.pre_tokenizer.get_counts()
    heap = PairMaxHeap(pair_freq)

    for _ in range(256, self.vocab_size): # no more than vocab_size
      best = heap.get_best()
      if not best:
        break

      deltas = Counter()
      for loc in pair_loc.pop(best, []):
        new_token, updates = merge(tokens[loc], best, counts[loc])
        if updates:
          tokens[loc] = new_token
          deltas.update(updates)
          
          for pair, delta in updates.items():
            if delta > 0:
              pair_loc[pair].add(loc)
            else:
              pair_loc[pair].discard(loc)
      
      heap.update(deltas)
      self.merges.append(best)

    self.vocab += self.merges

    


          


              