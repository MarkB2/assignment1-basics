from collections import Counter, defaultdict
import json
from pathlib import Path
import gc
from .types import Pair, Vocab, Merges

from .reader import Reader, Pattern, GPT4_PAT
from .max_heap import PairMaxHeap

class BPETrainer:
  def __init__(self,
    source:str | Path,
    vocab_size: int,
    special_tokens: list[str],
    pat: str = GPT4_PAT
  ) -> None:
    self.source:str | Path = source
    self.vocab_size: int = vocab_size
    self.pat: Pattern = Pattern(pat=pat, special_tokens=special_tokens)

  def setup(self) -> tuple[Vocab, Merges]:
    vocab = Vocab()
    for i, spec_tok in enumerate(self.pat.special_tokens):
      vocab[i] = bytes(spec_tok, encoding='utf-8')
    i = len(vocab)
    for j in range(256):
      vocab[i+j] = bytes([j])
    return vocab, Merges()

  def merge(self) -> tuple[Vocab, Merges]:
    words, pair_counts, pair_locs = Reader(self.source, self.pat).build()
    _ = gc.collect()
    heap, (vocab, merges) = PairMaxHeap(pair_counts), self.setup()
    i, pair = len(vocab), heap.get_best()
    while pair and i < self.vocab_size:
      global_updates: Counter[Pair] = Counter()
      merges.append(pair)
      a, b = pair
      vocab[i] = a + b
      for loc in list(pair_locs[pair]):
        updates = words[loc].merge(pair)
        for updated_pair in updates.keys():
            if updated_pair in words[loc].pairs():
                pair_locs[updated_pair].add(loc)
            else:
                pair_locs[updated_pair].discard(loc)
        global_updates.update(updates)
      heap.update(global_updates)
      i += 1
      pair = heap.get_best()
      if i % 5000 == 0:
        keys = [k for k, v in pair_counts.items() if v<=0]
        for key in keys:
          del pair_counts[key]
        keys = [k for k, v in pair_locs.items() if v == set()]
        for key in keys:
          del pair_locs[key]
        _ = gc.collect()

    return vocab, merges

# Recursively convert all bytes to latin-1 strings
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(x) for x in obj] # JSON arrays become lists
    elif isinstance(obj, bytes):
        return obj.decode('latin-1') # Safe for ALL binary data
    return obj

def serialize(obj, file_name):
  with open(file_name, 'w') as file:
    json.dump(make_json_safe(obj), file, indent=4)

def train(input_path: str, vocab_size: int = 32_000,
    special_tokens: list[str] = ['<|endoftext|>']) -> tuple[dict[int, bytes], list[Pair]]:
  return BPETrainer(Path(input_path), vocab_size, special_tokens).merge()
