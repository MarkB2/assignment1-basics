from collections import Counter, defaultdict
import json
from pathlib import Path

from tests.tokenizer.test_tokens import test_pairs
from .reader import Reader, Pattern, GPT4_PAT
from .max_heap import PairMaxHeap
from .tokens import Word, Pair

class BPETrainer:
  def __init__(self,
    source:str | Path,
    vocab_size: int,
    special_tokens: list[str],
    pat: str = GPT4_PAT
  ) -> None:
    self.source = source
    self.vocab_size = vocab_size
    self.pat = Pattern(pat=pat, special_tokens=special_tokens)

  def setup_vocab(self) -> dict[int, bytes]:
    vocab = {}
    for i, spec_tok in enumerate(self.pat.special_tokens):
      vocab[i] = bytes(spec_tok, encoding='utf-8')
    i = len(vocab)
    for j in range(256):
      vocab[i+j] = bytes([j])
    return vocab

  def merge(self) -> tuple[dict[int, bytes], list[Pair]]:
    words, pair_counts, pair_locs = Reader(self.source, self.pat).build()
    heap, vocab, merges = PairMaxHeap(pair_counts), self.setup_vocab(), []
    i, pair = len(vocab), heap.get_best()
    while pair and i < self.vocab_size:
      global_updates = Counter()
      merges.append(pair)
      #############
      #
      if pair == (b' g', b'ive'):
        print(f"i={i} pair={pair} count={pair_counts[pair]} count_lflf={pair_counts[(b'\n', b'\n')]}")
        real_pair_counts, real_pair_locs = Reader.get_counts(words)
        print(f"i={i} pair={pair} real_count={real_pair_counts[pair]} real_count_lflf={real_pair_counts[(b'\n', b'\n')]}")
      #
      #############
      a, b = pair
      vocab[i] = a + b
      for loc in list(pair_locs[pair]):
        updates = words[loc].merge(pair)
        for updated_pair, delta in updates.items():
            if updated_pair in words[loc].pairs():
                pair_locs[updated_pair].add(loc)
            else:
                pair_locs[updated_pair].discard(loc)
        global_updates.update(updates)
      heap.update(global_updates)
      i += 1
      pair = heap.get_best()
      #############
      #
      if pair == (b' g', b'ive'):
        print(f"i={i} pair={pair} count={pair_counts[pair]} count_lflf={pair_counts[(b'\n', b'\n')]}")
      #
      #############
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
