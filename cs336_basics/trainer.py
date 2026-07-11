from collections import Counter, defaultdict
from collections.abc import Iterator
import json
from pathlib import Path
import gc
from .types import Pair, Vocab, Merges, Pretoken, PairCount, PairLoc
from .tokens import Word

from .reader import Reader, Pattern, GPT4_PAT, pretokenize
from .max_heap import PairMaxHeap

class BPETrainer:
  def __init__(self,
    iterator: Iterator[Pretoken],
    vocab_size: int,
    special_tokens: list[str],
  ) -> None:
    self.iterator: Iterator[Pretoken] = iterator
    self.vocab_size: int = vocab_size
    vocab = Vocab({i: bytes(special_token, encoding='utf-8') for i, special_token in enumerate(special_tokens)})
    for j in range(256):
      vocab[len(vocab)] = bytes([j])
    self.vocab: Vocab = vocab
    self.merges: Merges = Merges()

  def build(self) -> tuple[list[Word], PairCount, PairLoc]:
      corpus: Counter[Pretoken] = Counter(self.iterator)
      words = [Word(word, freq) for word, freq in corpus.items()]
      pair_counts, pair_locs = self.get_counts(words)
      return words, pair_counts, pair_locs

  @classmethod
  def get_counts(cls, words: list[Word]) -> tuple[PairCount, PairLoc]:
      pair_counts, pair_locs = PairCount(), PairLoc()
      for loc, word in enumerate(words):
          for pair in word.pairs():
              pair_counts[pair] += word.freq
              pair_locs[pair].add(loc)
      return pair_counts, pair_locs

  # def setup(self) -> tuple[Vocab, Merges]:
  #   vocab = Vocab()
  #   for i, spec_tok in enumerate(self.pat.special_tokens):
  #     vocab[i] = bytes(spec_tok, encoding='utf-8')
  #   i = len(vocab)
  #   for j in range(256):
  #     vocab[i+j] = bytes([j])
  #   return vocab, Merges()

  def merge(self) -> tuple[Vocab, Merges]:
    words, pair_counts, pair_locs = self.build()
    _ = gc.collect()
    heap, vocab, merges = PairMaxHeap(pair_counts), self.vocab, self.merges
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

def train(input_path: str | Path, vocab_size: int = 32_000,
    special_tokens: list[str] = ['<|endoftext|>']) -> tuple[dict[int, bytes], list[Pair]]:
  iterator = pretokenize(input_path, special_tokens=special_tokens)
  return BPETrainer(iterator, vocab_size, special_tokens).merge()
