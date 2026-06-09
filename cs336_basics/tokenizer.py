from collections import defaultdict, Counter
import regex as re
from pair_heap import PairHeap

def build_pattern(special_tokens: list[str]) -> re.Pattern:
    base = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if not special_tokens:
        return re.compile(f"($^)|" + base)  # ($^) never matches anything
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    special_part = "|".join(re.escape(tok) for tok in sorted_specials)
    return re.compile(f"({special_part})|" + base)

def get_corpus_it(file_path, special_tokens):
  reg = build_pattern(special_tokens)
  with open(file_path, encoding='utf-8') as file:
    for line in file:
      for item in reg.finditer(line):
        if item.group(1):
          continue
        yield tuple(bytes([b]) for b in bytes(item.group(0), 'utf-8'))

def found_at(tokens, a, b, i):
  return i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b

def apply_merge(pair, vocab, freqs, pair_words, pair_heap):
  a, b = pair
  ab = a + b

  for word_id in list(pair_words.pop(pair, [])):
    tokens, freq, new_tokens, i = vocab[word_id], freqs[word_id], [], 0
    while(i < len(tokens)):
      if found_at(tokens, a, b, i):
        if new_tokens:
          prev = tokens[-1]
          pair_heap.update((prev, a), -freq)
          pair_heap.update((prev, ab), freq)
          pair_words[(prev, ab)].add(word_id)

        look = i + 2
        if look < len(tokens):
          nxt = tokens[look]
          pair_heap.update((b, nxt), -freq)
          pair_heap.update((ab, nxt), freq)
          pair_words[(ab, nxt)].add(word_id)

        new_tokens.append(ab)
        i += 2
      else:
        new_tokens.append(tokens[i])
        i += 1

    vocab[word_id] = new_tokens

def train(file_path, vocab_size=32_000, special_tokens=['<|endoftext|>']):
  corpus = Counter(get_corpus_it(file_path, special_tokens))
  
  vocab, freqs = {}, {}
  pair_counts = Counter()
  pair_words = defaultdict(set)

  for word_id, (tokens, freq) in enumerate(corpus.items()):
    vocab[word_id] = list(tokens)
    freqs[word_id] = freq

    for a, b in zip(tokens, tokens[1:]):
      pair_counts[(a, b)] += freq
      pair_words[(a, b)].add(word_id)

    pair_heap = PairHeap(pair_counts)

    merges = []
    num_merges = vocab_size - 256

    for _ in range(num_merges):
      best_pair, count = pair_heap.pop_best()
      if best_pair is None or count == 0:
        break
      apply_merge(best_pair, vocab, freqs, pair_words, pair_heap)
      merges.append(best_pair)

  return vocab, merges