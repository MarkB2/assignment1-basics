import pytest
import regex as re
from collections import Counter
from cs336_basics.splitter import to_bytes, PreTokenizer, BPETrainer

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

@pytest.fixture
def string_tokenizer():
  return BPETrainer(PreTokenizer.from_string(text, r'\S+'))

def test_to_bytes():
  assert to_bytes('one') == (b'o', b'n', b'e')

def test_from_string():
  pre = PreTokenizer.from_string(text, r'\S+')
  assert isinstance(pre, PreTokenizer)
  exp_toks = [to_bytes(word) for word in ['low', 'lower', 'widest', 'newest']]
  exp_counts = (5, 2, 3, 6)
  pr = {'lo': 7, 'ow': 7, 'we': 8, 'er': 2, 'wi': 3, 'id': 3, 'de': 3, 'es': 9, 'st': 9, 'ne': 6, 'ew': 6}
  exp_pair_freq = Counter({to_bytes(p): c for p, c in pr.items()})
  exp_pair_loc = {(b'l', b'o'): {0, 1}, (b'o', b'w'): {0, 1}, (b'w', b'e'): {1, 3}, (b'e', b'r'): {1}, (b'w', b'i'): {2}, (b'i', b'd'): {2}, (b'd', b'e'): {2}, (b'e', b's'): {2, 3}, (b's', b't'): {2, 3}, (b'n', b'e'): {3}, (b'e', b'w'): {3}}
  assert pre.tokens == exp_toks
  assert pre.counts == exp_counts
  assert pre.pair_freq == exp_pair_freq
  assert pre.pair_loc == exp_pair_loc

def test_tokenizer(string_tokenizer):
  assert isinstance(string_tokenizer, BPETrainer)

def test_merge(string_tokenizer):
  out = ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']
  est_merge = [tuple(a.encode('utf-8') for a in s.split()) for s in out]
  string_tokenizer.train()
  assert string_tokenizer.merges == est_merge

