import pytest
from cs336_basics.tokens import to_bytes, Word
from cs336_basics.tokenizer import Vocab, Merges
from ..common import FIXTURES_PATH

def test_to_bytes():
  assert to_bytes('abc') == (b'a', b'b', b'c')

@pytest.fixture
def word():
  return Word('abcd')

def test_found(word):
  assert word.found_at(0, (b'a', b'b'))
  assert word.found_at(1, (b'b', b'c'))
  assert word.found_at(2, (b'c', b'd'))
  # don't throw exception
  assert not word.found_at(3, (b'd', b'e'))

def test_pairs(word):
  pairs = word.pairs()
  assert pairs == [(b'a', b'b'), (b'b', b'c'), (b'c', b'd')]

@pytest.mark.parametrize(
  "word, pair, expected, exp_updates",
  [
    (Word('abcd'), to_bytes('ab'), (b'ab', b'c', b'd'),
      {(b'a', b'b'): -1, (b'ab', b'c'): 1, (b'b', b'c'): -1}),
    (Word('abcd'), to_bytes('bc'), (b'a', b'bc', b'd'),
      {(b'a', b'b'): -1, (b'a', b'bc'): 1, (b'bc', b'd'): 1, (b'b', b'c'): -1, (b'c', b'd'): -1}),
    (Word('abcd'), to_bytes('cd'), (b'a', b'b', b'cd'),
      {(b'b', b'c'): -1, (b'b', b'cd'): 1, (b'c', b'd'): -1}),
    (Word('abcccd'), to_bytes('cc'), (b'a', b'b', b'cc', b'c', b'd'),
      {(b'b', b'c'): -1, (b'b', b'cc'): 1, (b'cc', b'c'): 1, (b'c', b'c'): -2}),
    (Word('abcbcd'), to_bytes('bc'), (b'a', b'bc', b'bc', b'd'),
      {(b'a', b'b'): -1, (b'a', b'bc'): 1, (b'bc', b'bc'): 1, (b'b', b'c'): -2, (b'bc', b'd'): 1, (b'c', b'b'): -1, (b'c', b'd'): -1}),
    (Word('abcbcbcd'), to_bytes('bc'), (b'a', b'bc', b'bc', b'bc', b'd'),
      {(b'a', b'b'): -1, (b'a', b'bc'): 1, (b'bc', b'bc'): 2, (b'b', b'c'): -3, (b'bc', b'd'): 1, (b'c', b'b'): -2, (b'c', b'd'): -1}),
    (Word('abcd'), to_bytes('cv'), (b'a', b'b', b'c', b'd'), {}),
    (Word('ab'), to_bytes('ab'), (b'ab',), {(b'a', b'b'): -1}),
  ],
  ids=[
    "in the beginning",
    "in the middle",
    "at the end",
    "overlapping pairs",
    "two consequtive pairs",
    "three consequtive pairs",
    "not found pair",
    "short word",
  ]
)

def test_merge(word, pair, expected, exp_updates):
  updates = word.merge(pair)
  assert word.tokens == expected
  if exp_updates:
    assert updates == exp_updates


def test_encode():
  vocab = Vocab.load(FIXTURES_PATH / 'vocab_corpus.json')
  merges = Merges.load(FIXTURES_PATH / 'merges_corpus.json')
  lookup_table = set([pair for pair in merges])
  ids = {v:k for k,v in vocab.items()}
  word = Word('university')
  assert len(word.tokens) == 10
  id, pos = lookup(word.tokens, word.tokens[0], 1, lookup_table, ids)
  assert Pair([b'un', b'i']) not in lookup_table
  assert id == 426
  assert pos == 2
  id, pos = lookup(word.tokens, word.tokens[pos], pos+1, lookup_table, ids)
  assert id == 106
  assert pos == 3
  id, pos = lookup(word.tokens, word.tokens[pos], pos+1, lookup_table, ids)
  assert id == 327
  assert pos == 5
  id, pos = lookup(word.tokens, word.tokens[pos], pos+1, lookup_table, ids)
  assert id == 115
  assert pos == 6
  id, pos = lookup(word.tokens, word.tokens[pos], pos+1, lookup_table, ids)
  assert id == 116
  assert pos == 7
  id, pos = lookup(word.tokens, word.tokens[pos], pos+1, lookup_table, ids)
  assert id == 440
  assert pos == 10

  res = encode(word.tokens, lookup_table, ids)
  assert res == [426, 106, 327, 115, 116, 440]
