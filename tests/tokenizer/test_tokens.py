import pytest
from cs336_basics.tokens import to_bytes, Word

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
    (Word('abcbcbcd'), to_bytes('bc'), (b'a', b'bc', b'bc', b'bc', b'd'), None),
    (Word('abcd'), to_bytes('cv'), (b'a', b'b', b'c', b'd'), None),
    (Word('ab'), to_bytes('ab'), (b'ab',), None),
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