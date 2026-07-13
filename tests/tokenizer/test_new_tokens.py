import pytest
from cs336_basics.types import Vocab, MergeResult, IdPairCount
from cs336_basics.new_tokens import NewWord

@pytest.fixture(scope="module")
def read_vocab():
  # global VOCAB
  # VOCAB =
  return Vocab.load('tests/tokenizer/vocab.txt')

@pytest.fixture
def vocab(read_vocab):
  return read_vocab

def test_vocab(vocab):
  assert len(vocab) == 500

@pytest.fixture
def word():
  # offset = 1 accounts for one special token at id = 0
  return NewWord('this is test', offset=1)

def test_word(vocab, word):
  assert word.tokens.tolist() == [117, 105, 106, 116, 33, 106, 116, 33, 117, 102, 116, 117]
  assert word.found_at(3, (116, 33))
  assert not word.found_at(2, (116, 33))

def test_merge_at_first(vocab, word):
  a, b, ab = vocab.id_for(b't'), vocab.id_for(b'h'), vocab.id_for(b'th')
  assert (a, b, ab) == (117, 105, 349)
  result = word.merge((a, b, ab))
  assert word.tokens.tolist() == [ab, 106, 116, 33, 106, 116, 33, 117, 102, 116, 117]
  assert result == IdPairCount({(117,105):-1, (105,106):-1, (349,106):1})

def test_merge_at_last(vocab, word):
  a, b, ab = vocab.id_for(b's'), vocab.id_for(b't'), vocab.id_for(b'st')
  assert (a, b, ab) == (116, 117, 319)
  result = word.merge((a, b, ab))
  assert word.tokens.tolist() == [117, 105, 106, 116, 33, 106, 116, 33, 117, 102, ab]
  assert result == IdPairCount({(116,117):-1, (102,116):-1, (102,319):1})

def test_merge_in_the_middle_twice(vocab, word):
  a, b, ab = vocab.id_for(b'i'), vocab.id_for(b's'), vocab.id_for(b'is')
  assert (a, b, ab) == (106, 116, 270)
  result = word.merge((a, b, ab))
  assert word.tokens.tolist() == [117, 105, ab, 33, ab, 33, 117, 102, 116, 117]
  assert result == IdPairCount({(106,116):-2, (105,106):-1, (105,270):1, (270,33):2, (116,33):-2, (33,106):-1, (33,270):1})
