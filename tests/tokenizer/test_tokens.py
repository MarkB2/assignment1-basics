import sys
import numpy as np
import pytest
from collections import Counter

from cs336_basics.tokens import NewWord
from cs336_basics.vocab import Vocab
from cs336_basics.types import IdPairCount, PackedPair, pack_pair, unpack_pair
from .helpers import to_encoded_pairs, from_encoded_pairs

def test_pack_unpack_pair():
    pair = (31987, 31897)
    assert pack_pair(*pair) == 31987 * 65536 + 31897
    assert unpack_pair(pack_pair(*pair)) == pair
    assert pack_pair(*pair) > pack_pair(pair[1], pair[0])


@pytest.fixture
def vocab():
    return Vocab()


def test_vocab(vocab: Vocab):
    assert len(vocab) == 256


@pytest.fixture
def word(vocab: Vocab):
    return NewWord(vocab.to_ids("this is test"))


def test_word(word: NewWord):
    expected = [116, 104, 105, 115, 32, 105, 115, 32, 116, 101, 115, 116]
    assert word.tokens.tolist() == expected
    assert word.found_at(3, 115, 32)
    assert not word.found_at(2, 115, 32)

def test_pairs(word: NewWord):
    toks = [116, 104, 105, 115, 32, 105, 115, 32, 116, 101, 115, 116]
    expected = to_encoded_pairs([(a,b) for a, b in zip(toks[:-1], toks[1:])])
    assert word.pairs().tolist() == expected

def test_found(vocab: Vocab):
    word = NewWord(vocab.to_ids("this"))
    pair = vocab.to_ids('th')
    assert word.found_at(0, *pair)
    assert not word.found_at(1, *pair)
    assert not word.found_at(32, *pair)
    pair = vocab.to_ids('hi')
    assert word.found_at(1, *pair)
    pair = vocab.to_ids('is')
    assert word.found_at(2, *pair)


def test_merge_at_first(vocab: Vocab, word: NewWord):
    word = NewWord(vocab.to_ids("this"))
    assert word.tokens.tolist() == [116, 104, 105, 115]
    pair = vocab.to_ids('th')
    assert word.merge(pack_pair(*pair), 1000) == {pack_pair(1000, 105): 1, pack_pair(116, 104): -1, pack_pair(104, 105): -1}
    assert word.tokens.tolist() == [1000, 105, 115]

def test_merge_at_last(vocab: Vocab, word: NewWord):
  word = NewWord(vocab.to_ids("this"))
  assert word.tokens.tolist() == [116, 104, 105, 115]
  pair = vocab.to_ids('is')
  assert word.merge(pack_pair(*pair), 1000) == {pack_pair(104, 1000): 1, pack_pair(104, 105): -1, pack_pair(105, 115): -1}
  assert word.tokens.tolist() == [116, 104, 1000]


def test_merge_in_the_middle(vocab: Vocab, word: NewWord):
    word = NewWord(vocab.to_ids("this"))
    assert word.tokens.tolist() == [116, 104, 105, 115]
    pair = vocab.to_ids('hi')
    assert word.merge(pack_pair(*pair), 1000) == {pack_pair(116, 1000): 1, pack_pair(1000, 115): 1, pack_pair(104, 105): -1, pack_pair(116, 104): -1, pack_pair(105, 115): -1}
    assert word.tokens.tolist() == [116, 1000, 115]

def test_merge_in_the_middle_twice(vocab: Vocab):
    word = NewWord(vocab.to_ids("thaaaais"))
    assert word.tokens.tolist() == [116, 104, 97, 97, 97, 97, 105, 115]
    pair = vocab.to_ids('aa')
    assert word.merge(pack_pair(*pair), 1000) == {
      pack_pair(104, 1000): 1, pack_pair(1000, 1000): 1, pack_pair(1000, 105): 1,
      pack_pair(104, 97): -1, pack_pair(97, 97): -3, pack_pair(97, 105): -1}
    assert word.tokens.tolist() == [116, 104, 1000, 1000, 105, 115]

def test_merge_in_the_middle_twice_overlapping(vocab: Vocab):
    word = NewWord(vocab.to_ids("thaaais"))
    assert word.tokens.tolist() == [116, 104, 97, 97, 97, 105, 115]
    pair = vocab.to_ids('aa')
    assert word.found_at(2, *pair)
    assert word.found_at(3, *pair)
    assert word.merge(pack_pair(*pair), 1000) == Counter({pack_pair(104, 1000): 1, pack_pair(1000, 97): 1, pack_pair(104, 97): -1, pack_pair(97, 97): -2})
    assert word.tokens.tolist() == [116, 104, 1000, 97, 105, 115]
