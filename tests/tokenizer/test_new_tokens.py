import pytest

from cs336_basics.new_tokens import NewWord
from cs336_basics.new_vocab import Vocab
from cs336_basics.types import IdPairCount


@pytest.fixture
def vocab():
    return Vocab()


def test_vocab(vocab: Vocab):
    assert len(vocab) == 256


@pytest.fixture
def word(vocab: Vocab):
    return NewWord(vocab.to_ids("this is test"))


def test_word(word: NewWord):
    assert word.tokens.tolist() == [116, 104, 105, 115, 32, 105, 115, 32, 116, 101, 115, 116]
    assert word.found_at(3, (115, 32))
    assert not word.found_at(2, (115, 32))


def test_merge_at_first(vocab: Vocab, word: NewWord):
    a, b = vocab.id_for(b"t"), vocab.id_for(b"h")
    assert (a, b) == (116, 104)
    assert len(vocab) == 256
    ab = vocab.add_merge((116, 104))
    update = word.merge((a, b), ab)
    assert ab == 256
    assert word.tokens.tolist() == [256, 105, 115, 32, 105, 115, 32, 116, 101, 115, 116]
    assert update == IdPairCount({(116, 104): -1, (104, 105): -1, (256, 105): 1})


def test_merge_at_last(vocab: Vocab, word: NewWord):
    a, b = vocab.id_for(b"s"), vocab.id_for(b"t")
    assert (a, b) == (115, 116)
    assert len(vocab) == 256
    ab = vocab.add_merge((115, 116))
    update = word.merge((a, b), ab)
    assert ab == 256
    assert word.tokens.tolist() == [116, 104, 105, 115, 32, 105, 115, 32, 116, 101, 256]
    assert update == IdPairCount({(115, 116): -1, (101, 115): -1, (101, 256): 1})


def test_merge_in_the_middle_twice(vocab: Vocab, word: NewWord):
    a, b = vocab.id_for(b"i"), vocab.id_for(b"s")
    assert (a, b) == (105, 115)
    assert len(vocab) == 256
    ab = vocab.add_merge((105, 115))
    update = word.merge((a, b), ab)
    assert ab == 256
    assert word.tokens.tolist() == [116, 104, ab, 32, ab, 32, 116, 101, 115, 116]
    assert update == IdPairCount(
        {(105, 115): -2, (104, 105): -1, (104, ab): 1, (115, 32): -2, (ab, 32): 2, (32, 105): -1, (32, ab): 1}
    )


def test_merge_in_the_middle_twice_overlapping(vocab: Vocab):
    word = NewWord(vocab.to_ids("abcdaaaefg"))
    a = vocab.id_for(b"a")
    assert (a, a) == (97, 97)
    assert len(vocab) == 256
    assert word.tokens.tolist() == [97, 98, 99, 100, 97, 97, 97, 101, 102, 103]
    ab = vocab.add_merge((97, 97))
    update = word.merge((a, a), ab)
    assert ab == 256
    assert word.tokens.tolist() == [97, 98, 99, 100, 256, 97, 101, 102, 103]
    assert update == IdPairCount({(97, 97): -2, (100, 97): -1, (100, ab): 1, (ab, 97): 1})
