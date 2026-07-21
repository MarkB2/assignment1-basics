import pytest

from cs336_basics.new_vocab import EncodedVocab, Vocab, TieBreaker
from cs336_basics.types import IdPair, Pair, PackedPair, pack_pair, unpack_pair
from .helpers import to_encoded_pairs

def test_load_save():
    vocab = Vocab()
    vocab.save("test_vocab.json")
    loaded = Vocab.load("test_vocab.json")
    assert vocab._forward == loaded._forward  # pyright: ignore [reportPrivateUsage]
    assert vocab._merges == loaded._merges  # pyright: ignore [reportPrivateUsage]
    assert vocab._next_id == loaded._next_id  # pyright: ignore [reportPrivateUsage]


@pytest.fixture
def vocab() -> Vocab:
    v = Vocab(["<|endoftext|>"])
    for pair in [pack_pair(IdPair([v.id_for(b) for b in p])) for p in [(b"a", b"b"), (b"c", b"d"), (b"e", b"f")]]:
        _ = v.add_merge(pair)
    _ = v.add_special_token("<|endofchapter|>")
    return v


def test_id_for(vocab: Vocab):
    assert vocab.id_for(b"ab") == 257
    assert vocab.id_for(b"cd") == 258
    assert vocab.id_for(b"ef") == 259


def test_bytes_for(vocab: Vocab):
    assert vocab.bytes_for(257) == b"ab"
    assert vocab.bytes_for(258) == b"cd"
    assert vocab.bytes_for(259) == b"ef"


def test_merge(vocab: Vocab):
    assert vocab.merges() == to_encoded_pairs([(98, 99), (100, 101), (102, 103)])
    assert vocab.add_merge(pack_pair((257, 258))) == 261
    assert vocab.merges() == to_encoded_pairs([(98, 99), (100, 101), (102, 103), (257, 258)])


def test_len(vocab):
    assert len(vocab) == 261


def test_to_bytes():
    vocab = Vocab()
    assert (
        vocab.to_bytes(
            [
                116,
                104,
                97,
                116,
                32,
                119,
                97,
                115,
                32,
                97,
                32,
                119,
                111,
                110,
                100,
                101,
                114,
                102,
                117,
                108,
                32,
                100,
                97,
                121,
            ]
        )
        == b"that was a wonderful day"
    )


def test_to_ids():
    vocab = Vocab()
    assert vocab.to_ids("that was a wonderful day") == (
        116,
        104,
        97,
        116,
        32,
        119,
        97,
        115,
        32,
        97,
        32,
        119,
        111,
        110,
        100,
        101,
        114,
        102,
        117,
        108,
        32,
        100,
        97,
        121,
    )

def test_tie_breaker(vocab: Vocab):
  tie_breaker = TieBreaker(vocab)
  id1 = vocab.add(b'abc')
  id2 = vocab.add(b'abd')
  id3 = vocab.add(b'cba')
  assert [id1, id2, id3] == [261, 262, 263]
  pair1 = pack_pair((id1, id2)) # b'abc' b'abd'
  _ = vocab.add_merge(pair1)
  pair2 = pack_pair((id1, id1)) # b'abc' b'abc'
  _ = vocab.add_merge(pair2)
  pair3 = pack_pair((id3, id2)) # b'cba' b'abd'
  _ = vocab.add_merge(pair3)
  assert tie_breaker.lex_greater(pair1, pair2)
  assert tie_breaker.lex_greater(pair3, pair1)
  assert tie_breaker.lex_greater(pair3, pair2)
  assert not tie_breaker.lex_greater(pair1, pair3)
  assert not tie_breaker.lex_greater(pair2, pair3)
