import pytest

from cs336_basics.new_vocab import EncodedVocab, Vocab, TieBreaker
cs336_basics.types import IdPair, Pair


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
    for pair in [IdPair([v.id_for(b) for b in p]) for p in [(b"a", b"b"), (b"c", b"d"), (b"e", b"f")]]:
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
    assert vocab.merges() == [(98, 99), (100, 101), (102, 103)]
    assert vocab.add_merge((257, 258)) == 261
    assert vocab.merges() == [(98, 99), (100, 101), (102, 103), (257, 258)]


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
