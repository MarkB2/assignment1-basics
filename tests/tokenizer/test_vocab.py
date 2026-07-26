from pathlib import Path
from unittest.mock import patch

import pytest

from cs336_basics.types import IdPair, PackedPair, Pair, pack_pair, unpack_pair
from cs336_basics.vocab import TieBreaker, Vocab, load_vocab, save_vocab, vocab_to_bytes_vocab

from ..common import FIXTURES_PATH
from .helpers import to_encoded_pairs


def pair_helper(pairs: list[list[bytes]], vocab: Vocab) -> list[PackedPair]:
    for pair in pairs:
        for b in pair:
            if len(b) > 1:
                _ = vocab.add(b)
    return [pack_pair(*[vocab.id_for(b) for b in pair]) for pair in pairs]


@pytest.mark.parametrize(
    "pairs, expected",
    [
        ([[b"c", b"b"], [b"b", b"b"]], True),
        ([[b"b", b"b"], [b"b", b"c"]], False),
        ([[b"ca", b"b"], [b"c", b"b"]], True),
        ([[b"b", b"b"], [b"b", b"bc"]], False),
    ],
    ids=[
        "greater first elm",
        "greater second elm",
        "first elm longer",
        "second elm longer",
    ],
)
def test_tie_breaker(pairs: list[list[bytes]], expected: bool) -> None:
    vocab = Vocab()
    tie_breaker = TieBreaker(vocab)
    a, b = pair_helper(pairs, vocab)
    assert tie_breaker.lex_greater(a, b) == expected
    assert tie_breaker.greater(a, b) == expected


@pytest.mark.parametrize(
    "pairs, expected",
    [
        ([[b"c", b"b"], [b"b", b"b"]], True),
        ([[b"b", b"b"], [b"c", b"b"]], False),
    ],
    ids=[
        "greater first elm",
        "greater second elm",
    ],
)
def test_tie_breaker_cache(pairs: list[list[bytes]], expected: bool) -> None:
    vocab = Vocab()
    tie_breaker = TieBreaker(vocab)
    a, b = pair_helper(pairs, vocab)
    with patch.object(tie_breaker, "lex_greater", wraps=tie_breaker.lex_greater) as spy:
        result1 = tie_breaker.greater(a, b)
        assert spy.call_count == 1  # cache miss
        result2 = tie_breaker.greater(a, b)
        assert spy.call_count == 1  # cache hit
        result3 = tie_breaker.greater(b, a)
        assert spy.call_count == 1  # cache hit
    assert result1 == expected == result2 == (not result3)


@pytest.mark.parametrize(
    "pairs",
    [
        ([[b"c", b"b"], [b"b", b"b"]]),
    ],
)
def test_load_save(pairs: list[list[bytes]], tmp_path: Path):
    vocab = Vocab()
    packed = pair_helper(pairs, vocab)
    for pair in packed:
        _ = vocab.add_merge(pair)
    filename = tmp_path / "test_vocab.json"
    save_vocab(filename, vocab)
    loaded = load_vocab(filename)
    assert vocab._forward == loaded._forward  # pyright: ignore [reportPrivateUsage]
    assert vocab._merges == loaded._merges == packed  # pyright: ignore [reportPrivateUsage]
    assert vocab._next_id == loaded._next_id  # pyright: ignore [reportPrivateUsage]


def test_vocab_to_bytes_vocab(tmp_path):
    vocab = Vocab()
    encoded = vocab_to_bytes_vocab(vocab)
    assert encoded


# @pytest.fixture
# def vocab() -> Vocab:
#     v = Vocab(["<|endoftext|>"])
#     for pair in [pack_pair(IdPair([v.id_for(b) for b in p])) for p in [(b"a", b"b"), (b"c", b"d"), (b"e", b"f")]]:
#         _ = v.add_merge(pair)
#     _ = v.add_special_token("<|endofchapter|>")
#     return v


# def test_id_for(vocab: Vocab):
#     assert vocab.id_for(b"ab") == 257
#     assert vocab.id_for(b"cd") == 258
#     assert vocab.id_for(b"ef") == 259


# def test_bytes_for(vocab: Vocab):
#     assert vocab.bytes_for(257) == b"ab"
#     assert vocab.bytes_for(258) == b"cd"
#     assert vocab.bytes_for(259) == b"ef"


# def test_merge(vocab: Vocab):
#     assert vocab.merges() == to_encoded_pairs([(98, 99), (100, 101), (102, 103)])
#     assert vocab.add_merge(pack_pair((257, 258))) == 261
#     assert vocab.merges() == to_encoded_pairs([(98, 99), (100, 101), (102, 103), (257, 258)])


# def test_len(vocab):
#     assert len(vocab) == 261


# def test_to_bytes():
#     vocab = Vocab()
#     assert (
#         vocab.to_bytes(
#             [
#                 116,
#                 104,
#                 97,
#                 116,
#                 32,
#                 119,
#                 97,
#                 115,
#                 32,
#                 97,
#                 32,
#                 119,
#                 111,
#                 110,
#                 100,
#                 101,
#                 114,
#                 102,
#                 117,
#                 108,
#                 32,
#                 100,
#                 97,
#                 121,
#             ]
#         )
#         == b"that was a wonderful day"
#     )


# def test_to_ids():
#     vocab = Vocab()
#     assert vocab.to_ids("that was a wonderful day") == (
#         116,
#         104,
#         97,
#         116,
#         32,
#         119,
#         97,
#         115,
#         32,
#         97,
#         32,
#         119,
#         111,
#         110,
#         100,
#         101,
#         114,
#         102,
#         117,
#         108,
#         32,
#         100,
#         97,
#         121,
#     )

# def test_tie_breaker(vocab: Vocab):
#   tie_breaker = TieBreaker(vocab)
#   id1 = vocab.add(b'abc')
#   id2 = vocab.add(b'abd')
#   id3 = vocab.add(b'cba')
#   assert [id1, id2, id3] == [261, 262, 263]
#   pair1 = pack_pair((id1, id2)) # b'abc' b'abd'
#   _ = vocab.add_merge(pair1)
#   pair2 = pack_pair((id1, id1)) # b'abc' b'abc'
#   _ = vocab.add_merge(pair2)
#   pair3 = pack_pair((id3, id2)) # b'cba' b'abd'
#   _ = vocab.add_merge(pair3)
#   assert tie_breaker.lex_greater(pair1, pair2)
#   assert tie_breaker.lex_greater(pair3, pair1)
#   assert tie_breaker.lex_greater(pair3, pair2)
#   assert not tie_breaker.lex_greater(pair1, pair3)
#   assert not tie_breaker.lex_greater(pair2, pair3)
