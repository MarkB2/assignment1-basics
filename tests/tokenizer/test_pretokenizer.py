from collections import Counter
from pathlib import Path

import pytest
pytest.importorskip("test_pretokenizer.py", reason="WIP")

from cs336_basics.pretokenizer import Pretokenizer
from cs336_basics.types import SpecialToken


@pytest.fixture
def string():
    return "one two tr<|endoftext|>ee four"


def test_string_reader(string):
    words = Counter(Pretokenizer().iter_tokens(string))
    assert "one" in words
    assert " two" in words
    assert " four" in words
    assert " tr" in words
    assert "ee" in words
    assert "tree" not in words
    assert "<|endoftext|>" not in words


def test_file_reader(tmp_path):
    filename = tmp_path / "text.txt"
    filename.write_text("one two tr<|endoftext|>ee four")
    words = Counter(Pretokenizer().iter_file(filename))
    assert "one" in words
    assert " two" in words
    assert " four" in words
    assert " tr" in words
    assert "ee" in words
    assert "tree" not in words
    assert "<|endoftext|>" not in words


# def test_reader2():
#   words, _, _ = Reader(Path('tests/tokenizer/text.txt'), Pattern()).build()
#   assert len(words) == 6 # including \n


def test_pretokenizer():
    special_tokens = ["<|endoftext|>", "<|padding|>", "<|unk|>"]
    pt = Pretokenizer(special_tokens=special_tokens, keep_special_tokens=True)
    assert list(pt.iter_tokens("Hello<|endoftext|>world<|unk|>!")) == [
        "Hello",
        SpecialToken(text="<|endoftext|>"),
        "world",
        SpecialToken(text="<|unk|>"),
        "!",
    ]
    pt = Pretokenizer(special_tokens=special_tokens, keep_special_tokens=False)
    assert list(pt.iter_tokens("Hello<|endoftext|>world<|unk|>!")) == ["Hello", "world", "!"]


def test_pretokenizer_from_file(tmp_path):
    special_tokens = ["<|endoftext|>"]
    filename = tmp_path / "text.txt"
    filename.write_text("one two tr<|endoftext|>ee four")
    assert list(Pretokenizer(keep_special_tokens=True).iter_file(filename)) == [
        "one",
        " two",
        " tr",
        SpecialToken(text="<|endoftext|>"),
        "ee",
        " four",
    ]


def test_pretokenizer_from_file_multi():
    special_tokens = ["<|endoftext|>"]
    out = list(Pretokenizer().iter_file(Path("tests/fixtures/tinystories_sample_5M.txt"), max_chunk_size=10_000, num_workers=2))
    assert len(out) > 1_000_000
