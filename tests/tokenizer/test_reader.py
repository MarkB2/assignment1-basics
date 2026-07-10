import pytest
from collections import Counter
from pathlib import Path
from cs336_basics.types import SpecialToken
from cs336_basics.reader import Reader, reader, Pattern, Pretokenizer, StringPretokenizer, FilePretokenizer

@pytest.fixture
def string():
  return "one two tr<|endoftext|>ee four"

def test_string_reader(string):
  words = Counter(reader(string, Pattern()))
  assert "one" in words
  assert " two" in words
  assert " four" in words
  assert " tr" in words
  assert "ee" in words
  assert "tree" not in words
  assert "<|endoftext|>" not in words

def test_file_reader():
  words = Counter(reader(Path('tests/tokenizer/text.txt'), Pattern()))
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
  pt = StringPretokenizer("Hello<|endoftext|>world<|unk|>!", special_tokens=special_tokens, keep_special_tokens=True)
  assert list(pt.read()) == ["Hello", SpecialToken(text='<|endoftext|>'), "world", SpecialToken(text='<|unk|>'), "!"]
  pt = StringPretokenizer("Hello<|endoftext|>world<|unk|>!", special_tokens=special_tokens, keep_special_tokens=False)
  assert list(pt.read()) == ["Hello", "world", "!"]

def test_pretokenizer_from_file():
  special_tokens = ["<|endoftext|>"]
  pt = FilePretokenizer(Path('tests/tokenizer/text.txt'))
  assert list(pt.read()) == ['one', ' two', ' tr', 'ee', ' four', '\n']
  pt = FilePretokenizer(Path('tests/tokenizer/text.txt'), keep_special_tokens=True)
  assert list(pt.read()) == ['one', ' two', ' tr', SpecialToken(text='<|endoftext|>'), 'ee', ' four', '\n']
