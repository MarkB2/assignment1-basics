import pytest
from collections import Counter
from pathlib import Path
from cs336_basics.reader import Reader, reader, Pattern

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

def test_reader2():
  words, _, _ = Reader(Path('tests/tokenizer/text.txt'), Pattern()).build()
  assert len(words) == 6 # including \n
