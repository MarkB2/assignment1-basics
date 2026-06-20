import pytest
from collections import Counter
from cs336_basics.reader import string_reader, file_reader, Pattern, Reader

@pytest.fixture
def string():
  return "one two tr<|endoftext|>ee four"

def test_string_reader(string):
  words = Counter(string_reader(string, Pattern()))
  assert "one" in words
  assert " two" in words
  assert " four" in words
  assert " tr" in words
  assert "ee" in words
  assert "tree" not in words
  assert "<|endoftext|>" not in words

def test_file_reader():
  words = Counter(file_reader('tests/tokenizer/text.txt', Pattern()))
  assert "one" in words
  assert " two" in words
  assert " four" in words
  assert " tr" in words
  assert "ee" in words
  assert "tree" not in words
  assert "<|endoftext|>" not in words

def test_reader_from_string(string):
  words = Reader.from_string(string, Pattern()).get_corpus()
  assert len(words) == 5

def test_reader_from_file():
  words = Reader.from_file('tests/tokenizer/text.txt', Pattern()).get_corpus()
  assert len(words) == 6 # including \n
