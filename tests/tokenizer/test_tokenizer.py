from numpy.ma.extras import isin
import pytest
from pathlib import Path
from cs336_basics.tokenizer import deserialize


def test_deserialize():
  obj = deserialize(Path("/home/mark/projects/stanford-cs336/assignment1-basics/TinyStories.json"))
  assert isinstance(obj, tuple)
  vocab, merges = obj
  assert isinstance(vocab, dict)
  assert isinstance(merges, tuple)
  assert isinstance(vocab[0], bytes)
  assert isinstance(merges[0], tuple)
  a, b = merges[0]
  assert isinstance(a, bytes)
  assert isinstance(b, bytes)
