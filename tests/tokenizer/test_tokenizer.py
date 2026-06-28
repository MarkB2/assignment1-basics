import pytest
from pathlib import Path
from cs336_basics.types import Pair
from cs336_basics.tokenizer import Merges, Vocab
from ..common import FIXTURES_PATH

def test_serialization():
  out = FIXTURES_PATH / 'vocab.json'
  vocab = Vocab({3:b'abc'})
  vocab.save(out)
  assert Vocab.load(out) == vocab
  out = FIXTURES_PATH / 'merges.json'
  merges = Merges([Pair([b'a', b'\x00'])])
  merges.save(out)
  assert Merges.load(out) == merges
