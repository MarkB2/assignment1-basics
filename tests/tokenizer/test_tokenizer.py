from _pytest.fixtures import fixture
import pytest
pytest.importorskip("test_tokenizer.py", reason="WIP")

from pathlib import Path
from cs336_basics.types import Pair
from cs336_basics.tokenizer import Merges, Vocab, Tokenizer
from ..common import FIXTURES_PATH

@pytest.fixture
def tokenizer() -> Tokenizer:
  return Tokenizer.from_files(FIXTURES_PATH / 'vocab_corpus.json', FIXTURES_PATH / 'merges_corpus.json')

def test_serialization():
  out = FIXTURES_PATH / 'vocab.json'
  vocab = Vocab({3:b'abc'})
  vocab.save(out)
  assert Vocab.load(out) == vocab
  out = FIXTURES_PATH / 'merges.json'
  merges = Merges([Pair([b'a', b'\x00'])])
  merges.save(out)
  assert Merges.load(out) == merges

def test_tokenizer(tokenizer: Tokenizer):
  assert tokenizer is not None
  assert tokenizer.vocab is not None
  assert tokenizer.merges is not None

# def test_json_fixtures():
#   from cs336_basics.trainer import BPETrainer
#   vocab, merges = BPETrainer(FIXTURES_PATH / 'corpus.en', 500, ['<|endoftext|>']).merge()
#   vocab.save(FIXTURES_PATH / 'vocab_corpus.json')
#   merges.save(FIXTURES_PATH / 'merges_corpus.json')

def test_encoder(tokenizer: Tokenizer):
  string = "In the previous part of the assignment"
  encoded = tokenizer.encode(string)
  assert encoded == [74, 111, 294, 102, 345, 102, 119, 106, 277, 116, 282, 385, 281, 294, 102, 363, 116, 337, 111, 110, 306]
  assert tokenizer.decode(encoded) == string
