import pytest
from cs336_basics.types import Pair, Vocab

@pytest.fixture
def vocab() -> Vocab:
  return Vocab.from_dict({0: b'<|endoftext|>', 9: b'ag', 3: b'd'})

def test_from_dict(vocab):
  assert vocab.id_for(b'<|endoftext|>') == 0
  assert vocab.add(b'x') == 10
  assert vocab.add_special_token('<|endoftext|>') == 0
  assert vocab.add_special_token('<|beginoftext|>') == 11

def test_id_for(vocab):
  assert vocab.id_for(b'ag') == 9
  assert vocab.id_for(b'd') == 3

def test_bytes_for(vocab):
  assert vocab.bytes_for(9) == b'ag'
  assert vocab.bytes_for(3) == b'd'

def test_save_load(vocab):
  vocab.save('test.file')
  loaded = Vocab.load('test.file')
  assert loaded._forward == vocab._forward
  assert loaded._reverse == vocab._reverse
  assert loaded._next_id == vocab._next_id

def test_len(vocab):
  assert len(vocab) == 3

def test_add_merge(vocab):
  result = vocab.add_merge(Pair((b'ag', b'd')))
  assert result.a == 9
  assert result.b == 3
  assert result.ab == 10
  assert len(vocab) == 4
