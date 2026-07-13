import pytest
from cs336_basics.types import IdPair, Pair
from cs336_basics.new_vocab import Vocab

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
  merged_id = vocab.add_merge(IdPair((9, 3)))
  assert merged_id == 10
  assert vocab.bytes_for(merged_id) == b'agd'
  assert len(vocab) == 4

def test_to_bytes():
  vocab = Vocab()
  assert vocab.to_bytes([
    116, 104, 97, 116, 32, 119, 97, 115, 32, 97, 32, 119, 111,
    110, 100, 101, 114, 102, 117, 108, 32, 100, 97, 121
  ]) == b'that was a wonderful day'

def test_to_ids():
  vocab = Vocab()
  assert vocab.to_ids('that was a wonderful day') == (
    116, 104, 97, 116, 32, 119, 97, 115, 32, 97, 32, 119, 111,
    110, 100, 101, 114, 102, 117, 108, 32, 100, 97, 121
  )
