import pytest
from cs336_basics.types import Vocab

def test_from_dict():
  vocab = Vocab.from_dict({0: b'<|endoftext|>', 9: b'ag', 3: b'd'})
  assert vocab.id_for(b'<|endoftext|>') == 0
  assert vocab.add(b'x') == 10
  assert vocab.add_special_token('<|endoftext|>') == 0
  assert vocab.add_special_token('<|beginoftext|>') == 11
