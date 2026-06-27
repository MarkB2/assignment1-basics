import regex as re
from cs336_basics.trainer import BPETrainer, train

def setup():
  out = ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']
  return [tuple(bytes(s, encoding='utf-8') for s in el.split()) for el in out]

def test_train():
  source = """low low low low low
  lower lower widest widest widest
  newest newest newest newest newest newest"""
  pat = r"\w+"
  vocab, merges = BPETrainer(source, 500, ["<|endoftext|>"], pat).merge()
  assert list(vocab.values())[257:263] == [b'st', b'est', b'ow', b'low', b'west', b'ne']
  assert merges == setup()
