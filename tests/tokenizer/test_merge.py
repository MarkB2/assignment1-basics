import pytest
from collections import Counter
from cs336_basics.merge import merge

@pytest.mark.parametrize("token, pair, exp_token, exp_updates", [
  (('a', 'b', 'c', 'd', 'e', 'f', 'g'), ('a', 'b'), # first char
   ('ab', 'c', 'd', 'e', 'f', 'g'),
   Counter({
    ('ab', 'c'): 1,
    ('a', 'b'): -1,
    ('b', 'c'): -1})
  ),
  (('a', 'b', 'c', 'd', 'e', 'f', 'g'), ('c', 'd'), # in the middle
   ('a', 'b', 'cd', 'e', 'f', 'g'),
   Counter({
    ('b', 'cd'): 1,
    ('cd', 'e'): 1,
    ('b', 'c'): -1,
    ('d', 'e'): -1,
    ('c', 'd'): -1})
  ),
  (('a', 'b', 'c', 'd', 'e', 'f', 'g'), ('f', 'g'), # last char
   ('a', 'b', 'c', 'd', 'e', 'fg'),
   Counter({
    ('e', 'fg'): 1,
    ('e', 'f'): -1,
    ('f', 'g'): -1})
  ),
  (('a', 'b', 'c', 'd', 'd', 'd', 'd', 'e', 'f', 'g'), ('d', 'd'), # two consecutive pairs
   ('a', 'b', 'c', 'dd', 'dd', 'e', 'f', 'g'),
   Counter({
    ('c', 'dd'): 1,
    ('dd', 'e'): 1,
    ('dd', 'dd'): 1,
    ('c', 'd'): -1,
    ('d', 'e'): -1,
    ('d', 'd'): -3})
  ),
  (('a', 'b', 'c', 'd', 'd', 'd', 'd','d', 'd', 'e', 'f', 'g'), ('d', 'd'), # three consecutive pairs
   ('a', 'b', 'c', 'dd', 'dd', 'dd', 'e', 'f', 'g'),
   Counter({
    ('c', 'dd'): 1,
    ('dd', 'e'): 1,
    ('dd', 'dd'): 2,
    ('c', 'd'): -1,
    ('d', 'e'): -1,
    ('d', 'd'): -5})
  ),
  (('a', 'b', 'c', 'd', 'd', 'd', 'e', 'f', 'g'), ('d', 'd'), # two overlaping consecutive pairs
   ('a', 'b', 'c', 'dd', 'd', 'e', 'f', 'g'),
   Counter({
    ('c', 'dd'): 1,
    ('dd', 'd'): 1,
    ('c', 'd'): -1,
    ('d', 'd'): -2})
  ),
  (('a', 'b'), ('a', 'b'), # token is a pair
   ('ab',),
   Counter({
    ('a', 'b'): -1})
  ),
  (('a', 'b', 'c'), ('a', 'b'), # token of length 1 ater merge
   ('ab', 'c'),
   Counter({
    ('ab', 'c'): 1, 
    ('b', 'c'): -1, 
    ('a', 'b'): -1})
  ),
  (('a', 'b', 'c', 'a', 'b'), ('a', 'b'), # same pair in start and end
   ('ab', 'c', 'ab'),
   Counter({
    ('ab', 'c'): 1, 
    ('c', 'ab'): 1, 
    ('b', 'c'): -1, 
    ('c', 'a'): -1, 
    ('a', 'b'): -2})
  ),
  (('a', 'b', 'a', 'b'), ('a', 'b'), # pair with itself as neibour
   ('ab', 'ab'),
   Counter({
    ('ab', 'ab'): 1, 
    ('a', 'b'): -2, 
    ('b', 'a'): -1})
  ),
],
ids=["first char", "in the middle", "last char", "two consecutive pairs", 
     "three consecutive pairs", "two overlaping consecutive pairs", 
     "token is a pair", "token of length 1 ater merge", 
     "same pair in start and end", "pair with itself as neibour"]
)

# def test_merge(token, pair, exp_token, exp_updates):
#   count = 1
#   updates = Counter()
#   assert merge(token, pair, count, updates) == exp_token
#   assert updates == exp_updates

def test_merge2(token, pair, exp_token, exp_updates):
  count = 1
  # updates = Counter()
  assert merge(token, pair, count) == (exp_token, exp_updates)
  # assert updates == exp_updates