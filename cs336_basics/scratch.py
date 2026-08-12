from collections.abc import Callable

from cs336_basics.pretokenizer import reader, Pattern
from pathlib import Path

for match in reader("word \n\n\n\n word", Pattern()):
  print(bytes(match, encoding='utf-8'))
with open(Path('./text'), 'w') as f:
  f.write("word \n\n\n\n word")
for match in reader(Path('./text'), Pattern()):
  print(bytes(match, encoding='utf-8'))
# %%
from cs336_basics.pretokenizer import Pattern, reader
pat = Pattern()
print(pat, pat.pat, pat.p)
source = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
for  match in pat.compiled.finditer(source):
  print(match.group(1), match.group(2))
  # if match.group(2):
  #   yield match.group(2)


# %%
import regex as re
pat = "(<\\|endoftext\\|>)|('(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+)"
source = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
for match in re.finditer(pat, source):
  if match.group(2):
    print(match.group(2))

# %%
Pair = tuple
PairLoc = defaultdict[Pair, set[int]]

p = PairLoc()

# p = defaultdict(set)
print(type(p))
print(p.default_factory)
# print(p[(b'i', b'r')])


# %%
import numpy as np

tokens = b'this is a word'
array = np.array([b for b in tokens], dtype='uint16')
print(array)
d = dict()
# d[array]=5
print(d)
b't'[0], b'h'[0]


# %%
import numpy as np

a = int(np.uint16(105))
print((10, a))


# %%
def to_bytes(x): return bytes(chr(x), encoding='utf-8')
print([i for i in 'that was a wonderful day'.encode('utf-8')], end = ' ')
# bb = b"".join([to_bytes(b) for b in [117, 105, 106, 116, 33, 106, 116, 33, 117, 102, 116, 117]])
# [ord(bytes([b])) for b in bb]

# %%
import numpy as np
from timeit import timeit
from collections.abc import Callable

a = np.array([1,2,3,2,1,1,5], dtype='uint16')
b = np.array([1,2,6,1,1,5], dtype='uint16')

def pair_diff(old_pairs: np.ndarray, new_pairs: np.ndarray) -> np.ndarray:
    old_packed = old_pairs.astype(np.uint32)[:-1] << 16 | old_pairs[1:]
    new_packed = new_pairs.astype(np.uint32)[:-1] << 16 | new_pairs[1:]

    all_packed = np.concatenate([old_packed, new_packed])
    weights = np.concatenate([
        -np.ones(len(old_packed), dtype=np.int64),   # old pairs: -1 each (being removed)
        np.ones(len(new_packed), dtype=np.int64),    # new pairs: +1 each (being added)
    ])

    unique_pairs, inverse = np.unique(all_packed, return_inverse=True)
    diff_counts = np.bincount(inverse, weights=weights).astype(np.int64)

    # drop entries where diff is zero — pair appeared equally in both, net no change
    nonzero = diff_counts != 0
    return np.column_stack([unique_pairs[nonzero], diff_counts[nonzero]])

def check_it(c: Callable, number=1000):
  print(f"{timeit(c, number=number)/number*1e6:.2f} um/call")

# timeit(lambda: pair_diff(a,b), number=1000)/1000
check_it(lambda: pair_diff(a,b))

def aa():
  d = {}
  d['a']=1
  d['b']=2
  d['c']=3
  return d

check_it(lambda: aa())

# %%
a = 6357097
print(a >> 16, a & 0xFFFF)

# %%
from cs336_basics.model_types import PackedPair

a: PackedPair = 12
isinstance(a, PackedPair)

# %%
from collections.abc import Callable
from timeit import timeit
import numpy as np

def check_it(c: Callable, number=1000):
  print(f"{timeit(c, number=number)/number*1e6:.2f} um/call")


toks = np.arange(10)  # grab a realistic one from a debugger/print
def numpy_version(toks):
    return toks.astype(dtype='uint32')[:-1] << 16 | toks[1:]

def python_loop_version(toks):
    return [(int(a)<<16)|int(b) for a,b in zip(toks, toks[1:])]
for n in [3, 5, 10, 15, 20, 40]:
    toks = np.arange(n)
    print(n)
    check_it(lambda: numpy_version(toks), number=100_000)
    check_it(lambda: python_loop_version(toks), number=100_000)

# %%
import numpy as np
word = np.frombuffer(b'revolution', dtype="uint8")
print(word)

def merge(tokens: list[int], pos: int, id: int) -> list[int]:
    return tokens[:pos] + [id] + tokens[pos+2:]

merge(word.tolist(), 4, 120)

# %%
import torch

a = torch.randn(2,3)
x,y = a.shape
x+y
