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
