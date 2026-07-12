from collections import Counter
from cs336_basics.pretokenizer import reader, Pattern
from pathlib import Path

# corpus = Counter(reader(Path("/home/mark/projects/stanford-cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"), Pattern()))

# total_pairs = 0
# for word, freq in corpus.items():
#     if word and set(word) == {'\n'}:
#         n = len(word)
#         contributed = (n - 1) * freq
#         total_pairs += contributed
#         print(f"run of {n} newlines, freq={freq}, contributes {contributed} pairs")

# print("TOTAL \\n\\n pairs:", total_pairs)

# for match in Pattern().compiled.finditer(open("/home/mark/projects/stanford-cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", encoding='utf-8').read()):
#     if match.group(1):
#         print(repr(match.group(0)), match.start())


with open("/home/mark/projects/stanford-cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", 'rb') as f:
    data = f.read(10000)
print(b'\r\n' in data, data.count(b'\r\n'), data.count(b'\n'))
