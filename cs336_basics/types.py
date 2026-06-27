from collections import Counter, defaultdict

Pair = tuple[bytes, bytes]
PairCount = Counter[Pair]
PairLoc = defaultdict[Pair, set[int]]
