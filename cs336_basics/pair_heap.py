from dataclasses import dataclass
import heapq
from collections import defaultdict

class RevBytes:
    __slots__ = ('b',)

    def __init__(self, b): self.b = b
    def __eq__(self, other): return self.b == other.b
    def __lt__(self, other): return self.b > other.b
    def __le__(self, other): return self.b >= other.b
    def __gt__(self, other): return self.b < other.b
    def __ge__(self, other): return self.b <= other.b
    
@dataclass(order=True)
class HeapEntry:
    neg_count: int
    neg_pair: tuple

    @staticmethod
    def make(count, pair):
        return HeapEntry(-count, tuple(RevBytes(b) for b in pair))

    @property
    def pair(self):
        return tuple(rb.b for rb in self.neg_pair)

    @property
    def count(self):
        return -self.neg_count


class PairHeap:

    def __init__(self, pair_counts: dict):
#        self._heap = [HeapEntry.make(cnt, pair) for pair, cnt in pair_counts.items()]
        self._heap = [(-cnt, pair) for pair, cnt in pair_counts.items()]
        heapq.heapify(self._heap)
        self._counts = pair_counts   # live reference, not a copy

    def push(self, pair, count):
#        heapq.heappush(self._heap, HeapEntry.make(count, pair))
        heapq.heappush(self._heap, (- self._counts[pair], pair))

    def pop_best(self):
        while self._heap:
            count, pair = heapq.heappop(self._heap)
            if self._counts.get(pair, 0) == -count:
                return pair, -count
        return None, 0

    def update(self, pair, delta):
        """Increment or decrement a pair's count and push updated entry."""
        self._counts[pair] += delta
        if self._counts[pair] <= 0:
            del self._counts[pair]
        else:
            self.push(pair, self._counts[pair])

    def __len__(self):
        return len(self._heap)
    
