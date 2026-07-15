import heapq
from dataclasses import dataclass

from typing_extensions import final
from .types import IdPair, IdPairCount

@dataclass
class MaxNode:
    count: int
    pair: IdPair

    def __lt__(self, other: "MaxNode"):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair

@final
class PairMaxHeap:
    def __init__(self, pair_counts: IdPairCount) -> None:
        self.pair_counts = pair_counts
        self.heap = [MaxNode(count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(self.heap)

    def update(self, deltas: IdPairCount) -> None:
        for pair, delta in deltas.items():
            self.pair_counts[pair] += delta
            freq = self.pair_counts[pair]
            if freq > 0:
                heapq.heappush(self.heap, MaxNode(freq, pair))
            else:
                del self.pair_counts[pair]

    def get_best(self) -> IdPair | None:
        while self.heap:
            best = heapq.heappop(self.heap)
            if best.count == self.pair_counts.get(best.pair):
                return best.pair
        return None
