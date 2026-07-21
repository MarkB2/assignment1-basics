import heapq
from dataclasses import dataclass
from typing import final

from .new_vocab import TieBreaker
from .types import PackedPair, PackedPairCount


@dataclass(slots=True)
class MaxNode:
    count: int
    pair: PackedPair
    tie_breaker: TieBreaker = field(compare=False)

    def __lt__(self, other: "MaxNode"):
        if self.count != other.count:
            return self.count > other.count
        return self.tie_breaker.lex_greater(self.pair, other.pair)


@final
class PairMaxHeap:
    def __init__(self, pair_counts: PackedPairCount, tie_breaker: TieBreaker) -> None:
        self.pair_counts = pair_counts
        self._tie_breaker = tie_breaker
        self.heap = [self.make_node(count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(self.heap)

    def make_node(self, count: int, pair: PackedPair):
        return MaxNode(count, pair, self._tie_breaker)

    def update(self, deltas: PackedPairCount) -> None:
        for pair, delta in deltas.items():
            self.pair_counts[pair] += delta
            freq = self.pair_counts[pair]
            if freq > 0:
                heapq.heappush(self.heap, self.make_node(freq, pair))
            else:
                del self.pair_counts[pair]

    def get_best(self) -> PackedPair | None:
        while self.heap:
            best = heapq.heappop(self.heap)
            if best.count == self.pair_counts.get(best.pair):
                return best.pair
        return None
