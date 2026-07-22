from collections.abc import Callable
import heapq
from dataclasses import dataclass, field
from typing import final

from .types import PackedPair, PackedPairCount, TieBreak

@dataclass(slots=True)
class MaxNode:
    count: int
    pair: PackedPair
    lex_greater: TieBreak = field(compare=False)

    def __lt__(self, other: "MaxNode"):
        if self.count != other.count:
            return self.count > other.count
        return self.lex_greater(self.pair, other.pair)


@final
class MaxHeap:
    def __init__(self, pair_counts: PackedPairCount, tie_break: TieBreak) -> None:
        self.pair_counts = pair_counts
        self._tie_break = tie_break
        self.heap = [self.make_node(count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(self.heap)

    def make_node(self, count: int, pair: PackedPair):
        return MaxNode(count, pair, self._tie_break)

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
