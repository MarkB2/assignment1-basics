import pytest
from cs336_basics.max_heap import PairMaxHeap, Pair, PairCount

AA, BB, CC, AAA = tuple(map(Pair, [(b'a', b'a'), (b'b', b'b'), (b'c', b'c'), (b'aa', b'a')]))

@pytest.mark.parametrize("pair_count, expected", [
  (PairCount({AA: 7, BB: 5}),  AA),
  (PairCount({AAA: 5, AA: 5}),  AAA),
],
ids=["higher_num_wins", "tiebreak_bytes"])

def test_returns_max_number(pair_count, expected):
    heap = PairMaxHeap(pair_count)
    assert heap.get_best() == expected

def test_update():
  pair_count = PairCount({AA:10, BB:6, CC:6})
  heap = PairMaxHeap(pair_count)
  deltas = PairCount({AA: -10, BB: 1})
  heap.update(deltas)
  assert heap.get_best() == BB
  assert heap.pair_counts.get(AAA) is None
