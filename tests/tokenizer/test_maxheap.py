import pytest
from cs336_basics.maxheap import PairMaxHeap

@pytest.mark.parametrize("a, b, expected", [
    ((b'a', 7),   (b'ab', 5),  b'a'),
    ((b'aba', 5), (b'ab', 5),  b'aba'),
],
ids=["higher_num_wins", "tiebreak_bytes"])

def test_returns_max_number(a, b, expected):
    pair_freq = dict([a, b])
    heap = PairMaxHeap(pair_freq)
    assert heap.get_best() == expected

def test_update():
    pair_freq = {'a':10, 'b':6, 'c':6}
    heap = PairMaxHeap(pair_freq)
    deltas = {'a': -10, 'b': 1}
    heap.update(deltas)
    assert heap.get_best() == 'b'
    assert heap.pair_counts.get('a') is None
    