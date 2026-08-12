import pytest

from cs336_basics.max_heap import MaxHeap, MaxNode
from cs336_basics.model_types import PackedPair, PackedPairCount, TieBreak


def gt_break(a: PackedPair, b: PackedPair) -> bool:
    return a > b


def lt_break(a: PackedPair, b: PackedPair) -> bool:
    return a < b


def node_helper(tie_break: TieBreak):
    def make_node(count: int, pair: PackedPair) -> MaxNode:
        return MaxNode(count, pair, tie_break)

    return make_node


# Python Heap uses __ls__ in its heap. To build max heap
# greater node should be reported as least
@pytest.mark.parametrize(
    "tie_break, count_a, pair_a, count_b, pair_b, a_lt_b",
    [
        # gt_break
        (gt_break, 5, 12, 4, 13, True),  # count a > count b
        (gt_break, 3, 13, 4, 12, False),  # count a < count b
        (gt_break, 3, 12, 4, 13, False),  # count a < count b, pair a < pair b
        (gt_break, 4, 12, 4, 13, False),  # tie, pair a < pair b
        (gt_break, 4, 14, 4, 13, True),  # tie, pair a > pair b
        (gt_break, 4, 14, 4, 14, False),  # equal, doen't matter
        # lt_break
        (lt_break, 5, 12, 4, 13, True),
        (lt_break, 3, 13, 4, 12, False),
        (lt_break, 3, 12, 4, 13, False),
        (lt_break, 4, 12, 4, 13, True),  # tie, pair a < pair b -> a wins with lt_break
        (lt_break, 4, 14, 4, 13, False),
    ],
)
def test_max_node(
    tie_break: TieBreak, count_a: int, pair_a: PackedPair, count_b: int, pair_b: PackedPair, a_lt_b: bool
) -> None:
    make = node_helper(tie_break)
    a, b = make(count_a, pair_a), make(count_b, pair_b)
    assert (a < b) == a_lt_b


def pair_count_helper(pairs: list[int]) -> PackedPairCount:
    return PackedPairCount({pair: count for count, pair in zip(pairs, pairs[1:])})


def heap_helper(pairs: list[int], tie_break: TieBreak = gt_break) -> MaxHeap:
    return MaxHeap(pair_count_helper(pairs), tie_break)


@pytest.mark.parametrize(
    "pairs, expected",
    [
        ([5, 1, 4, 2], 1),
        ([3, 1, 4, 2], 2),
        ([4, 3, 4, 2], 3),
        ([4, 3, 4, 3, 4, 2], 3),  # two equals
    ],
    ids=["greater_count, least_pair", "greater_count, greater_pair", "equal_count, greatest_pair", "equal_pairs"],
)
def test_get_best(pairs: list[int], expected: int) -> None:
    heap = heap_helper(pairs)
    assert heap.get_best() == expected
    assert heap.get_best() != expected  # second time other

@pytest.mark.parametrize(
    "pairs, updates, expected",
    [
        ([5, 1, 4, 2], [1, 2], 2),
        ([5, 1, 4, 2], [-1, 1], 2),
    ],
    ids=["increment_pair", "decrement_pair"],
)
def test_update(pairs: list[int], updates: list[int], expected: int) -> None:
    heap = heap_helper(pairs)
    deltas = pair_count_helper(updates)
    heap.update(deltas)
    assert heap.get_best() == expected
