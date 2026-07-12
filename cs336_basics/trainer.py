import gc
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from .max_heap import PairMaxHeap
from .pretokenizer import Pretokenizer
from .tokens import Word
from .types import Merges, Pair, PairCount, PairLoc, Pretoken, Vocab


class BPETrainer:
    def __init__(
        self,
        iterator: Iterator[Pretoken],
        vocab_size: int,
        special_tokens: list[str],
    ) -> None:
        self.iterator: Iterator[Pretoken] = iterator
        self.vocab_size: int = vocab_size
        vocab = Vocab({i: bytes(special_token, encoding="utf-8") for i, special_token in enumerate(special_tokens)})
        for j in range(256):
            vocab[len(vocab)] = bytes([j])
        self.vocab: Vocab = vocab
        self.merges: Merges = Merges()

    def build(self) -> tuple[list[Word], PairCount, PairLoc]:
        corpus: Counter[Pretoken] = Counter(self.iterator)
        words = [Word(word, freq) for word, freq in corpus.items()]
        pair_counts, pair_locs = self.get_counts(words)
        return words, pair_counts, pair_locs

    @classmethod
    def get_counts(cls, words: list[Word]) -> tuple[PairCount, PairLoc]:
        pair_counts, pair_locs = PairCount(), PairLoc()
        for loc, word in enumerate(words):
            for pair in word.pairs():
                pair_counts[pair] += word.freq
                pair_locs[pair].add(loc)
        return pair_counts, pair_locs

    def merge(self) -> tuple[Vocab, Merges]:
        words, pair_counts, pair_locs = self.build()
        _ = gc.collect()
        heap, vocab, merges = PairMaxHeap(pair_counts), self.vocab, self.merges
        i, pair = len(vocab), heap.get_best()
        while pair and i < self.vocab_size:
            global_updates: Counter[Pair] = Counter()
            merges.append(pair)
            a, b = pair
            vocab[i] = a + b
            for loc in list(pair_locs[pair]):
                updates = words[loc].merge(pair)
                for updated_pair in updates.keys():
                    if updated_pair in words[loc].pairs():
                        pair_locs[updated_pair].add(loc)
                    else:
                        pair_locs[updated_pair].discard(loc)
                global_updates.update(updates)
            heap.update(global_updates)
            i += 1
            pair = heap.get_best()
            if i % 5000 == 0:
                keys = [k for k, v in pair_counts.items() if v <= 0]
                for key in keys:
                    del pair_counts[key]
                keys = [k for k, v in pair_locs.items() if v == set()]
                for key in keys:
                    del pair_locs[key]
                _ = gc.collect()

        return vocab, merges


def train(
    input_path: str | Path, vocab_size: int = 32_000, special_tokens: list[str] = ["<|endoftext|>"]
) -> tuple[dict[int, bytes], list[Pair]]:
    iterator = Pretokenizer(special_tokens=special_tokens).iter_file(input_path)
    return BPETrainer(iterator, vocab_size, special_tokens).merge()


def train_and_save(
    input_path: str | Path,
    vocab_path: str | Path,
    merges_path: str | Path,
    vocab_size: int = 32_000,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_workers: int = 4,
    max_chunk_size: int = 1_000_000,
) -> None:
    iterator = Pretokenizer(special_tokens=special_tokens).iter_file(input_path, max_chunk_size, num_workers)
    vocab, merges = BPETrainer(iterator, vocab_size, special_tokens).merge()
    vocab.save(vocab_path)
    merges.save(merges_path)
