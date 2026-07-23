import gc
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from .max_heap import MaxHeap
from .pretokenizer import Pretokenizer
from .tokens import Word
from .vocab import Vocab, make_heap
from .types import Pretoken, PackedPairCount, PackedPairLoc, unpack_pair
# import tracemalloc



class BPETrainer:
    def __init__(
        self,
        iterator: Iterator[Pretoken],
        vocab_size: int,
        special_tokens: list[str],
    ) -> None:
        self.iterator: Iterator[Pretoken] = iterator
        self.vocab_size: int = vocab_size
        self.vocab: Vocab = Vocab(special_tokens)

    def build(self) -> tuple[list[Word], PackedPairCount, PackedPairLoc]:
        corpus: Counter[Pretoken] = Counter(self.iterator)
        words = [Word(self.vocab.to_ids(cast(str, word)), freq) for word, freq in corpus.items()]
        pair_counts, pair_locs = self.get_counts(words)
        return words, pair_counts, pair_locs

    @classmethod
    def get_counts(cls, words: list[Word]) -> tuple[PackedPairCount, PackedPairLoc]:
        pair_counts, pair_locs = PackedPairCount(), PackedPairLoc()
        for loc, word in enumerate(words):
            for pair in word.pairs():
                p = int(pair)
                pair_counts[p] += word.freq
                pair_locs[p].add(loc)
        return pair_counts, pair_locs

    def merge(self) -> Vocab:
        words, pair_counts, pair_locs = self.build()
        _ = gc.collect()
        heap, vocab = make_heap(pair_counts, self.vocab), self.vocab
        pair = heap.get_best()
        i = len(vocab)
        #
        # tracemalloc.start()
        #
        while pair and i < self.vocab_size:
            global_updates = PackedPairCount()
            merged_id = vocab.add_merge(pair)
            for loc in list(pair_locs[pair]):
                updates = words[loc].merge(pair, merged_id)
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
                # pair_counts, pair_locs = self.get_counts(words)
                # heap = PairMaxHeap(pair_counts)
                keys = [k for k, v in pair_counts.items() if v <= 0]
                for key in keys:
                    del pair_counts[key]
                keys = [k for k, v in pair_locs.items() if v == set()]
                for key in keys:
                    del pair_locs[key]
                # heap.build()
                # current, peak = tracemalloc.get_traced_memory()
                # print(f"Before GC.collect current={current/1e6:.1f}MB peak={peak/1e6:.1f}MB")
                _ = gc.collect()
                # current, peak = tracemalloc.get_traced_memory()
                # print(f"After GC.collect current={current/1e6:.1f}MB peak={peak/1e6:.1f}MB")

        return vocab


def train(
    input_path: str | Path, vocab_size: int = 32_000, special_tokens: list[str] = ["<|endoftext|>"]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    iterator = Pretokenizer(special_tokens=special_tokens).iter_file(input_path)
    vocab = BPETrainer(iterator, vocab_size, special_tokens).merge()
    return vocab._forward, [tuple([vocab.bytes_for(b) for b in unpack_pair(pair)]) for pair in vocab.merges()]  # pyright ignore


def train_and_save(
    input_path: str | Path,
    vocab_path: str | Path,
    vocab_size: int = 32_000,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_workers: int = 4,
    max_chunk_size: int = 1_000_000,
) -> None:
    iterator = Pretokenizer(special_tokens=special_tokens).iter_file(input_path, max_chunk_size, num_workers)
    vocab = BPETrainer(iterator, vocab_size, special_tokens).merge()
    vocab.save(vocab_path)
