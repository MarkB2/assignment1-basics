from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import regex as re

from .model_types import Pretoken, SpecialToken

GPT4_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def choose_num_chunks(file_size: int, max_chunk_size: int) -> int:
    return max(1, -(-file_size // max_chunk_size))  # ceil div, no cpu cap needed


def find_chunk_boundaries(file, special_token: bytes = b"<|endoftext|>", max_chunk_size: int = 50_000_000) -> list[int]:
    file.seek(0, 2)
    file_size = file.tell()
    num_chunks = choose_num_chunks(file_size, max_chunk_size)
    chunk_size = file_size // num_chunks

    boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    boundaries[-1] = file_size

    for i in range(1, len(boundaries) - 1):
        pos = boundaries[i]
        file.seek(pos)
        # read forward in a small window until we find the special token
        window = 4096
        while True:
            chunk = file.read(window)
            if not chunk:
                boundaries[i] = file_size
                break
            found = chunk.find(special_token)
            if found != -1:
                boundaries[i] = pos + found
                break
            pos += window
            file.seek(pos)

    return sorted(set(boundaries))


class Pretokenizer:
    def __init__(
        self, pat: str = GPT4_PAT, special_tokens: list[str] | None = None, keep_special_tokens: bool = False
    ) -> None:
        self.pat: re.Pattern[str] = re.compile(pat)
        special_tokens = sorted(special_tokens or ["<|endoftext|>"], key=len, reverse=True)
        split_pat = "|".join([re.escape(b) for b in special_tokens])
        self.split_pat: re.Pattern[str] = re.compile(rf"({split_pat})")
        self.special_tokens: set[str] = set(special_tokens)
        self.keep_special_tokens: bool = keep_special_tokens

    def iter_tokens(self, text: str) -> Iterator[Pretoken]:
        for part in re.split(self.split_pat, text):
            if part in self.special_tokens:
                if self.keep_special_tokens:
                    yield SpecialToken(part)
            else:
                for match in re.finditer(self.pat, part):
                    yield match.group()

    def iter_file(
        self,
        source: str | Path,
        max_chunk_size: int = 50_000_000,
        num_workers: int = 1,
    ) -> Iterator[Pretoken]:
        with open(source, "rb") as file:
            boundaries = find_chunk_boundaries(file, max_chunk_size=max_chunk_size)
        tasks = [FileRange(self, source, start, end) for start, end in zip(boundaries, boundaries[1:])]
        if num_workers == 1:
            for task in tasks:
                yield from _iter_file_range(task)
        else:
            with Pool(num_workers) as pool:
                for chunk in pool.imap(_list_file_range, tasks):
                    yield from chunk


class FileRange(NamedTuple):
    pretokenizer: Pretokenizer
    source: str | Path
    start: int
    end: int


def _iter_file_range(args: FileRange) -> Iterator[Pretoken]:
    pretokenizer, source, start, end = args
    with open(source, "rb") as file:
        _ = file.seek(start)
        chunk_bytes = file.read(end - start)
        yield from pretokenizer.iter_tokens(chunk_bytes.decode("utf-8", errors="ignore"))


def _list_file_range(args: FileRange) -> list[Pretoken]:
    return list(_iter_file_range(args))
