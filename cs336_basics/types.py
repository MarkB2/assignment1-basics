import json
from collections import Counter, defaultdict
from pathlib import Path

Pair = tuple[bytes, bytes]
PairCount = Counter[Pair]
PairLoc = defaultdict[Pair, set[int]]


def save(path: Path | str, obj: dict[int, str] | list[list[str]]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load(path: Path | str) -> dict[int, str] | list[list[str]]:
    with open(path) as f:
        return json.load(f)  # pyright: ignore[reportAny]


class Vocab(dict[int, bytes]):
    def save(self, path: Path | str) -> None:
        save(path, {k: v.decode("latin1") for k, v in self.items()})

    @classmethod
    def load(cls, path: Path | str):
        return Vocab({int(k): v.encode("latin1") for k, v in load(path).items()})  # pyright: ignore


class Merges(list[Pair]):
    def save(self, path: Path | str) -> None:
        save(path, [[s.decode("latin1") for s in pair] for pair in self])

    @classmethod
    def load(cls, path: Path | str):
        return Merges([Pair([s.encode("latin1") for s in pair]) for pair in load(path)])  # pyright: ignore
