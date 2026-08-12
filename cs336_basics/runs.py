from __future__ import annotations
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

RUNS_DIR = Path("runs")


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None

def git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(out.strip())

def hash_file(path: Path, n: int = 12) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


# @dataclass
# class Manifest:
#     run_id: str
#     kind: str                       # "tokenizer" | "training" | "eval"
#     created_at: str
#     git_commit: Optional[str] = None
#     parent_run: Optional[str] = None    # e.g. training -> its tokenizer run
#     config_path: Optional[str] = None   # relative path to resolved config
#     extra: dict = field(default_factory=dict)

#     def save(self, run_dir: Path) -> None:
#         (run_dir / "manifest.json").write_text(json.dumps(self.__dict__, indent=2))

#     @classmethod
#     def load(cls, run_dir: Path) -> "Manifest":
#         return cls(**json.loads((run_dir / "manifest.json").read_text()))


def new_run(kind: str, run_id: str, parent_run: Optional[str] = None) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    Manifest(
        run_id=run_id,
        kind=kind,
        created_at=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit(),
        parent_run=parent_run,
    ).save(run_dir)
    return run_dir


def resolve_tokenizer(training_run_id: str) -> Path:
    """Given a training run, find its tokenizer's vocab/merges dir."""
    m = Manifest.load(RUNS_DIR / training_run_id)
    if m.parent_run is None:
        raise ValueError(f"{training_run_id} has no linked tokenizer run")
    return RUNS_DIR / m.parent_run

@dataclass
class Manifest:
    run_id: str
    kind: str
    created_at: str
    git_commit: Optional[str] = None
    git_dirty: bool = False
    parent_run: Optional[str] = None
    inputs: dict[str, str] = field(default_factory=dict)   # name -> hash
    outputs: dict[str, str] = field(default_factory=dict)  # name -> hash

    def save(self, run_dir: Path) -> None:
        (run_dir / "manifest.json").write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> "Manifest":
        return cls(**json.loads((run_dir / "manifest.json").read_text()))
