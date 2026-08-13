from __future__ import annotations
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from cs336_basics.config import Config

DEFAULT_PATH = Path(".")

def git_commit() -> str | None:
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


@dataclass
class Manifest:
    dir: str
    stage: str
    created_at: str
    git_commit: str | None = None
    git_dirty: bool = False
    parent: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)   # name -> hash
    outputs: dict[str, str] = field(default_factory=dict)  # name -> hash

    def save(self) -> None:
        (Path(self.dir) / "manifest.json").write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, source:str) -> "Manifest":
        return cls(**json.loads((Path(source) / "manifest.json").read_text()))


def write_manifest(config: Config) -> None:
    Manifest(
        dir=config.dir,
        stage=config.stage,
        created_at=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit(),
        parent=config.parent,
    ).save()


def resolve_parent(current: str) -> str | None:
    """Given a current dir, find its parent one."""
    return Manifest.load(current).parent

def update_manifest(run_dir: Path, **updates) -> None:
    m = Manifest.load(run_dir)
    for k, v in updates.items():
        getattr(m, k).update(v) if isinstance(getattr(m, k), dict) else setattr(m, k, v)
    m.save(run_dir)

# run_dir = new_run(kind="train", run_id=job_name, parent_run=cfg.train.vocab_run)

# # ... training loop, periodically ...
# h = save_checkpoint(model, optimizer, step, run_dir / f"step_{step}.pt")
# update_manifest(run_dir, outputs={f"checkpoint_step_{step}": h})

# # ... at the end ...
# update_manifest(run_dir, outputs={"final_loss": final_loss})
