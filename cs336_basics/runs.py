from __future__ import annotations
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import io

from omegaconf import ValidationError
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

def hash_file_read(dir: str, source: str, n: int = 12) -> str | None:
    path = Path(dir) / source
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()[:n]
    return None

def hash_file_write(dir: str, source: str, state_dict, n: int = 12) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    data = buffer.getvalue()
    hasher = hashlib.sha256()
    hasher.update(data)
    (Path(dir) / source).write_bytes(date)
    return hasher.hexdigest()[:n]


@dataclass
class Manifest:
    dir: Path
    stage: str
    created_at: str
    git_commit: str | None = None
    git_dirty: bool = False
    parent: Path | None = None
    inputs: dict[Path, str] = field(default_factory=dict)   # name -> hash
    outputs: dict[Path, str] = field(default_factory=dict)  # name -> hash

    def save(self) -> None:
        (self.dir / "manifest.json").write_text(json.dumps(asdict(self), default=str, indent=2))

    @classmethod
    def load(cls, source:str) -> "Manifest":
        return cls(**json.loads((Path(source) / "manifest.json").read_text()))


def write_manifest(cfg: Config) -> None:
    Manifest(
        dir=cfg.dir,
        stage=cfg.stage.name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit(),
        parent=cfg.stage.parent,
    ).save()


def resolve_parent(current: str) -> str | None:
    """Given a current dir, find its parent one."""
    return Manifest.load(current).parent

def update_manifest(run_dir: str, **updates) -> None:
    m = Manifest.load(run_dir)
    for k, v in updates.items():
        current = getattr(m, k)
        if isinstance(current, dict):
            current.update(v)
        else:
            setattr(m, k, v)
    m.save()

def validate_input(parent: str, sources:list[str]) -> None:
    m = Manifest.load(parent)
    v = m.inputs.get(sources, None)
    if v:
        for source in v:
            hash = hash_file_read(parent, source)
            if hash and hash != v:
                raise ValidationError(f"Input {parent}/{source} hash {hash} does not match expected {v}")

# run_dir = new_run(kind="train", run_id=job_name, parent_run=cfg.train.vocab_run)

# # ... training loop, periodically ...
# h = save_checkpoint(model, optimizer, step, run_dir / f"step_{step}.pt")
# update_manifest(run_dir, outputs={f"checkpoint_step_{step}": h})

# # ... at the end ...
# update_manifest(run_dir, outputs={"final_loss": final_loss})
