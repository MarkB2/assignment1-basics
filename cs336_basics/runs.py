import hashlib
import io
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from omegaconf import ValidationError

from cs336_basics.config import Config, PathCoercingConfig


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(out.strip())


def file_hash(source: Path, n: int = 12) -> str | None:
    if source.exists():
        return hashlib.sha256(source.read_bytes()).hexdigest()[:n]
    return None


@dataclass
class Manifest:
    dir: str
    stage: str
    created_at: str
    git_commit: str | None = None
    git_dirty: bool = False
    parent: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)  # name -> hash
    outputs: dict[str, str] = field(default_factory=dict)  # name -> hash

    def save(self) -> None:
        (Path(self.dir) / "manifest.json").write_text(json.dumps(asdict(self), default=str, indent=2))

    @classmethod
    def load(cls, source: str) -> "Manifest":
        return cls(**json.loads((Path(source) / "manifest.json").read_text()))


def write_manifest(cfg: Config) -> None:
    Manifest(
        dir=cfg.current_path,
        stage=cfg.stage.name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit(),
        parent=cfg.parent_path
    ).save()


def resolve_parent(current: Path) -> Path | None:
    """Given a current dir, find its parent one."""
    return Manifest.load(current).parent


def update_manifest(current_path: Path, outputs: list[Path]) -> None:
    manifest = Manifest.load(current_path)
    for output in outputs:
        hash = file_hash(output)
        if hash is not None:
            manifest.outputs[output] = hash
    manifest.save()


def validate_input(parent: Path, sources: list[str]) -> None:
    m = Manifest.load(parent)
    v = m.inputs.get(sources, None)
    if v:
        for source in v:
            hash = hash_file_read(parent, source)
            if hash and hash != v:
                raise ValidationError(f"Input {parent}/{source} hash {hash} does not match expected {v}")


class Runner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self) -> None:
        write_manifest(self.cfg)
