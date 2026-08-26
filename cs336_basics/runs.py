import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from omegaconf import ValidationError

from cs336_basics.config import Config
from cs336_basics.stages import IOSpec, IOSpecList, Stage


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(out.strip())


def file_hash(source: str, n: int = 12) -> str | None:
    path = Path(source)
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()[:n]
    return None


def compute_hash(specs: IOSpecList) -> IOSpecList:
    return [replace(spec, hash=file_hash(spec.path)) for spec in specs]


def get_expected_hashes(parent_path: str | None = None) -> dict[str, str | None]:
    if parent_path is not None:
        m = Manifest.load(parent_path)
        return m.hash_dict
    return {}


def validate(specs: IOSpecList, parent: str | None) -> IOSpecList:
    hashed = compute_hash(specs)
    expected_hashes = get_expected_hashes(parent)
    failures: list[str] = []
    for spec in hashed:
        if spec.needs_validate and expected_hashes.get(spec.path) != spec.hash:
            failures.append(spec.path)
    if failures:
        raise ValidationError(f"{len(failures)} file fail hash validation: {', '.join(failures)}")
    return hashed


def asdict(specs: IOSpecList) -> dict[str, str | None]:
    return {s.path: s.hash for s in specs}


@dataclass
class Manifest:
    dir: str
    stage: str
    created_at: str
    git_commit: str | None = None
    git_dirty: bool = False
    parent: str | None = None
    hash_dict: dict[str, str | None] = field(default_factory=dict)  # name -> hash

    def save(self) -> None:
        _ = (Path(self.dir) / "manifest.json").write_text(json.dumps(dataclasses.asdict(self), default=str, indent=2))

    @classmethod
    def load(cls, source: str) -> "Manifest":
        return cls(**json.loads((Path(source) / "manifest.json").read_text()))


def write_manifest(cfg: Config, input_spec: list[IOSpec]) -> None:
    Manifest(
        dir=cfg.current_path,
        stage=cfg.stage,
        created_at=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit(),
        parent=cfg.parent_path,
        hash_dict=asdict(input_spec),
    ).save()


def resolve_parent(current: str) -> str | None:
    """Given a current dir, find its parent one."""
    return Manifest.load(current).parent


def update_manifest(output_spec: IOSpecList, current_path: str) -> None:
    manifest = Manifest.load(current_path)
    manifest.hash_dict.update(asdict(output_spec))
    manifest.save()


class Runner:
    def __init__(self, config: Config, stage: Stage) -> None:
        self.config: Config = config
        self.stage: Stage = stage

    def setup_manifest(self):
        inputs = self.stage.get_input_specs()
        hashed = validate(inputs, self.config.parent_path)
        write_manifest(self.config, hashed)

    def update_manifest(self):
        outputs = self.stage.get_output_specs()
        hashed = compute_hash(outputs)
        update_manifest(hashed, self.config.current_path)

    def run(self) -> None:
        self.setup_manifest()
        self.stage.run()
        self.update_manifest()
