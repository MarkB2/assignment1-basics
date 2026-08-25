import hashlib
import io
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from omegaconf import ValidationError

from cs336_basics.config import Config, PathCoercingConfig
from cs336_basics.stages import IOSpec, Stage


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
        (Path(self.dir) / "manifest.json").write_text(json.dumps(asdict(self), default=str, indent=2))

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
        hash_dict={b.path: b.hash for b in input_spec},
    ).save()


def resolve_parent(current: Path) -> Path | None:
    """Given a current dir, find its parent one."""
    return Manifest.load(current).parent


def update_manifest(current_path: str, outputs: list[str]) -> None:
    manifest = Manifest.load(current_path)
    for output in outputs:
        hash = file_hash(output)
        if hash is not None:
            manifest.outputs[output] = hash
    manifest.save()


def get_source_hashes(parent_path: str | None) -> dict[str, str]:
    if parent_path is None:
        return {}
    manifest = Manifest.load(parent_path)
    manifest.inputs.update(manifest.outputs)
    return manifest.inputs


def validate_inputs(source_hashes: dict[str, str], inputs: dict[str, bool]) -> dict[str, str]:
    input_hashes = {}
    for input, validate in inputs.items():
        if validate:
            source_hash = source_hashes.get(input, None)
            if source_hash is None:
                raise ValidationError(f"Input {input} does not has hash")
            else:
                hash = file_hash(input)
                if hash != source_hash:
                    raise ValidationError(f"Input {input} hash {hash} does not match expected {source_hash}")
                else:
                    input_hashes[input] = hash
        else:
            input_hashes[input] = None
    return input_hashes


class Runner:
    def __init__(self, config: Config, stage: Stage) -> None:
        self.config: Config = config
        self.stage: Stage = stage

    def setup(self):
        inputs = self.stage.get_input_specs()
        if self.config.parent_path is not None:
            source_hashes = {}
        else:
            source_hashes = get_source_hashes(self.config.parent_path)
        input_hashes = validate_inputs(source_hashes, inputs)
        write_manifest(self.cfg, input_hashes)

    # def after(self):
    #     outputs = self.cfg.stage.get_outputs()
    #     update_manifest(self.cfg.current_path, outputs)

    def run(self) -> None:
        # self.before()
        # self.cfg.stage.run()
        # self.after()
        # cls = Stage(self.cfg,stage)
        ...
