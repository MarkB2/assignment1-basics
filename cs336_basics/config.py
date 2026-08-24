from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, get_type_hints, get_origin, get_args, Union, override
from pathlib import Path

from omegaconf import MISSING


@dataclass
class PathCoercingConfig:
    def __post_init__(self):
        hints = get_type_hints(self)
        for name, typ in hints.items():
            value = getattr(self, name)
            if value is None:
                continue

            origin = get_origin(typ)
            args = get_args(typ)

            # plain Path or Path | None
            if typ is Path or (origin is type(int | None) and Path in args):
                if not isinstance(value, Path):
                    setattr(self, name, Path(value))

            # dict[str, Path]
            elif origin is dict and args == (str, Path):
                setattr(self, name, {
                    k: (v if isinstance(v, Path) else Path(v))
                    for k, v in value.items()
                })

            # list[Path]
            elif origin is list and args == (Path,):
                setattr(self, name, [
                    v if isinstance(v, Path) else Path(v) for v in value
                ])

@dataclass
class Config:
    prefix: str = MISSING
    stage: Any = MISSING
    current_path: str = MISSING
    parent_path: str | None = None


@dataclass
class Stage(ABC, PathCoercingConfig):
    name: str = MISSING

    @abstractmethod
    def validate_input(self) -> dict[Path, str]: ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def hash_output(self) -> list[Path]: ...

@dataclass
class VocabConfig(Stage):
    input_path: str = MISSING
    vocab_path: str = MISSING
    vocab_size: int = MISSING
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    num_workers: int = 4
    max_chunk_size: int = 1_000_000

    @override
    def validate_input(self) -> dict[Path, str]:
        return {}

    @override
    def hash_output(self) -> list[Path]:
        return []

    @override
    def run(self): ...

@dataclass
class TokenizerConfig:
    vocab_path: str
    input_data: list[str]
    special_tokens: list[str] | None = None



@dataclass
class DeviceParams:
    device: str = MISSING
    dtype: str = MISSING


@dataclass
class ModelParams:
    vocab_size: int = MISSING
    context_length: int = MISSING
    num_layers: int = MISSING
    d_model: int = MISSING
    num_heads: int = MISSING
    d_ff: int = MISSING
    theta: float = MISSING


@dataclass
class OptParams:
    lr: float = MISSING
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SourceParams:
    dummy: int = 0


@dataclass
class TrainParams:
    device: DeviceParams = field(default_factory=DeviceParams)
    model: ModelParams = field(default_factory=ModelParams)
    mode: str = MISSING
    optimizer: Optional[OptParams] = None
    source: SourceParams = field(default_factory=SourceParams)
