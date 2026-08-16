from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pathlib import Path

from omegaconf import MISSING


@dataclass
class Config:
    prefix: str = MISSING
    stage: Any = MISSING
    dir: Path = MISSING

    def __post_init__(self):
        self.dir = Path(self.dir)

@dataclass
class Stage(ABC):
    name: str = MISSING
    parent: str | None = None
    sources: list[Path] | None = None

    @abstractmethod
    def run(self): ...


@dataclass
class VocabConfig(Stage):
    input_path: str = MISSING
    vocab_path: str = MISSING
    vocab_size: int = MISSING
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    num_workers: int = 4
    max_chunk_size: int = 1_000_000

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
