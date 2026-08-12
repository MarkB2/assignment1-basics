from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pathlib import Path

from omegaconf import MISSING


class Stage(str, Enum):
    VOCAB = "vocab"
    TOKENIZE = "tokenize"
    TRAIN = "train"
    EVAL = "eval"

@dataclass
class VocabConfig:
    input_path: str | Path = MISSING
    vocab_path: str | Path = MISSING
    # vocab_size: int = 500
    # special_tokens: list[str] = field(default_factory=list) # =["<|endoftext|>"])
    # num_workers: int = 4
    # max_chunk_size: int = 1_000_000


@dataclass
class Config:
    job_type: Stage = MISSING
    job_name: str = MISSING
    parent_name: str = MISSING
    vocab: Optional[VocabConfig] = None
    params: Optional[Any] = None


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
