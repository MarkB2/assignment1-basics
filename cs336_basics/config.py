from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING


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
