# %%
from os import PathLike
from typing import IO, BinaryIO
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import MISSING

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str | None = None) -> tuple[Tensor, Tensor]:
    window = sliding_window_view(x, context_length + 1)
    starts = np.random.randint(0, x.size - context_length, size=batch_size)
    batch = window[starts]
    return torch.from_numpy(batch[:, :-1]).to(device), torch.from_numpy(batch[:, 1:]).to(device)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | PathLike | BinaryIO | IO[bytes]):
    d = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(d, out)

def load_checkpoint(src: str | PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer):
    d = torch.load(src)
    _ = model.load_state_dict(d["model"])
    optimizer.load_state_dict(d["optimizer"])
    return d["iteration"]

@dataclass
class ModelParms:
    vocab_size: int = MISSING
    context_length: int = MISSING
    num_layers: int = MISSING
    d_model: int = MISSING
    num_heads: int = MISSING
    d_ff: int = MISSING
    theta: float = MISSING
    device: str = MISSING
    dtype: str = MISSING
