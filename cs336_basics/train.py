# %%
import argparse
from os import PathLike
from typing import IO, BinaryIO
from dataclasses import dataclass, field
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import OmegaConf, MISSING

from cs336_basics.transformer import TransformerLM
from cs336_basics.config import DeviceParams, ModelParams, TrainParams

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


def train(): ...



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to YAML config file",
    )
    # allow ad-hoc overrides, e.g. --override d_model=512 device=cuda
    _ = parser.add_argument(
        "--override", nargs="*", default=[],
        help="Dotlist overrides, e.g. d_model=512 device=cuda",
    )
    return parser.parse_args()


def build_model_params(config_path: str, overrides: list[str]) -> TrainParams:
    schema = OmegaConf.structured(TrainParams)
    cfg = OmegaConf.load(config_path)
    merged = OmegaConf.merge(schema, cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(merged)
    return OmegaConf.to_object(merged)


def main() -> None:
    args = parse_args()
    params: TrainParams = build_model_params(args.config, args.override)

    print(OmegaConf.to_yaml(OmegaConf.structured(params)))  # sanity-check printout

    model = TransformerLM.from_config(params.model)          # your model constructor
    _ = model.to(device=torch.device(params.device.device), dtype=getattr(torch, params.device.dtype))

    if params.mode == "training":
        if params.optimizer is None:
           raise ValueError("optimizer must be set when mode='training'")
        # train(model)
        ...
    else:
        ...


    # training / eval loop goes here
    assert isinstance(model, TransformerLM)



if __name__ == "__main__":
    main()
