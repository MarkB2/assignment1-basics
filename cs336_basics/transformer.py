from math import sqrt
from typing import override

import einx
import torch
import torch.nn as nn


def linear_init(d_in: int, d_out: int) -> dict[str, float]:
    std = sqrt(2 / (d_in + d_out))
    return {"std": std, "a": -3 * std, "b": 3 * std}


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weights: nn.Parameter = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights, **linear_init(self.in_features, self.out_features))  # pyright: ignore

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("d_out [d_in], ... [d_in] -> ... d_out", self.weights, x)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.weights: nn.Parameter = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weights, a=-3, b=3)  # pyright: ignore

    @override
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return einx.get_at("[vocab_size] d_model, b seq -> b seq d_model", self.weights, x)


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.gains: nn.Parameter = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(einx.mean("... d_model -> ...", torch.square(x)) + self.eps)
        x = einx.multiply("... d_model, d_model, ... -> ... d_model", x, self.gains, 1 / rms)
        return x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.d_ff: int = d_ff  # int(round(self.d_model * 8 / 3.0)) * 64
        # assert self.d_ff > 0, "Feed Forward dimention should be positive"
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.W1: nn.Parameter = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W2: nn.Parameter = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.W3: nn.Parameter = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        init_params = linear_init(self.d_model, self.d_ff)
        nn.init.trunc_normal_(self.W1, **init_params)  # pyright: ignore
        nn.init.trunc_normal_(self.W2, **init_params)  # pyright: ignore
        nn.init.trunc_normal_(self.W3, **init_params)  # pyright: ignore

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x = einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.W1, x)
        return einx.dot(
            "d_model d_ff, ... d_ff -> ... d_model",
            self.W2,
            einx.multiply(
                "... d_ff, ... d_ff, ... d_ff -> ... d_ff",
                W1x,
                torch.sigmoid(W1x),
                einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.W3, x),
            ),
        )
