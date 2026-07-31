from math import sqrt
from typing import override

import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor


def linear_init(d_in: int, d_out: int) -> dict[str, float]:
    std = sqrt(2 / (d_in + d_out))
    return {"std": std, "a": -3 * std, "b": 3 * std}


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weights: nn.Parameter = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights, **linear_init(in_features, out_features))  # pyright: ignore

    @override
    def forward(self, x: Tensor) -> Tensor:
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
        self.weights: nn.Parameter = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weights, a=-3, b=3)  # pyright: ignore

    @override
    def forward(self, x: torch.LongTensor) -> Tensor:
        return einx.get_at("[vocab_size] d_model, b seq -> b seq d_model", self.weights, x)


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.eps: float = eps
        self.gains: nn.Parameter = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))

    @override
    def forward(self, x: Tensor) -> Tensor:
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
        self.W1: nn.Parameter = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W2: nn.Parameter = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.W3: nn.Parameter = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        init_params = linear_init(d_model, d_ff)
        nn.init.trunc_normal_(self.W1, **init_params)  # pyright: ignore
        nn.init.trunc_normal_(self.W2, **init_params)  # pyright: ignore
        nn.init.trunc_normal_(self.W3, **init_params)  # pyright: ignore

    @override
    def forward(self, x: Tensor) -> Tensor:
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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_model: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        k = torch.arange(d_model // 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-2 * k / d_model)
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32, device=device), inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    @override
    def forward(self, x: Tensor, token_pos: Tensor) -> Tensor:
        in_dtype = x.dtype
        cos_cached = self.cos_cached[token_pos].to(in_dtype) # pyright: ignore
        sin_cached = self.sin_cached[token_pos].to(in_dtype) # pyright: ignore
        x1, x2 = einx.id("... (c d) -> d ... c", x, d=2)
        return einx.id(
            "... (c d), ... (c e) -> ... (c (d+e))",
            x1 * cos_cached - x2 * sin_cached,
            x1 * sin_cached + x2 * cos_cached,
            d=1,
            e=1,
        ) # pyright: ignore

def softmax(x: Tensor, dim: int) -> Tensor:
    y = torch.exp(x - x.max(dim, keepdim=True)[0])
    return y / torch.sum(y, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    d_k = Q.shape[-1]
    QK = d_k ** (-0.5) * einx.dot("... q d_k, ... k d_k -> ... q k", Q, K)
    if mask is not None:
        QK = einx.where("... q k, ... q k, ", mask, QK, -torch.inf) # pyright: ignore
    return einx.dot("... q k, ... k d_k -> ... q d_k", softmax(QK, -1), V)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W = nn.Parameter(torch.empty(3 * d_model, d_model, device=device, dtype=dtype))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, **linear_init(3 * d_model, d_model)) # pyright: ignore
        nn.init.trunc_normal_(self.Wo, **linear_init(d_model, d_model)) # pyright: ignore

    @override
    def forward(self, x: Tensor) -> Tensor:
