# %%
from collections.abc import Iterable
from enum import Enum
from math import sqrt
from typing import override

import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Init(Enum):
    LINEAR = 1
    EMBEDDING = 2


def t_std(tensor: Tensor) -> float:
    d_out, d_in = tensor.shape
    return sqrt(2 / (d_in + d_out))


def init_it(tensor: Tensor, std: float | None = None) -> None:
    if std is None:
        std = t_std(tensor)
    _ = nn.init.trunc_normal_(tensor, std, a=-3 * std, b=3 * std)


def t_init(tensor: Tensor | tuple[Tensor, ...], cls: Init) -> None:
    if isinstance(tensor, tuple):
        for t in tensor:
            t_init(t, cls)
    else:
        match cls:
            case Init.LINEAR:
                init_it(tensor)
            case Init.EMBEDDING:
                init_it(tensor, 1.0)


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weights: nn.Parameter = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        t_init(self.weights, Init.LINEAR)

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
        t_init(self.weights, Init.EMBEDDING)

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
        t_init((self.W1, self.W2, self.W3), Init.LINEAR)

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
        cos_cached = self.cos_cached[token_pos].to(in_dtype)  # pyright: ignore
        sin_cached = self.sin_cached[token_pos].to(in_dtype)  # pyright: ignore
        x1, x2 = einx.id("... (c d) -> d ... c", x, d=2)
        return einx.id(
            "... (c d), ... (c e) -> ... (c (d+e))",
            x1 * cos_cached - x2 * sin_cached,
            x1 * sin_cached + x2 * cos_cached,
            d=1,
            e=1,
        )  # pyright: ignore


def softmax(x: Tensor, dim: int) -> Tensor:
    y = torch.exp(x - x.max(dim, keepdim=True)[0])
    return y / torch.sum(y, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    d_k = Q.shape[-1]
    QK = d_k ** (-0.5) * einx.dot("... q d_k, ... k d_k -> ... q k", Q, K)
    if mask is not None:
        QK = einx.where("q k, ... q k, ", mask, QK, -torch.inf)  # pyright: ignore
    return einx.dot("... q k, ... k d_k -> ... q d_k", softmax(QK, -1), V)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.W = nn.Parameter(torch.empty(3 * d_model, d_model, device=device, dtype=dtype))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        t_init((self.W, self.Wo), Init.LINEAR)

    def apply_rope(self, head: Tensor, token_pos: Tensor | None = None) -> Tensor:  # noop
        return head

    @override
    def forward(self, x: Tensor, token_pos: Tensor | None = None) -> Tensor:
        seq_len = x.shape[-2]

        Q, K, V = einx.dot("(b d_out) d_in, ... seq d_in -> b ... seq d_out", self.W, x, b=3)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        mhd = torch.cat(
            [
                scaled_dot_product_attention(
                    self.apply_rope(qh, token_pos), self.apply_rope(kh, token_pos), vh, mask=causal_mask
                )
                for qh, kh, vh in zip(
                    torch.split(Q, dim=-1, split_size_or_sections=self.head_dim),
                    torch.split(K, dim=-1, split_size_or_sections=self.head_dim),
                    torch.split(V, dim=-1, split_size_or_sections=self.head_dim),
                )
            ],
            dim=-1,
        )

        return einx.dot("d_out d_in, ... seq d_in -> ... seq d_out", self.Wo, mhd)


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(d_model, num_heads, device, dtype)
        self.rope: RotaryPositionalEmbedding = RotaryPositionalEmbedding(
            theta, self.head_dim, max_seq_len, device=device
        )

    @override
    def apply_rope(self, head: Tensor, token_pos: Tensor | None = None) -> Tensor:
        return self.rope(head, token_pos)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.rms_norm_in = nn.RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.self_attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.rms_norm_out = nn.RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x:Tensor, token_pos: Tensor) -> Tensor:
        x = self.rms_norm_in(x)
        x = self.attn(x, token_pos)
        x = self.rms_norm_out(x)
        return self.ffn(x)

t = TransformerBlock(16, 4, 32, 30, 10_000)
t.state_dict().keys()
