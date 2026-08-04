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
        self.W: nn.Parameter = nn.Parameter(torch.empty((2 * d_ff, d_model), device=device, dtype=dtype))
        self.W2: nn.Parameter = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        t_init((self.W, self.W2), Init.LINEAR)

    @override
    def forward(self, x: Tensor) -> Tensor:
        W1x, W3x = einx.dot("(b d_ff) d_model, ... d_model -> b ... d_ff", self.W, x, b=2)
        return einx.dot(
            "d_model d_ff, ... d_ff -> ... d_model",
            self.W2,
            einx.multiply("... d_ff, ... d_ff, ... d_ff -> ... d_ff", W1x, torch.sigmoid(W1x), W3x),
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
        while mask.dim() < QK.dim():
            mask = mask.unsqueeze(-3)
        QK = torch.where(mask, QK, float("-inf"))
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

        causal_mask = torch.tril(torch.ones(*x.shape[:-2], seq_len, seq_len, dtype=torch.bool, device=x.device))
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
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rms_norm_in = RMSNorm(d_model, device=device, dtype=dtype)
        self.self_attn = MultiHeadSelfAttentionWithRoPE(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype
        )
        self.rms_norm_out = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    @override
    def forward(self, x: Tensor, token_pos: Tensor | None = None) -> Tensor:
        if token_pos is None:
            token_pos = torch.arange(x.shape[-2], device=x.device)
        y = self.rms_norm_in(x)
        x = self.self_attn(y, token_pos) + x
        y = self.rms_norm_out(x)
        return self.ffn(y) + x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        return self.linear(x)


def cross_entropy(x: Tensor, targets: Tensor) -> Tensor:
    x = x - einx.max("... d -> ... 1", x)
    x = x - torch.log(einx.sum("... d -> ... 1", torch.exp(x)))
    return -torch.mean(einx.get_at("b [p], b -> b", x, targets))


# cross_entropy(torch.randn(2,3,5))
