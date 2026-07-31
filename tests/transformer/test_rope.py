import pytest
import torch
import numpy as np
import einx

def test_precompiled():
    theta, seq_length, d_model = 10_000, 128, 20
    d = d_model
    k = torch.arange(d // 2, dtype=torch.float32)
    inv_freq = theta ** (-2 * k / d)
    freqs = torch.outer(torch.arange(seq_length, dtype=torch.float32), inv_freq)
    cosf = torch.cos(freqs)
    sinf = torch.sin(freqs)
    assert freqs.dtype == torch.float32
    assert freqs.shape == (seq_length, d_model // 2)

# %%
import torch

x = torch.randn(5, 6, 5, 4)
# x.shape
dim = 3
# for dim in range(4):
#     print(dim, torch.max(x, dim=dim, keepdim=True)[0].shape)
y = torch.exp(x - x.max(dim, keepdim=True)[0])
z = y / torch.sum(y, dim=dim, keepdim=True)
torch.allclose(z, torch.softmax(x, dim))
# x.max
