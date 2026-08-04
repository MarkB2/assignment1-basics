# %%
from collections.abc import Callable, Iterable
from typing import override
import math
import torch
from torch import Tensor

class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3):
        super().__init__(params, {"lr" : lr})

    @override
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

weights = torch.nn.Parameter(5 * torch.randn(10, 10))
opt = SGD([weights], lr = 1000)

for t in range(10):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()
