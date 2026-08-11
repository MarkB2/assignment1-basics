# %%
from collections.abc import Callable, Iterable
from modulefinder import packagePathMap
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
                grad = p.grad.detach()
                param = p.detach()

                param.add_(grad, alpha =-lr / math.sqrt(t + 1))
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, weight_decay: float = 1e-2, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @override
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 0)
                t += 1

                grad = p.grad.detach()
                param = p.detach()


                lr_t = lr * (1 - beta_2**t)**0.5 / (1 - beta_1**t)

                param -= lr * weight_decay * param

                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                denom = torch.sqrt(v).add_(eps)
                param.addcdiv_(m, denom, value=-lr_t)

                state["m"] = m
                state["v"] = v
                state['t'] = t
            return loss

def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t / T_w * lr_max
    if T_c >= t >= T_w:
        return lr_min + .5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi )) * (lr_max - lr_min)
    return lr_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in grads]))
    if norm >= max_l2_norm:
        for grad in grads:
                grad.data *= max_l2_norm / (norm + eps)

# weights = torch.nn.Parameter(5 * torch.randn(10, 10))
# opt = SGD([weights], lr = 1000)

# for t in range(10):
#     opt.zero_grad()
#     loss = (weights**2).mean()
#     print(loss.cpu().item())
#     loss.backward()
#     opt.step()
