from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class RuntimeConfig:
    model_path: str
    device: str = "cpu"
    dtype: str = "bf16"


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.suffix == ".pth":
        return str(path)
    return str(path.with_suffix(".pth"))


def _resolve_dtype(device: str, dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


class RWKVRNN:
    def __init__(self, config: RuntimeConfig):
        self.model_path = _resolve_model_path(config.model_path)
        self.device = torch.device(config.device)
        self.dtype = _resolve_dtype(config.device, config.dtype)
        self.weights = torch.load(self.model_path, map_location="cpu")
        self.n_layer = max(int(key.split(".")[1]) for key in self.weights if key.startswith("blocks.")) + 1
        self.n_embd = int(self.weights["emb.weight"].shape[-1])
        self.vocab_size = int(self.weights["emb.weight"].shape[0])
        self.n_head, self.head_size = self.weights["blocks.0.att.r_k"].shape
        self._prepare_weights()

    def _prepare_weights(self) -> None:
        keys = list(self.weights.keys())
        for key in keys:
            tensor = self.weights[key].squeeze()
            if key.endswith("att.w0"):
                tensor = tensor.float()
            else:
                tensor = tensor.to(dtype=self.dtype)
            if key.endswith("att.r_k"):
                tensor = tensor.flatten()
            self.weights[key] = tensor.to(self.device)

        self.weights["emb.weight"] = F.layer_norm(
            self.weights["emb.weight"],
            (self.n_embd,),
            weight=self.weights["blocks.0.ln0.weight"],
            bias=self.weights["blocks.0.ln0.bias"],
        )
        self.weights["blocks.0.att.v0"] = self.weights["blocks.0.att.a0"]
        self.weights["blocks.0.att.v1"] = self.weights["blocks.0.att.a1"]
        self.weights["blocks.0.att.v2"] = self.weights["blocks.0.att.a2"]

    def zero_state(self) -> list[torch.Tensor]:
        state = [None for _ in range(self.n_layer * 3)]
        for idx in range(self.n_layer):
            state[idx * 3 + 0] = torch.zeros(self.n_embd, dtype=self.dtype, device=self.device)
            state[idx * 3 + 1] = torch.zeros(
                (self.n_embd // self.head_size, self.head_size, self.head_size),
                dtype=torch.float32,
                device=self.device,
            )
            state[idx * 3 + 2] = torch.zeros(self.n_embd, dtype=self.dtype, device=self.device)
        return state

    def clone_state(self, state: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor.clone() for tensor in state]

    def forward(self, token: int, state: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        with torch.no_grad():
            z = self.weights
            x = z["emb.weight"][token]
            v_first = torch.empty_like(x)

            for layer_id in range(self.n_layer):
                bbb = f"blocks.{layer_id}."
                att = f"blocks.{layer_id}.att."
                ffn = f"blocks.{layer_id}.ffn."

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])
                xx, state[layer_id * 3 + 0], state[layer_id * 3 + 1], v_first = time_mixing(
                    layer_id,
                    self.n_head,
                    self.head_size,
                    xx,
                    state[layer_id * 3 + 0],
                    v_first,
                    state[layer_id * 3 + 1],
                    z[att + "x_r"],
                    z[att + "x_w"],
                    z[att + "x_k"],
                    z[att + "x_v"],
                    z[att + "x_a"],
                    z[att + "x_g"],
                    z[att + "w0"],
                    z[att + "w1"],
                    z[att + "w2"],
                    z[att + "a0"],
                    z[att + "a1"],
                    z[att + "a2"],
                    z[att + "v0"],
                    z[att + "v1"],
                    z[att + "v2"],
                    z[att + "g1"],
                    z[att + "g2"],
                    z[att + "k_k"],
                    z[att + "k_a"],
                    z[att + "r_k"],
                    z[att + "key.weight"],
                    z[att + "value.weight"],
                    z[att + "receptance.weight"],
                    z[att + "output.weight"],
                    z[att + "ln_x.weight"],
                    z[att + "ln_x.bias"],
                )
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
                xx, state[layer_id * 3 + 2] = channel_mixing(
                    xx,
                    state[layer_id * 3 + 2],
                    z[ffn + "x_k"],
                    z[ffn + "key.weight"],
                    z[ffn + "value.weight"],
                )
                x = x + xx

            x = F.layer_norm(x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
            logits = z["head.weight"] @ x
            return logits.float(), state


def time_mixing(
    layer_id: int,
    n_head: int,
    head_size: int,
    x: torch.Tensor,
    x_prev: torch.Tensor,
    v_first: torch.Tensor,
    state: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    kw: torch.Tensor,
    vw: torch.Tensor,
    rw: torch.Tensor,
    ow: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x + xx * x_r, x + xx * x_w, x + xx * x_k, x + xx * x_v, x + xx * x_a, x + xx * x_g

    r = rw @ xr
    w = torch.tanh(xw @ w1) @ w2
    k = kw @ xk
    v = vw @ xv
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = k * k_k
    kk = F.normalize(kk.view(n_head, head_size), dim=-1, p=2.0).view(-1)
    k = k * (1 + (a - 1) * k_a)

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = w0 + w.float()
    w = torch.exp(-0.606531 * torch.sigmoid(w))

    vk = v.view(n_head, head_size, 1) @ k.view(n_head, 1, head_size)
    ab = (-kk).view(n_head, head_size, 1) @ (kk * a).view(n_head, 1, head_size)
    state = state * w.view(n_head, 1, head_size) + state @ ab.float() + vk.float()
    out = state.to(dtype=x.dtype) @ r.view(n_head, head_size, 1)

    out = F.group_norm(out.view(1, n_head * head_size), num_groups=n_head, weight=ln_w, bias=ln_b, eps=64e-5)
    out = out.view(n_head * head_size)
    out = out + ((r * k * r_k).view(n_head, head_size).sum(dim=-1, keepdim=True) * v.view(n_head, head_size)).view(
        n_head * head_size
    )
    return ow @ (out * g), x, state, v_first


def channel_mixing(
    x: torch.Tensor, x_prev: torch.Tensor, x_k: torch.Tensor, kw: torch.Tensor, vw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(kw @ k) ** 2
    return vw @ k, x

