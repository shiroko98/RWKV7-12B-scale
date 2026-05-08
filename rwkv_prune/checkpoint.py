from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LayerInfo:
    layer_ids: list[int]
    layer_param_counts: dict[int, int]
    total_params: int
    non_block_params: int
    n_embd: int
    vocab_size: int
    head_size: int
    n_head: int

    @property
    def n_layer(self) -> int:
        return len(self.layer_ids)


def load_checkpoint(model_path: str | Path) -> tuple[dict[str, Any], OrderedDict, bool]:
    raw = torch.load(str(model_path), map_location="cpu")
    if isinstance(raw, dict) and "model" in raw:
        state_dict = raw["model"]
        wrapped = True
    else:
        state_dict = raw
        wrapped = False
    if not isinstance(state_dict, OrderedDict):
        state_dict = OrderedDict(state_dict)
    return raw, state_dict, wrapped


def inspect_checkpoint(state_dict: OrderedDict) -> LayerInfo:
    layer_ids = sorted({int(key.split(".")[1]) for key in state_dict if key.startswith("blocks.")})
    if not layer_ids:
        raise ValueError("No transformer blocks found in checkpoint.")

    layer_param_counts: dict[int, int] = {}
    total_params = 0
    non_block_params = 0

    for key, value in state_dict.items():
        param_count = value.numel()
        total_params += param_count
        if key.startswith("blocks."):
            layer_id = int(key.split(".")[1])
            layer_param_counts[layer_id] = layer_param_counts.get(layer_id, 0) + param_count
        else:
            non_block_params += param_count

    emb = state_dict["emb.weight"]
    head_size = int(state_dict["blocks.0.att.r_k"].shape[-1])
    n_head = int(state_dict["blocks.0.att.r_k"].shape[0])

    return LayerInfo(
        layer_ids=layer_ids,
        layer_param_counts=layer_param_counts,
        total_params=total_params,
        non_block_params=non_block_params,
        n_embd=int(emb.shape[-1]),
        vocab_size=int(emb.shape[0]),
        head_size=head_size,
        n_head=n_head,
    )


def estimate_parameter_count(info: LayerInfo, kept_layers: list[int]) -> int:
    return info.non_block_params + sum(info.layer_param_counts[layer_id] for layer_id in kept_layers)


def build_layer_map(layer_ids: list[int], drop_layers: set[int]) -> dict[int, int]:
    layer_map: dict[int, int] = {}
    new_idx = 0
    for old_idx in layer_ids:
        if old_idx in drop_layers:
            continue
        layer_map[old_idx] = new_idx
        new_idx += 1
    return layer_map


def prune_state_dict(state_dict: OrderedDict, drop_layers: set[int], layer_map: dict[int, int]) -> tuple[OrderedDict, int, int]:
    new_state = OrderedDict()
    removed_params = 0
    kept_params = 0

    for key, value in state_dict.items():
        if key.startswith("blocks."):
            parts = key.split(".")
            old_layer = int(parts[1])
            if old_layer in drop_layers:
                removed_params += value.numel()
                continue
            parts[1] = str(layer_map[old_layer])
            new_key = ".".join(parts)
            new_state[new_key] = value
            kept_params += value.numel()
        else:
            new_state[key] = value
            kept_params += value.numel()

    return new_state, removed_params, kept_params


def _update_nested_metadata(obj: dict[str, Any], new_n_layer: int) -> None:
    if "args" in obj:
        args_obj = obj["args"]
        if hasattr(args_obj, "n_layer"):
            args_obj.n_layer = new_n_layer
    if "config" in obj and isinstance(obj["config"], dict) and "n_layer" in obj["config"]:
        obj["config"]["n_layer"] = new_n_layer


def save_pruned_checkpoint(raw_obj: dict[str, Any], wrapped: bool, new_state: OrderedDict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if wrapped:
        if not isinstance(raw_obj, dict):
            raise TypeError("Wrapped checkpoint must be a dictionary.")
        raw_obj["model"] = new_state
        _update_nested_metadata(raw_obj, max(int(key.split(".")[1]) for key in new_state if key.startswith("blocks.")) + 1)
        save_obj = raw_obj
    else:
        save_obj = new_state
    torch.save(save_obj, str(output_path))

