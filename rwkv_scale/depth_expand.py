from __future__ import annotations

import argparse
import json
import math
import re
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
    n_head: int
    head_size: int

    @property
    def n_layer(self) -> int:
        return len(self.layer_ids)


@dataclass
class ExpansionPlan:
    strategy: str
    target_layers: int
    insertion_count: int
    inserted_after_layers: list[int]
    insertion_ops: dict[int, str]
    scores: dict[int, float]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth-only RWKV expansion by copy / interpolation.")
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "uniform_interp",
            "uniform_copy",
            "hybrid_alt",
            "tail_interp",
            "tail_copy",
            "importance_copy",
            "boundary_delta_copy",
        ],
    )
    parser.add_argument("--target-layers", type=int, default=56)
    parser.add_argument("--alpha", type=float, default=0.5, help="Interpolation weight for the left layer.")
    parser.add_argument("--metadata-out")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


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
    layer_param_counts: dict[int, int] = {}
    total_params = 0
    non_block_params = 0

    for key, value in state_dict.items():
        count = value.numel()
        total_params += count
        if key.startswith("blocks."):
            layer_id = int(key.split(".")[1])
            layer_param_counts[layer_id] = layer_param_counts.get(layer_id, 0) + count
        else:
            non_block_params += count

    emb = state_dict["emb.weight"]
    n_head, head_size = state_dict["blocks.0.att.r_k"].shape
    return LayerInfo(
        layer_ids=layer_ids,
        layer_param_counts=layer_param_counts,
        total_params=total_params,
        non_block_params=non_block_params,
        n_embd=int(emb.shape[1]),
        vocab_size=int(emb.shape[0]),
        n_head=int(n_head),
        head_size=int(head_size),
    )


def _uniform_positions(count: int, pick_count: int) -> list[int]:
    if pick_count <= 0:
        return []
    if pick_count >= count:
        return list(range(count))
    positions = []
    for idx in range(pick_count):
        pos = round((idx + 1) * (count + 1) / (pick_count + 1)) - 1
        pos = min(max(pos, 0), count - 1)
        positions.append(pos)
    seen = set()
    deduped = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            deduped.append(pos)
    cur = 0
    while len(deduped) < pick_count:
        if cur not in seen:
            seen.add(cur)
            deduped.append(cur)
        cur += 1
    return sorted(deduped)


def _sample_tensor(tensor: torch.Tensor, sample_size: int = 256) -> torch.Tensor:
    flat = tensor.reshape(-1).float()
    if flat.numel() <= sample_size:
        return flat
    idx = torch.linspace(0, flat.numel() - 1, steps=sample_size, dtype=torch.long)
    return flat.index_select(0, idx)


def _layer_signature(state_dict: OrderedDict, layer_id: int) -> torch.Tensor:
    keys = [
        f"blocks.{layer_id}.att.key.weight",
        f"blocks.{layer_id}.att.value.weight",
        f"blocks.{layer_id}.ffn.key.weight",
        f"blocks.{layer_id}.ffn.value.weight",
    ]
    samples = [_sample_tensor(state_dict[key], sample_size=128) for key in keys]
    stats = []
    for key in keys:
        tensor = state_dict[key].float()
        stats.extend(
            [
                tensor.mean(),
                tensor.std(),
                tensor.norm() / math.sqrt(tensor.numel()),
            ]
        )
    return torch.cat(samples + [torch.stack(stats)])


def _importance_score(state_dict: OrderedDict, layer_id: int) -> float:
    keys = [
        f"blocks.{layer_id}.att.key.weight",
        f"blocks.{layer_id}.att.value.weight",
        f"blocks.{layer_id}.att.receptance.weight",
        f"blocks.{layer_id}.att.output.weight",
        f"blocks.{layer_id}.ffn.key.weight",
        f"blocks.{layer_id}.ffn.value.weight",
    ]
    score = 0.0
    for key in keys:
        tensor = state_dict[key].float()
        score += float(tensor.norm() / math.sqrt(tensor.numel()))
    return score


def build_plan(info: LayerInfo, state_dict: OrderedDict, target_layers: int, strategy: str) -> ExpansionPlan:
    insertion_count = target_layers - info.n_layer
    if insertion_count <= 0:
        raise ValueError(f"target_layers must be larger than original layer count ({info.n_layer}).")

    eligible_after_layers = info.layer_ids[1:-1]
    if insertion_count > len(eligible_after_layers):
        raise ValueError(
            f"Need {insertion_count} insertions, but only {len(eligible_after_layers)} safe interior positions are available."
        )

    scores: dict[int, float] = {layer_id: 0.0 for layer_id in eligible_after_layers}

    if strategy in {"uniform_interp", "uniform_copy", "hybrid_alt"}:
        positions = _uniform_positions(len(eligible_after_layers), insertion_count)
        inserted_after_layers = [eligible_after_layers[pos] for pos in positions]
    elif strategy == "tail_interp":
        inserted_after_layers = eligible_after_layers[-insertion_count:]
    elif strategy == "tail_copy":
        inserted_after_layers = eligible_after_layers[-insertion_count:]
    elif strategy == "importance_copy":
        for layer_id in eligible_after_layers:
            scores[layer_id] = _importance_score(state_dict, layer_id)
        ranked = sorted(eligible_after_layers, key=lambda layer_id: (-scores[layer_id], layer_id))
        inserted_after_layers = sorted(ranked[:insertion_count])
    elif strategy == "boundary_delta_copy":
        signatures = {layer_id: _layer_signature(state_dict, layer_id) for layer_id in info.layer_ids}
        for layer_id in eligible_after_layers:
            scores[layer_id] = float(torch.norm(signatures[layer_id] - signatures[layer_id + 1], p=2))
        ranked = sorted(eligible_after_layers, key=lambda layer_id: (-scores[layer_id], layer_id))
        inserted_after_layers = sorted(ranked[:insertion_count])
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    insertion_ops: dict[int, str] = {}
    notes = ""
    if strategy == "uniform_interp":
        insertion_ops = {layer_id: "interp" for layer_id in inserted_after_layers}
        notes = "Uniformly insert interpolated layers between interior blocks."
    elif strategy == "uniform_copy":
        insertion_ops = {layer_id: "copy" for layer_id in inserted_after_layers}
        notes = "Uniformly insert copied layers after interior blocks."
    elif strategy == "hybrid_alt":
        for idx, layer_id in enumerate(inserted_after_layers):
            insertion_ops[layer_id] = "interp" if idx % 2 == 0 else "copy"
        notes = "Uniform interior insertions, alternating interpolation and copy."
    elif strategy == "tail_interp":
        insertion_ops = {layer_id: "interp" for layer_id in inserted_after_layers}
        notes = "Bias new interpolated layers toward the later part of the network."
    elif strategy == "tail_copy":
        insertion_ops = {layer_id: "copy" for layer_id in inserted_after_layers}
        notes = "Bias copied insertions toward the later part of the network."
    elif strategy == "importance_copy":
        insertion_ops = {layer_id: "copy" for layer_id in inserted_after_layers}
        notes = "Duplicate layers with the highest data-free importance scores."
    elif strategy == "boundary_delta_copy":
        insertion_ops = {layer_id: "copy" for layer_id in inserted_after_layers}
        notes = "Duplicate layers after the strongest neighbor-to-neighbor weight transitions."

    return ExpansionPlan(
        strategy=strategy,
        target_layers=target_layers,
        insertion_count=insertion_count,
        inserted_after_layers=inserted_after_layers,
        insertion_ops=insertion_ops,
        scores=scores,
        notes=notes,
    )


def _classify_keys(state_dict: OrderedDict, n_layer: int) -> tuple[list[dict[str, str]], list[str]]:
    keys_by_layer = [{} for _ in range(n_layer)]
    non_block_keys: list[str] = []
    pattern = re.compile(r"blocks\.(\d+)\.(.*)")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            layer_id = int(match.group(1))
            if layer_id < n_layer:
                keys_by_layer[layer_id][match.group(2)] = key
        else:
            non_block_keys.append(key)
    return keys_by_layer, non_block_keys


def _copy_layer(
    original_state_dict: OrderedDict,
    keys_by_layer: list[dict[str, str]],
    source_layer: int,
    new_layer: int,
    new_state_dict: OrderedDict,
) -> None:
    for remaining_key, original_key in keys_by_layer[source_layer].items():
        if new_layer > 0 and remaining_key.startswith("ln0."):
            continue
        new_key = f"blocks.{new_layer}.{remaining_key}"
        new_state_dict[new_key] = original_state_dict[original_key].clone()


def _interp_layer(
    original_state_dict: OrderedDict,
    keys_by_layer: list[dict[str, str]],
    left_layer: int,
    right_layer: int,
    new_layer: int,
    alpha: float,
    new_state_dict: OrderedDict,
) -> None:
    for remaining_key, original_key_left in keys_by_layer[left_layer].items():
        if remaining_key.startswith("ln0."):
            continue

        new_key = f"blocks.{new_layer}.{remaining_key}"
        w_left = original_state_dict[original_key_left]
        dtype = w_left.dtype

        right_key = keys_by_layer[right_layer].get(remaining_key)
        if right_key is None:
            w_right = w_left
        else:
            w_right = original_state_dict[right_key]
            if w_right.shape != w_left.shape:
                w_right = w_left

        if not torch.is_floating_point(w_left):
            new_state_dict[new_key] = w_left.clone()
            continue

        merged = alpha * w_left.float() + (1.0 - alpha) * w_right.float()
        new_state_dict[new_key] = merged.to(dtype=dtype)


def expand_depth_only(state_dict: OrderedDict, info: LayerInfo, plan: ExpansionPlan, alpha: float) -> OrderedDict:
    keys_by_layer, non_block_keys = _classify_keys(state_dict, info.n_layer)
    new_state_dict = OrderedDict()

    for key in non_block_keys:
        new_state_dict[key] = state_dict[key].clone()

    new_layer_idx = 0
    for layer_id in info.layer_ids:
        _copy_layer(state_dict, keys_by_layer, layer_id, new_layer_idx, new_state_dict)
        new_layer_idx += 1

        op = plan.insertion_ops.get(layer_id)
        if op is None:
            continue
        if op == "copy":
            _copy_layer(state_dict, keys_by_layer, layer_id, new_layer_idx, new_state_dict)
        elif op == "interp":
            _interp_layer(state_dict, keys_by_layer, layer_id, layer_id + 1, new_layer_idx, alpha, new_state_dict)
        else:
            raise ValueError(f"Unsupported op: {op}")
        new_layer_idx += 1

    if new_layer_idx != plan.target_layers:
        raise RuntimeError(f"Expected {plan.target_layers} layers after expansion, got {new_layer_idx}.")

    return new_state_dict


def _update_nested_metadata(obj: dict[str, Any], new_n_layer: int) -> None:
    if "args" in obj:
        args_obj = obj["args"]
        if hasattr(args_obj, "n_layer"):
            args_obj.n_layer = new_n_layer
    if "config" in obj and isinstance(obj["config"], dict) and "n_layer" in obj["config"]:
        obj["config"]["n_layer"] = new_n_layer


def save_checkpoint(raw_obj: dict[str, Any], wrapped: bool, new_state_dict: OrderedDict, output_path: str | Path, new_n_layer: int) -> None:
    output_path = Path(output_path)
    if wrapped:
        raw_obj["model"] = new_state_dict
        _update_nested_metadata(raw_obj, new_n_layer)
        save_obj = raw_obj
    else:
        save_obj = new_state_dict
    torch.save(save_obj, str(output_path))


def estimated_params(info: LayerInfo, target_layers: int) -> int:
    avg_per_layer = sum(info.layer_param_counts.values()) / len(info.layer_param_counts)
    return int(round(info.non_block_params + target_layers * avg_per_layer))


def main() -> None:
    args = parse_args()
    raw_obj, state_dict, wrapped = load_checkpoint(args.input_model)
    info = inspect_checkpoint(state_dict)

    print(f"Input model      : {args.input_model}")
    print(f"Layers           : {info.n_layer} ({info.layer_ids[0]}..{info.layer_ids[-1]})")
    print(f"Embedding dim    : {info.n_embd}")
    print(f"Vocabulary size  : {info.vocab_size}")
    print(f"Head size / count: {info.head_size} / {info.n_head}")
    print(f"Total params     : {info.total_params:,}")
    print(f"Non-block params : {info.non_block_params:,}")

    if args.inspect_only:
        return

    plan = build_plan(info, state_dict, args.target_layers, args.strategy)
    print(f"Strategy         : {plan.strategy}")
    print(f"Strategy notes   : {plan.notes}")
    print(f"Target layers    : {plan.target_layers}")
    print(f"Insertions       : {plan.insertion_count}")
    print(f"Inserted after   : {plan.inserted_after_layers}")
    print(f"Estimated params : {estimated_params(info, plan.target_layers):,}")

    metadata = {
        "input_model": str(args.input_model),
        "output_model": str(args.output_model),
        "strategy": plan.strategy,
        "strategy_notes": plan.notes,
        "original_layers": info.n_layer,
        "target_layers": plan.target_layers,
        "insertions": plan.insertion_count,
        "inserted_after_layers": plan.inserted_after_layers,
        "insertion_ops": plan.insertion_ops,
        "scores": plan.scores,
        "alpha": args.alpha,
        "n_embd": info.n_embd,
        "vocab_size": info.vocab_size,
        "n_head": info.n_head,
        "head_size": info.head_size,
        "estimated_params": estimated_params(info, plan.target_layers),
    }

    if args.plan_only:
        metadata_path = Path(args.metadata_out) if args.metadata_out else Path(args.output_model).with_suffix(".json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved metadata   : {metadata_path}")
        return

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_state_dict = expand_depth_only(state_dict, info, plan, alpha=args.alpha)
    save_checkpoint(raw_obj, wrapped, new_state_dict, output_path, plan.target_layers)
    print(f"Saved checkpoint : {output_path}")

    metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata   : {metadata_path}")


if __name__ == "__main__":
    main()
