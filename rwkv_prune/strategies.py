from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from .checkpoint import LayerInfo


@dataclass
class StrategyResult:
    strategy: str
    drop_layers: list[int]
    kept_layers: list[int]
    scores: dict[int, float]
    notes: str


def _candidate_layers(layer_ids: list[int], preserve_first: int, preserve_last: int, force_keep: set[int]) -> list[int]:
    protected = set(layer_ids[:preserve_first]) | set(layer_ids[-preserve_last:]) | set(force_keep)
    return [layer_id for layer_id in layer_ids if layer_id not in protected]


def _uniform_positions(count: int, drop_count: int) -> list[int]:
    if drop_count <= 0:
        return []
    if drop_count >= count:
        return list(range(count))
    positions = []
    for idx in range(drop_count):
        pos = round((idx + 1) * (count + 1) / (drop_count + 1)) - 1
        pos = min(max(pos, 0), count - 1)
        positions.append(pos)
    deduped = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            deduped.append(pos)
    cur = 0
    while len(deduped) < drop_count:
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


def _layer_signature(state_dict: dict[str, torch.Tensor], layer_id: int) -> torch.Tensor:
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


def _importance_score(state_dict: dict[str, torch.Tensor], layer_id: int) -> float:
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


def choose_layers(
    strategy: str,
    state_dict: dict[str, torch.Tensor],
    info: LayerInfo,
    drop_count: int,
    preserve_first: int = 1,
    preserve_last: int = 1,
    force_keep: Iterable[int] | None = None,
) -> StrategyResult:
    if drop_count <= 0:
        return StrategyResult(strategy=strategy, drop_layers=[], kept_layers=info.layer_ids[:], scores={}, notes="No layers dropped.")

    force_keep_set = set(force_keep or [])
    candidate_layers = _candidate_layers(info.layer_ids, preserve_first, preserve_last, force_keep_set)
    if drop_count > len(candidate_layers):
        raise ValueError(
            f"Requested drop_count={drop_count}, but only {len(candidate_layers)} layers are available after preservation."
        )

    scores: dict[int, float] = {layer_id: 0.0 for layer_id in info.layer_ids}

    if strategy == "uniform":
        positions = _uniform_positions(len(candidate_layers), drop_count)
        drop_layers = [candidate_layers[pos] for pos in positions]
        notes = "Uniformly drop layers from the remaining candidate span."
    elif strategy == "last_layer_preserving":
        preserve_last = max(preserve_last, 6)
        preserve_first = max(preserve_first, 1)
        candidate_layers = _candidate_layers(info.layer_ids, preserve_first, preserve_last, force_keep_set)
        if drop_count > len(candidate_layers):
            raise ValueError(
                f"Requested drop_count={drop_count}, but only {len(candidate_layers)} layers remain after preserving tail layers."
            )
        positions = _uniform_positions(len(candidate_layers), drop_count)
        drop_layers = [candidate_layers[pos] for pos in positions]
        notes = "Preserve a wider tail segment, then uniformly prune the rest."
    elif strategy == "importance":
        for layer_id in candidate_layers:
            scores[layer_id] = _importance_score(state_dict, layer_id)
        ranked = sorted(candidate_layers, key=lambda layer_id: (scores[layer_id], layer_id))
        drop_layers = ranked[:drop_count]
        notes = "Data-free importance proxy using normalized weight norms."
    elif strategy == "neighbor_delta":
        signatures = {layer_id: _layer_signature(state_dict, layer_id) for layer_id in info.layer_ids}
        for layer_id in candidate_layers:
            prev_id = layer_id - 1
            next_id = layer_id + 1
            deltas = []
            if prev_id in signatures:
                deltas.append(float(torch.norm(signatures[layer_id] - signatures[prev_id], p=2)))
            if next_id in signatures:
                deltas.append(float(torch.norm(signatures[layer_id] - signatures[next_id], p=2)))
            scores[layer_id] = sum(deltas) / len(deltas) if deltas else float("inf")
        ranked = sorted(candidate_layers, key=lambda layer_id: (scores[layer_id], layer_id))
        drop_layers = ranked[:drop_count]
        notes = "Drop layers whose sampled weight fingerprints are most similar to their neighbors."
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    drop_layers = sorted(drop_layers)
    kept_layers = [layer_id for layer_id in info.layer_ids if layer_id not in set(drop_layers)]
    return StrategyResult(
        strategy=strategy,
        drop_layers=drop_layers,
        kept_layers=kept_layers,
        scores=scores,
        notes=notes,
    )
