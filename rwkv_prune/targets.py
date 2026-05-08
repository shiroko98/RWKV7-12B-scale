from __future__ import annotations

from dataclasses import dataclass

from .checkpoint import LayerInfo


@dataclass
class TargetEstimate:
    keep_layers: int
    estimated_params: int
    distance_to_target: int
    preferred_divisor: int


def estimate_keep_layer_options(
    info: LayerInfo,
    target_params: int,
    min_keep: int = 1,
    preferred_divisors: tuple[int, ...] = (8, 4, 2),
) -> list[TargetEstimate]:
    estimates: list[TargetEstimate] = []
    for keep_layers in range(min_keep, info.n_layer + 1):
        kept_layer_ids = info.layer_ids[:keep_layers]
        estimated_params = info.non_block_params + sum(info.layer_param_counts[layer_id] for layer_id in kept_layer_ids)
        matched_divisor = 0
        for divisor in preferred_divisors:
            if keep_layers % divisor == 0:
                matched_divisor = divisor
                break
        estimates.append(
            TargetEstimate(
                keep_layers=keep_layers,
                estimated_params=estimated_params,
                distance_to_target=abs(estimated_params - target_params),
                preferred_divisor=matched_divisor,
            )
        )
    estimates.sort(
        key=lambda item: (
            item.preferred_divisor == 0,
            -item.preferred_divisor,
            item.distance_to_target,
            -item.keep_layers,
        )
    )
    return estimates
