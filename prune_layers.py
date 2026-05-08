from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_prune.checkpoint import (
    build_layer_map,
    estimate_parameter_count,
    inspect_checkpoint,
    load_checkpoint,
    prune_state_dict,
    save_pruned_checkpoint,
)
from rwkv_prune.strategies import choose_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune full RWKV blocks from a checkpoint.")
    parser.add_argument("--input-model", required=True, help="Path to the source .pth checkpoint.")
    parser.add_argument("--output-model", required=True, help="Path to the output .pth checkpoint.")
    parser.add_argument(
        "--strategy",
        default="uniform",
        choices=["uniform", "last_layer_preserving", "importance", "neighbor_delta"],
        help="Layer dropping strategy.",
    )
    parser.add_argument("--target-layers", type=int, help="Final number of layers to keep.")
    parser.add_argument("--drop-count", type=int, help="Number of layers to remove.")
    parser.add_argument("--preserve-first", type=int, default=1, help="Always keep the first N layers.")
    parser.add_argument("--preserve-last", type=int, default=1, help="Always keep the last N layers.")
    parser.add_argument(
        "--force-keep",
        type=int,
        nargs="*",
        default=[],
        help="Specific layer ids to protect from pruning.",
    )
    parser.add_argument("--metadata-out", help="Optional JSON file for pruning metadata.")
    parser.add_argument("--inspect-only", action="store_true", help="Print checkpoint stats and exit.")
    return parser.parse_args()


def _resolve_drop_count(n_layer: int, target_layers: int | None, drop_count: int | None) -> int:
    if drop_count is not None and target_layers is not None:
        expected = n_layer - target_layers
        if expected != drop_count:
            raise ValueError(f"target_layers={target_layers} implies drop_count={expected}, but got {drop_count}.")
    if drop_count is not None:
        return drop_count
    if target_layers is not None:
        return n_layer - target_layers
    raise ValueError("Either --target-layers or --drop-count must be provided.")


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

    drop_count = _resolve_drop_count(info.n_layer, args.target_layers, args.drop_count)
    result = choose_layers(
        strategy=args.strategy,
        state_dict=state_dict,
        info=info,
        drop_count=drop_count,
        preserve_first=args.preserve_first,
        preserve_last=args.preserve_last,
        force_keep=args.force_keep,
    )
    drop_layers = set(result.drop_layers)
    layer_map = build_layer_map(info.layer_ids, drop_layers)
    new_state, removed_params, kept_params = prune_state_dict(state_dict, drop_layers, layer_map)

    print(f"Strategy         : {result.strategy}")
    print(f"Strategy notes   : {result.notes}")
    print(f"Dropped layers   : {result.drop_layers}")
    print(f"Final layers     : {len(result.kept_layers)}")
    print(f"Removed params   : {removed_params:,}")
    print(f"Remaining params : {kept_params:,}")
    print(f"Estimated bf16 GB: {kept_params * 2 / 1024 / 1024 / 1024:.2f}")

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pruned_checkpoint(raw_obj, wrapped, new_state, output_path)
    print(f"Saved checkpoint : {output_path}")

    metadata = {
        "input_model": str(args.input_model),
        "output_model": str(output_path),
        "strategy": result.strategy,
        "strategy_notes": result.notes,
        "drop_count": drop_count,
        "original_layers": info.n_layer,
        "final_layers": len(result.kept_layers),
        "dropped_layers": result.drop_layers,
        "kept_layers": result.kept_layers,
        "preserve_first": args.preserve_first,
        "preserve_last": args.preserve_last,
        "force_keep": args.force_keep,
        "removed_params": removed_params,
        "remaining_params": kept_params,
        "estimated_bf16_gb": kept_params * 2 / 1024 / 1024 / 1024,
        "estimated_params_from_kept_layers": estimate_parameter_count(info, result.kept_layers),
        "scores": result.scores,
    }
    metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata   : {metadata_path}")


if __name__ == "__main__":
    main()

