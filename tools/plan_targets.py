from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_prune.checkpoint import inspect_checkpoint, load_checkpoint
from rwkv_prune.targets import estimate_keep_layer_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate which keep-layer count is closest to a target parameter budget.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-params", type=int, default=12_000_000_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preferred-divisors", type=int, nargs="*", default=[8, 4, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, state_dict, _ = load_checkpoint(args.model_path)
    info = inspect_checkpoint(state_dict)
    estimates = estimate_keep_layer_options(
        info,
        args.target_params,
        preferred_divisors=tuple(args.preferred_divisors),
    )
    print(f"Model layers: {info.n_layer}")
    print(f"Target params: {args.target_params:,}")
    for item in estimates[: args.top_k]:
        print(
            f"keep={item.keep_layers:02d}  "
            f"estimated_params={item.estimated_params:,}  "
            f"distance={item.distance_to_target:,}  "
            f"parallel_divisor={item.preferred_divisor or 'none'}"
        )


if __name__ == "__main__":
    main()
