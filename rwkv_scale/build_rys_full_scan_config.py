from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict-RYS full-scan config over all legal start/block pairs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--original-layers", type=int, default=32)
    parser.add_argument("--target-layers", type=int, default=56)
    parser.add_argument("--max-block-size", type=int)
    parser.add_argument("--min-block-size", type=int, default=1)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--name-prefix", default="rwkv7-g1f-12b-expand-56l")
    parser.add_argument("--limit-half", action="store_true", help="Only scan block sizes up to original_layers // 2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    insertion_count = args.target_layers - args.original_layers
    if insertion_count <= 0:
        raise ValueError("target-layers must be larger than original-layers.")

    max_block_size = args.max_block_size or args.original_layers
    if args.limit_half:
        max_block_size = min(max_block_size, args.original_layers // 2)

    config: list[dict] = []
    for block_size in range(args.min_block_size, max_block_size + 1):
        if insertion_count % block_size != 0:
            continue
        repeat_count = insertion_count // block_size
        for start in range(0, args.original_layers - block_size + 1):
            config.append(
                {
                    "name": f"{args.name_prefix}-rys-s{start}-b{block_size}",
                    "strategy": "rys_repeat",
                    "target_layers": args.target_layers,
                    "alpha": 0.5,
                    "rys_start_layer": start,
                    "rys_block_size": block_size,
                    "rys_repeat_count": repeat_count,
                }
            )
            if args.max_configs is not None and len(config) >= args.max_configs:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"{output_path} ({len(config)} configs)")
                return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{output_path} ({len(config)} configs)")


if __name__ == "__main__":
    main()
