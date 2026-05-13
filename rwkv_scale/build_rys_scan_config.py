from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSON config for RYS-style depth expansion scans.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--original-layers", type=int, default=32)
    parser.add_argument("--target-layers", type=int, default=56)
    parser.add_argument("--starts", default="0,4,8,12,16,20,24,28")
    parser.add_argument("--block-sizes", default="3,4,6,8,12,24")
    parser.add_argument("--name-prefix", default="rwkv7-g1f-12b-expand-56l")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def main() -> None:
    args = parse_args()
    starts = parse_int_list(args.starts)
    block_sizes = parse_int_list(args.block_sizes)
    insertion_count = args.target_layers - args.original_layers
    if insertion_count <= 0:
        raise ValueError("target-layers must be larger than original-layers for RYS expansion.")

    config: list[dict] = []
    for start in starts:
        for block_size in block_sizes:
            if start < 0 or start >= args.original_layers:
                continue
            if start + block_size > args.original_layers:
                continue
            if insertion_count % block_size != 0:
                continue
            config.append(
                {
                    "name": f"{args.name_prefix}-rys-s{start}-b{block_size}",
                    "strategy": "rys_repeat",
                    "target_layers": args.target_layers,
                    "alpha": 0.5,
                    "rys_start_layer": start,
                    "rys_block_size": block_size,
                    "rys_repeat_count": insertion_count // block_size,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
