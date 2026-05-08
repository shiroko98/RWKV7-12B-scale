from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_EXPERIMENTS = [
    {
        "name": "rwkv7-g1f-12b-56l-uniform",
        "strategy": "uniform",
        "target_layers": 56,
        "preserve_first": 1,
        "preserve_last": 1,
        "force_keep": [],
    },
    {
        "name": "rwkv7-g1f-12b-56l-last6",
        "strategy": "last_layer_preserving",
        "target_layers": 56,
        "preserve_first": 1,
        "preserve_last": 6,
        "force_keep": [],
    },
    {
        "name": "rwkv7-g1f-12b-56l-importance",
        "strategy": "importance",
        "target_layers": 56,
        "preserve_first": 1,
        "preserve_last": 4,
        "force_keep": [],
    },
    {
        "name": "rwkv7-g1f-12b-56l-neighbor-delta",
        "strategy": "neighbor_delta",
        "target_layers": 56,
        "preserve_first": 1,
        "preserve_last": 4,
        "force_keep": [],
    },
    {
        "name": "rwkv7-g1f-12b-54l-uniform",
        "strategy": "uniform",
        "target_layers": 54,
        "preserve_first": 1,
        "preserve_last": 1,
        "force_keep": [],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a batch of pruned RWKV checkpoints.")
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pruned"))
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run prune_layers.py")
    parser.add_argument("--manifest-out", default=str(ROOT / "outputs" / "pruned" / "manifest.json"))
    parser.add_argument("--config", help="Optional JSON file overriding the default experiment matrix.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_experiments(config_path: str | None) -> list[dict]:
    if not config_path:
        return DEFAULT_EXPERIMENTS
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(args.config)
    manifest: list[dict] = []

    for exp in experiments:
        output_model = output_dir / f"{exp['name']}.pth"
        metadata_out = output_dir / f"{exp['name']}.json"
        cmd = [
            args.python,
            str(ROOT / "prune_layers.py"),
            "--input-model",
            args.input_model,
            "--output-model",
            str(output_model),
            "--strategy",
            exp["strategy"],
            "--target-layers",
            str(exp["target_layers"]),
            "--preserve-first",
            str(exp.get("preserve_first", 1)),
            "--preserve-last",
            str(exp.get("preserve_last", 1)),
            "--metadata-out",
            str(metadata_out),
        ]
        force_keep = exp.get("force_keep", [])
        if force_keep:
            cmd.extend(["--force-keep", *[str(x) for x in force_keep]])

        record = {
            **exp,
            "output_model": str(output_model),
            "metadata_out": str(metadata_out),
            "command": cmd,
        }
        manifest.append(record)

        print(f"\n=== {exp['name']} ===")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

