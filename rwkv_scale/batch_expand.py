from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_EXPERIMENTS = [
    {
        "name": "rwkv7-g1f-12b-expand-56l-uniform-copy",
        "strategy": "uniform_copy",
        "target_layers": 56,
        "alpha": 0.5,
    },
    {
        "name": "rwkv7-g1f-12b-expand-56l-tail-copy",
        "strategy": "tail_copy",
        "target_layers": 56,
        "alpha": 0.5,
    },
    {
        "name": "rwkv7-g1f-12b-expand-56l-importance-copy",
        "strategy": "importance_copy",
        "target_layers": 56,
        "alpha": 0.5,
    },
    {
        "name": "rwkv7-g1f-12b-expand-56l-boundary-delta-copy",
        "strategy": "boundary_delta_copy",
        "target_layers": 56,
        "alpha": 0.5,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a batch of copy-focused depth-expanded RWKV checkpoints.")
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "expanded_copy_focus"))
    parser.add_argument(
        "--manifest-out",
        default=str(ROOT / "outputs" / "expanded_copy_focus" / "manifest_copy_focus.json"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_experiments(config_path: str | None) -> list[dict]:
    if not config_path:
        return DEFAULT_EXPERIMENTS
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    experiments = load_experiments(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for exp in experiments:
        output_model = output_dir / f"{exp['name']}.pth"
        metadata_out = output_dir / f"{exp['name']}.json"
        cmd = [
            args.python,
            str(THIS_DIR / "depth_expand.py"),
            "--input-model",
            args.input_model,
            "--output-model",
            str(output_model),
            "--strategy",
            exp["strategy"],
            "--target-layers",
            str(exp["target_layers"]),
            "--alpha",
            str(exp.get("alpha", 0.5)),
            "--metadata-out",
            str(metadata_out),
        ]
        if args.plan_only:
            cmd.append("--plan-only")

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
