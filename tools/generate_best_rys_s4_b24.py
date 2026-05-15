from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the best full-scan RYS model: s4-b24 -> 56 layers.")
    parser.add_argument("--input-model", required=True)
    parser.add_argument(
        "--output-model",
        default=str(ROOT / "outputs" / "expanded_rys_best" / "rwkv7-g1f-12b-expand-56l-rys-s4-b24.pth"),
    )
    parser.add_argument(
        "--metadata-out",
        default=str(ROOT / "outputs" / "expanded_rys_best" / "rwkv7-g1f-12b-expand-56l-rys-s4-b24.json"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_model = Path(args.output_model)
    metadata_out = Path(args.metadata_out)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(ROOT / "rwkv_scale" / "depth_expand.py"),
        "--input-model",
        args.input_model,
        "--output-model",
        str(output_model),
        "--strategy",
        "rys_repeat",
        "--target-layers",
        "56",
        "--alpha",
        "0.5",
        "--rys-start-layer",
        "4",
        "--rys-block-size",
        "24",
        "--rys-repeat-count",
        "1",
        "--metadata-out",
        str(metadata_out),
    ]
    if args.plan_only:
        cmd.append("--plan-only")

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
