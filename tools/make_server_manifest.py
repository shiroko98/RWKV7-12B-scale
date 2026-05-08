from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PureWindowsPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a pruning manifest for a Linux server path.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--server-root", required=True, help="Workspace root on the target server.")
    parser.add_argument("--output-manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    server_root = Path(args.server_root)
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))

    rewritten = []
    for item in manifest:
        copied = dict(item)
        copied["output_model"] = str(server_root / "outputs" / "pruned" / PureWindowsPath(item["output_model"]).name)
        copied["metadata_out"] = str(server_root / "outputs" / "pruned" / PureWindowsPath(item["metadata_out"]).name)
        copied.pop("command", None)
        rewritten.append(copied)

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_manifest)


if __name__ == "__main__":
    main()
