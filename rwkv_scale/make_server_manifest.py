from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite an expansion manifest for Linux server paths.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser.parse_args()


def rewrite_output_path(original_path: str, server_root: PurePosixPath) -> str:
    windows_path = PureWindowsPath(original_path)
    parts = list(windows_path.parts)
    try:
        outputs_idx = next(i for i, part in enumerate(parts) if part.lower() == "outputs")
        relative_parts = parts[outputs_idx:]
    except StopIteration:
        relative_parts = [windows_path.name]
    return str(server_root.joinpath(*relative_parts))


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    server_root = PurePosixPath(args.server_root)
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))

    rewritten = []
    for item in manifest:
        copied = dict(item)
        copied["output_model"] = rewrite_output_path(item["output_model"], server_root)
        copied["metadata_out"] = rewrite_output_path(item["metadata_out"], server_root)
        copied.pop("command", None)
        rewritten.append(copied)

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_manifest)


if __name__ == "__main__":
    main()
