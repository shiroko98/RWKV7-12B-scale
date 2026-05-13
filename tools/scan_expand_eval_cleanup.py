from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from pathlib import PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, evaluate, summarize, and optionally delete expanded RWKV checkpoints one by one."
    )
    parser.add_argument("--manifest", required=True, help="Expansion manifest from rwkv_scale/batch_expand.py")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--task", default="both", choices=["ppl", "generate", "both"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "lambada", "textfile"])
    parser.add_argument("--dataset-path")
    parser.add_argument("--max-docs", type=int, default=128)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--prompt", default="User: 请介绍一下北京。\n\nAssistant: ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--probes", default="", help="Comma-separated probes to run: math,eq,json")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--markdown-out", help="Optional Markdown summary output path.")
    parser.add_argument("--log-dir", help="Directory for per-model stdout/stderr logs.")
    parser.add_argument(
        "--keep-per-model-json",
        action="store_true",
        help="Keep each per-model eval JSON file. By default only the aggregate summary is preserved.",
    )
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--keep-metadata", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _basename_from_any_path(path_str: str) -> str:
    if "\\" in path_str or (len(path_str) >= 2 and path_str[1] == ":"):
        return PureWindowsPath(path_str).name
    return Path(path_str).name


def resolve_model_path(manifest_path: Path, model_path: str) -> Path:
    candidate = Path(model_path)
    if candidate.exists():
        return candidate
    fallback = manifest_path.parent / _basename_from_any_path(model_path)
    if fallback.exists():
        return fallback
    return candidate


def resolve_metadata_path(manifest_path: Path, metadata_path: str | None) -> Path | None:
    if not metadata_path:
        return None
    candidate = Path(metadata_path)
    if candidate.exists():
        return candidate
    fallback = manifest_path.parent / _basename_from_any_path(metadata_path)
    if fallback.exists():
        return fallback
    return candidate


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def build_markdown_summary(summary: list[dict]) -> str:
    lines: list[str] = ["# RYS Scan Summary", ""]
    if not summary:
        lines.append("No results yet.")
        return "\n".join(lines) + "\n"

    completed = [item for item in summary if "error" not in item]
    failed = [item for item in summary if "error" in item]
    lines.append(f"- Total records: {len(summary)}")
    lines.append(f"- Completed: {len(completed)}")
    lines.append(f"- Failed: {len(failed)}")
    lines.append("")

    if completed:
        ranked = sorted(
            completed,
            key=lambda item: float(item.get("ppl", {}).get("ppl", float("inf"))),
        )
        lines.append("## By PPL")
        lines.append("")
        lines.append("| Rank | Name | Block | Repeat | PPL | Math | EQ | JSON | Loop | Unknown |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for idx, item in enumerate(ranked, start=1):
            ppl = item.get("ppl", {}).get("ppl", "")
            probes = item.get("probes", {})
            math_score = probes.get("math", {}).get("mean_score", "")
            eq_score = probes.get("eq", {}).get("mean_score", "")
            json_score = probes.get("json", {}).get("mean_score", "")
            loop = item.get("generation_metrics", {}).get("max_loop_repeats", "")
            unknown = item.get("generation_metrics", {}).get("unknown_token_count", "")
            rys_blocks = item.get("rys_blocks") or []
            if rys_blocks:
                block_desc = ",".join(f"{block.get('start')}-{block.get('start', 0) + block.get('size', 0) - 1}" for block in rys_blocks)
                repeat_desc = ",".join(str(block.get("repeat", "")) for block in rys_blocks)
            else:
                start = item.get("rys_start_layer")
                size = item.get("rys_block_size")
                repeat = item.get("rys_repeat_count", "")
                block_desc = f"{start}-{start + size - 1}" if start is not None and size is not None else ""
                repeat_desc = str(repeat) if repeat != "" else ""
            lines.append(
                f"| {idx} | {item.get('name', '')} | {block_desc} | {repeat_desc} | {ppl} | {math_score} | {eq_score} | {json_score} | {loop} | {unknown} |"
            )
        lines.append("")

    if failed:
        lines.append("## Failures")
        lines.append("")
        for item in failed:
            lines.append(f"- `{item.get('name', '')}`: {item.get('error', '')}")
        lines.append("")

    return "\n".join(lines) + "\n"


def atomic_write_markdown(path: Path, summary: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(build_markdown_summary(summary), encoding="utf-8")
    tmp_path.replace(path)


def run_eval(args: argparse.Namespace, model_name: str, model_path: Path) -> tuple[int, Path]:
    summary_path = Path(args.summary_out)
    output_dir = summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".per_model_eval_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    json_out = temp_dir / f"{model_name}.eval.json"
    cmd = [
        args.python,
        str(ROOT / "evals" / "rwkv_eval.py"),
        "--model-path",
        str(model_path),
        "--tokenizer-path",
        args.tokenizer_path,
        "--task",
        args.task,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--dataset",
        args.dataset,
        "--max-docs",
        str(args.max_docs),
        "--token-budget",
        str(args.token_budget),
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--probes",
        args.probes,
        "--json-out",
        str(json_out),
    ]
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])

    print(f"\n=== eval {model_name} ===")
    print(" ".join(cmd))
    completed = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model_name}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("COMMAND:\n")
            log_handle.write(" ".join(cmd) + "\n\n")
            log_handle.flush()
            completed = subprocess.run(cmd, check=False, stdout=log_handle, stderr=subprocess.STDOUT)
    else:
        completed = subprocess.run(cmd, check=False)
    return completed.returncode, json_out


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary: list[dict] = []

    for item in manifest:
        model_name = item["name"]
        model_path = resolve_model_path(manifest_path, item["output_model"])
        metadata_path = resolve_metadata_path(manifest_path, item.get("metadata_out"))

        record: dict[str, object] = {
            "name": model_name,
            "strategy": item.get("strategy"),
            "target_layers": item.get("target_layers"),
            "output_model": str(model_path),
            "metadata_out": str(metadata_path) if metadata_path else None,
            "rys_start_layer": item.get("rys_start_layer"),
            "rys_block_size": item.get("rys_block_size"),
            "rys_repeat_count": item.get("rys_repeat_count"),
            "rys_blocks": item.get("rys_blocks"),
            "block_label": "",
        }
        if record["rys_blocks"]:
            blocks = record["rys_blocks"]
            record["block_label"] = ",".join(
                f"{block.get('start')}-{block.get('start', 0) + block.get('size', 0) - 1}x{block.get('repeat', 1)}"
                for block in blocks
            )
        elif record["rys_start_layer"] is not None and record["rys_block_size"] is not None:
            start = int(record["rys_start_layer"])
            size = int(record["rys_block_size"])
            repeat = record["rys_repeat_count"]
            record["block_label"] = f"{start}-{start + size - 1}x{repeat}"

        exit_code, json_out = run_eval(args, model_name, model_path)
        record["eval_json"] = str(json_out)

        if exit_code != 0:
            record["error"] = f"evaluation failed with exit code {exit_code}"
            summary.append(record)
            atomic_write_json(Path(args.summary_out), summary)
            if args.markdown_out:
                atomic_write_markdown(Path(args.markdown_out), summary)
            if not args.keep_models and model_path.exists():
                model_path.unlink()
            if not args.keep_metadata and metadata_path and metadata_path.exists():
                metadata_path.unlink()
            if not args.keep_per_model_json and json_out.exists():
                json_out.unlink()
            if args.stop_on_error:
                raise SystemExit(exit_code)
            continue

        result = json.loads(json_out.read_text(encoding="utf-8"))
        record["ppl"] = result.get("ppl", {})
        record["generation_metrics"] = result.get("generation", {}).get("metrics", {})
        record["probes"] = result.get("probes", {})
        summary.append(record)
        atomic_write_json(Path(args.summary_out), summary)
        if args.markdown_out:
            atomic_write_markdown(Path(args.markdown_out), summary)

        if not args.keep_models and model_path.exists():
            model_path.unlink()
        if not args.keep_metadata and metadata_path and metadata_path.exists():
            metadata_path.unlink()
        if not args.keep_per_model_json and json_out.exists():
            json_out.unlink()

    print(f"\nSaved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
