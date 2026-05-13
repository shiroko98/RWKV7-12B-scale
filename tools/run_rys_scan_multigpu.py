from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from scan_expand_eval_cleanup import atomic_write_json
from scan_expand_eval_cleanup import build_markdown_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RYS scan across multiple GPUs by sharding the config.")
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--config", required=True, help="JSON config list generated for RYS scan experiments.")
    parser.add_argument("--work-dir", required=True, help="Temporary directory to place worker configs and checkpoints.")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--markdown-out", help="Optional merged Markdown summary output path.")
    parser.add_argument("--log-dir", help="Optional log directory. Worker logs are written into per-GPU subdirs.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7")
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
    parser.add_argument("--keep-per-model-json", action="store_true")
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--keep-metadata", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the shard plan and commands without executing them.")
    return parser.parse_args()


def parse_gpu_ids(text: str) -> list[str]:
    gpu_ids = [item.strip() for item in text.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id.")
    return gpu_ids


def sanitize_label(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text)


def shard_experiments(experiments: list[dict], gpu_ids: list[str]) -> dict[str, list[dict]]:
    shards = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, exp in enumerate(experiments):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        shards[gpu_id].append(exp)
    return shards


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_worker_command(
    args: argparse.Namespace,
    shard_config: Path,
    shard_work_dir: Path,
    shard_summary: Path,
    shard_markdown: Path | None,
    shard_log_dir: Path | None,
) -> list[str]:
    cmd = [
        args.python,
        str(ROOT / "tools" / "run_rys_scan.py"),
        "--input-model",
        args.input_model,
        "--config",
        str(shard_config),
        "--work-dir",
        str(shard_work_dir),
        "--tokenizer-path",
        args.tokenizer_path,
        "--summary-out",
        str(shard_summary),
        "--device",
        "cuda" if args.device.startswith("cuda") else args.device,
        "--dtype",
        args.dtype,
        "--task",
        args.task,
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
    ]
    if shard_markdown:
        cmd.extend(["--markdown-out", str(shard_markdown)])
    if shard_log_dir:
        cmd.extend(["--log-dir", str(shard_log_dir)])
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])
    if args.keep_per_model_json:
        cmd.append("--keep-per-model-json")
    if args.keep_models:
        cmd.append("--keep-models")
    if args.keep_metadata:
        cmd.append("--keep-metadata")
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    return cmd


def load_shard_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    experiments = json.loads(Path(args.config).read_text(encoding="utf-8"))
    shards = shard_experiments(experiments, gpu_ids)
    shard_root = work_dir / "_multigpu"
    shard_root.mkdir(parents=True, exist_ok=True)

    order_map = {exp["name"]: idx for idx, exp in enumerate(experiments)}
    processes: list[tuple[str, subprocess.Popen, Path]] = []
    shard_summary_paths: list[Path] = []

    for gpu_id in gpu_ids:
        shard = shards[gpu_id]
        if not shard:
            continue

        gpu_label = sanitize_label(gpu_id)
        shard_config = shard_root / f"config_gpu{gpu_label}.json"
        shard_summary = shard_root / f"summary_gpu{gpu_label}.json"
        shard_markdown = shard_root / f"summary_gpu{gpu_label}.md" if args.markdown_out else None
        shard_work_dir = work_dir / f"gpu{gpu_label}"
        shard_log_dir = (Path(args.log_dir) / f"gpu{gpu_label}") if args.log_dir else None

        write_json(shard_config, shard)
        cmd = build_worker_command(args, shard_config, shard_work_dir, shard_summary, shard_markdown, shard_log_dir)
        shard_summary_paths.append(shard_summary)

        print(f"\n=== gpu {gpu_id} | {len(shard)} experiments ===")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        env = os.environ.copy()
        if args.device.startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        proc = subprocess.Popen(cmd, env=env)
        processes.append((gpu_id, proc, shard_summary))

    if args.dry_run:
        return

    failed_workers: list[str] = []
    for gpu_id, proc, _shard_summary in processes:
        return_code = proc.wait()
        if return_code != 0:
            failed_workers.append(f"{gpu_id}:{return_code}")

    merged: list[dict] = []
    for shard_summary in shard_summary_paths:
        merged.extend(load_shard_summary(shard_summary))

    merged.sort(key=lambda item: order_map.get(item.get("name", ""), 10**9))
    summary_path = Path(args.summary_out)
    atomic_write_json(summary_path, merged)

    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(build_markdown_summary(merged), encoding="utf-8")

    print(f"\nSaved merged summary: {summary_path}")
    if args.markdown_out:
        print(f"Saved merged markdown: {args.markdown_out}")

    if failed_workers:
        raise SystemExit(f"Workers failed: {', '.join(failed_workers)}")


if __name__ == "__main__":
    main()
