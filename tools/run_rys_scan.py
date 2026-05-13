from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RYS scan end-to-end: build one model, eval it, record, then delete it.")
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--config", required=True, help="JSON config list generated for RYS scan experiments.")
    parser.add_argument("--work-dir", required=True, help="Temporary directory to place generated checkpoints and metadata.")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--markdown-out", help="Optional Markdown summary output path.")
    parser.add_argument("--log-dir", help="Directory for per-model build/eval logs.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--task", default="both", choices=["ppl", "generate", "both"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "lambada", "textfile"])
    parser.add_argument("--dataset-path")
    parser.add_argument("--max-docs", type=int, default=128)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--prompt", default="User: 请介绍一下北京。\n\nAssistant: ")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--probes", default="", help="Comma-separated probes to run: math,eq,json")
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--keep-metadata", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    experiments = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest: list[dict] = []

    for exp in experiments:
        output_model = work_dir / f"{exp['name']}.pth"
        metadata_out = work_dir / f"{exp['name']}.json"
        cmd = [
            args.python,
            str(ROOT / "rwkv_scale" / "depth_expand.py"),
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
        if "rys_start_layer" in exp:
            cmd.extend(["--rys-start-layer", str(exp["rys_start_layer"])])
        if "rys_block_size" in exp:
            cmd.extend(["--rys-block-size", str(exp["rys_block_size"])])
        if "rys_repeat_count" in exp:
            cmd.extend(["--rys-repeat-count", str(exp["rys_repeat_count"])])
        if "rys_blocks" in exp:
            cmd.extend(["--rys-blocks", json.dumps(exp["rys_blocks"], ensure_ascii=False)])

        print(f"\n=== build {exp['name']} ===")
        print(" ".join(cmd))
        if args.log_dir:
            log_dir = Path(args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            build_log = log_dir / f"{exp['name']}.build.log"
            with build_log.open("w", encoding="utf-8") as log_handle:
                log_handle.write("COMMAND:\n")
                log_handle.write(" ".join(cmd) + "\n\n")
                log_handle.flush()
                completed = subprocess.run(cmd, check=False, stdout=log_handle, stderr=subprocess.STDOUT)
        else:
            completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            if args.stop_on_error:
                raise SystemExit(completed.returncode)
            manifest.append(
                {
                    **exp,
                    "output_model": str(output_model),
                    "metadata_out": str(metadata_out),
                    "error": f"build failed with exit code {completed.returncode}",
                }
            )
            Path(args.summary_out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            continue

        manifest.append(
            {
                **exp,
                "output_model": str(output_model),
                "metadata_out": str(metadata_out),
            }
        )

        temp_manifest = work_dir / "current_manifest.json"
        temp_manifest.write_text(json.dumps([manifest[-1]], indent=2, ensure_ascii=False), encoding="utf-8")

        eval_cmd = [
            args.python,
            str(ROOT / "tools" / "scan_expand_eval_cleanup.py"),
            "--manifest",
            str(temp_manifest),
            "--tokenizer-path",
            args.tokenizer_path,
            "--summary-out",
            args.summary_out,
            "--device",
            args.device,
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
        if args.markdown_out:
            eval_cmd.extend(["--markdown-out", args.markdown_out])
        if args.log_dir:
            eval_cmd.extend(["--log-dir", args.log_dir])
        if args.dataset_path:
            eval_cmd.extend(["--dataset-path", args.dataset_path])
        if args.keep_models:
            eval_cmd.append("--keep-models")
        if args.keep_metadata:
            eval_cmd.append("--keep-metadata")
        if args.stop_on_error:
            eval_cmd.append("--stop-on-error")

        completed = subprocess.run(eval_cmd, check=False)
        if completed.returncode != 0 and args.stop_on_error:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
