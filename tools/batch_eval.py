from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for every model in a pruning manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json from batch_prune.py")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "evals"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--task", default="both", choices=["ppl", "generate", "both"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "lambada", "textfile"])
    parser.add_argument("--dataset-path")
    parser.add_argument("--max-docs", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--prompt", default="User: 请介绍一下北京。\n\nAssistant: ")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--summary-out", default=str(ROOT / "outputs" / "evals" / "summary.json"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    summary: list[dict] = []

    for item in manifest:
        model_name = item["name"]
        json_out = output_dir / f"{model_name}.eval.json"
        cmd = [
            args.python,
            str(ROOT / "evals" / "rwkv_eval.py"),
            "--model-path",
            item["output_model"],
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
            "--json-out",
            str(json_out),
        ]
        if args.dataset_path:
            cmd.extend(["--dataset-path", args.dataset_path])

        print(f"\n=== eval {model_name} ===")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            result = json.loads(json_out.read_text(encoding="utf-8"))
            summary.append(
                {
                    "name": model_name,
                    "strategy": item["strategy"],
                    "target_layers": item["target_layers"],
                    "output_model": item["output_model"],
                    "ppl": result.get("ppl", {}),
                    "generation_metrics": result.get("generation", {}).get("metrics", {}),
                }
            )

    if not args.dry_run:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
