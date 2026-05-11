from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = [
    {
        "name": "rwkv7-g1f-7.2b-base",
        "model_path": str(ROOT / "rwkv7-g1f-7.2b-20260414-ctx8192.pth"),
        "family": "base",
        "source_layers": 32,
    },
    {
        "name": "rwkv7-g1f-13.3b-base",
        "model_path": str(ROOT / "rwkv7-g1f-13.3b-20260415-ctx8192.pth"),
        "family": "base",
        "source_layers": 61,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the two original base checkpoints as baselines for PPL and generation."
    )
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
    parser.add_argument("--summary-out", default=str(ROOT / "outputs" / "evals" / "base_model_summary.json"))
    parser.add_argument("--config", help="Optional JSON list overriding the default two-model baseline set.")
    parser.add_argument("--model-7b-path", help="Optional override path for the 7.2B base checkpoint.")
    parser.add_argument("--model-13b-path", help="Optional override path for the 13.3B base checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_models(config_path: str | None, model_7b_path: str | None, model_13b_path: str | None) -> list[dict]:
    if config_path:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))

    models = [dict(item) for item in DEFAULT_MODELS]
    for item in models:
        if item["name"] == "rwkv7-g1f-7.2b-base" and model_7b_path:
            item["model_path"] = model_7b_path
        if item["name"] == "rwkv7-g1f-13.3b-base" and model_13b_path:
            item["model_path"] = model_13b_path
    return models


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(args.config, args.model_7b_path, args.model_13b_path)
    summary: list[dict] = []

    for item in models:
        model_name = item["name"]
        model_path = Path(item["model_path"])
        json_out = output_dir / f"{model_name}.eval.json"

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
            "--json-out",
            str(json_out),
        ]
        if args.dataset_path:
            cmd.extend(["--dataset-path", args.dataset_path])

        print(f"\n=== eval {model_name} ===")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            summary.append(
                {
                    "name": model_name,
                    "family": item.get("family", "base"),
                    "source_layers": item.get("source_layers"),
                    "model_path": str(model_path),
                    "error": f"evaluation failed with exit code {completed.returncode}",
                }
            )
            print(f"Evaluation failed for {model_name}, continuing to the next model.")
            continue

        result = json.loads(json_out.read_text(encoding="utf-8"))
        summary.append(
            {
                "name": model_name,
                "family": item.get("family", "base"),
                "source_layers": item.get("source_layers"),
                "model_path": str(model_path),
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
