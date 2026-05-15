from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RYS_VARIANTS = {
    "s1-b24": {
        "name": "rwkv7-g1f-12b-expand-56l-rys-s1-b24",
        "target_layers": 56,
        "rys_start_layer": 1,
        "rys_block_size": 24,
        "rys_repeat_count": 1,
    },
    "s9-b12": {
        "name": "rwkv7-g1f-12b-expand-56l-rys-s9-b12",
        "target_layers": 56,
        "rys_start_layer": 9,
        "rys_block_size": 12,
        "rys_repeat_count": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate selected 56-layer RYS expansion models from the 7.2B base checkpoint."
    )
    parser.add_argument("--input-model", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "expanded_rys_selected"),
        help="Directory to store generated checkpoints and metadata.",
    )
    parser.add_argument(
        "--variants",
        default="s1-b24,s9-b12",
        help="Comma-separated preset names. Available: s1-b24,s9-b12",
    )
    parser.add_argument(
        "--ranking-source",
        default=str(ROOT / "outputs" / "evals" / "rys_full_scan_summary.json"),
        help="Aggregate RYS scan summary JSON used to look up PPL and composite ranks.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not invoke depth_expand.py. Only emit commands and write ranking artifacts.",
    )
    return parser.parse_args()


def parse_variants(raw_variants: str) -> list[str]:
    variants = [item.strip() for item in raw_variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one variant must be selected.")

    unknown = [item for item in variants if item not in RYS_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")

    return variants


def build_command(
    python: str,
    input_model: str,
    output_dir: Path,
    variant_key: str,
    plan_only: bool,
) -> list[str]:
    variant = RYS_VARIANTS[variant_key]
    output_model = output_dir / f"{variant['name']}.pth"
    metadata_out = output_dir / f"{variant['name']}.json"

    cmd = [
        python,
        str(ROOT / "rwkv_scale" / "depth_expand.py"),
        "--input-model",
        input_model,
        "--output-model",
        str(output_model),
        "--strategy",
        "rys_repeat",
        "--target-layers",
        str(variant["target_layers"]),
        "--alpha",
        "0.5",
        "--rys-start-layer",
        str(variant["rys_start_layer"]),
        "--rys-block-size",
        str(variant["rys_block_size"]),
        "--rys-repeat-count",
        str(variant["rys_repeat_count"]),
        "--metadata-out",
        str(metadata_out),
    ]
    if plan_only:
        cmd.append("--plan-only")
    return cmd


def load_ranking_summary(summary_path: Path) -> dict[str, dict]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Ranking source not found: {summary_path}")

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for item in data:
        if "error" in item or "ppl" not in item:
            continue
        probes = item.get("probes", {})
        math_score = probes.get("math", {}).get("mean_score")
        eq_score = probes.get("eq", {}).get("mean_score")
        json_score = probes.get("json", {}).get("mean_score")
        if not all(isinstance(value, (int, float)) for value in [math_score, eq_score, json_score]):
            continue

        rows.append(
            {
                "name": item["name"],
                "ppl": float(item["ppl"]["ppl"]),
                "math": float(math_score),
                "eq": float(eq_score),
                "json": float(json_score),
                "composite": float(math_score + eq_score + json_score),
            }
        )

    ppl_rank = {
        row["name"]: index
        for index, row in enumerate(sorted(rows, key=lambda entry: entry["ppl"]), start=1)
    }
    composite_rank = {
        row["name"]: index
        for index, row in enumerate(
            sorted(rows, key=lambda entry: (-entry["composite"], entry["ppl"])),
            start=1,
        )
    }

    return {
        row["name"]: {
            **row,
            "ppl_rank": ppl_rank[row["name"]],
            "composite_rank": composite_rank[row["name"]],
        }
        for row in rows
    }


def write_rank_artifacts(output_dir: Path, variants: list[str], ranking_rows: dict[str, dict], summary_path: Path) -> None:
    records: list[dict] = []
    markdown_lines = [
        "# Selected RYS Model Ranks",
        "",
        f"- Ranking source: `{summary_path}`",
        "",
        "| Variant | Model | Block | Repeat | PPL | PPL Rank | Math | EQ | JSON | Composite | Composite Rank |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for variant_key in variants:
        variant = RYS_VARIANTS[variant_key]
        name = variant["name"]
        ranking = ranking_rows.get(name)
        if ranking is None:
            records.append(
                {
                    "variant": variant_key,
                    "name": name,
                    "block": f"{variant['rys_start_layer']}-{variant['rys_start_layer'] + variant['rys_block_size'] - 1}",
                    "repeat": variant["rys_repeat_count"],
                    "error": "model not found in ranking source",
                }
            )
            markdown_lines.append(
                f"| {variant_key} | {name} | {variant['rys_start_layer']}-{variant['rys_start_layer'] + variant['rys_block_size'] - 1} | {variant['rys_repeat_count']} |  |  |  |  |  |  |  |"
            )
            continue

        record = {
            "variant": variant_key,
            "name": name,
            "block": f"{variant['rys_start_layer']}-{variant['rys_start_layer'] + variant['rys_block_size'] - 1}",
            "repeat": variant["rys_repeat_count"],
            **ranking,
        }
        records.append(record)
        markdown_lines.append(
            f"| {variant_key} | {name} | {record['block']} | {record['repeat']} | "
            f"{record['ppl']:.6f} | {record['ppl_rank']} | {record['math']:.6f} | {record['eq']:.6f} | "
            f"{record['json']:.6f} | {record['composite']:.6f} | {record['composite_rank']} |"
        )

    (output_dir / "selected_rys_rankings.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "selected_rys_rankings.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant_key in variants:
        cmd = build_command(args.python, args.input_model, output_dir, variant_key, args.plan_only)
        print(f"\n=== generate {variant_key} ===")
        print(" ".join(cmd))
        if not args.skip_build:
            subprocess.run(cmd, check=True)

    summary_path = Path(args.ranking_source)
    ranking_rows = load_ranking_summary(summary_path)
    write_rank_artifacts(output_dir, variants, ranking_rows, summary_path)
    print(f"\nSaved ranking summary: {output_dir / 'selected_rys_rankings.json'}")
    print(f"Saved ranking markdown: {output_dir / 'selected_rys_rankings.md'}")


if __name__ == "__main__":
    main()
