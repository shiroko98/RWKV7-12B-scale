from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlockRecord:
    name: str
    start: int
    size: int
    end: int
    ppl: float
    math_score: float
    eq_score: float
    json_score: float
    unknown_tokens: int
    max_loop_repeats: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select non-overlapping multi-block RYS combos from repeat=1 scan results and emit 56-layer configs."
    )
    parser.add_argument("--summary-json", required=True, help="repeat=1 scan summary JSON, e.g. rys_repeat1_b2_b9_summary.json")
    parser.add_argument("--output-config", required=True, help="Output config JSON for run_rys_scan(.py/.multigpu.py)")
    parser.add_argument("--output-report", help="Optional markdown report with recommended combinations.")
    parser.add_argument("--name-prefix", default="rwkv7-g1f-12b-expand-56l-rys-combo")
    parser.add_argument("--target-layers", type=int, default=56)
    parser.add_argument("--original-layers", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=6)
    parser.add_argument("--min-blocks", type=int, default=3)
    parser.add_argument("--candidate-ppl-threshold", type=float, default=12.0)
    parser.add_argument("--min-start", type=int, default=2)
    parser.add_argument("--max-end", type=int, default=28)
    parser.add_argument("--top-k-per-mode", type=int, default=2)
    parser.add_argument(
        "--anchor-combo-count",
        type=int,
        default=2,
        help="Add this many combos anchored on top-ranked single blocks from the mid-layer sweet spot.",
    )
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _safe_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def load_records(path: Path) -> list[BlockRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: list[BlockRecord] = []
    for item in raw:
        if "error" in item:
            continue
        ppl = item.get("ppl", {}).get("ppl")
        start = item.get("rys_start_layer")
        size = item.get("rys_block_size")
        repeat = item.get("rys_repeat_count")
        if ppl is None or start is None or size is None:
            continue
        if int(repeat or 0) != 1:
            continue
        probes = item.get("probes", {})
        generation = item.get("generation_metrics", {})
        start_i = int(start)
        size_i = int(size)
        records.append(
            BlockRecord(
                name=str(item.get("name", f"s{start_i}-b{size_i}-r1")),
                start=start_i,
                size=size_i,
                end=start_i + size_i - 1,
                ppl=float(ppl),
                math_score=_safe_float(probes.get("math", {}).get("mean_score")),
                eq_score=_safe_float(probes.get("eq", {}).get("mean_score")),
                json_score=_safe_float(probes.get("json", {}).get("mean_score")),
                unknown_tokens=_safe_int(generation.get("unknown_token_count")),
                max_loop_repeats=_safe_int(generation.get("max_loop_repeats")),
            )
        )
    return records


def filter_candidates(records: list[BlockRecord], *, min_start: int, max_end: int, candidate_ppl_threshold: float) -> list[BlockRecord]:
    return [
        record
        for record in records
        if record.start >= min_start
        and record.end <= max_end
        and math.isfinite(record.ppl)
        and record.ppl < candidate_ppl_threshold
    ]


def block_cost(mode: str, record: BlockRecord) -> float:
    center = (record.start + record.end) / 2.0
    if mode == "few_blocks":
        return record.ppl - 0.03 * record.size + 0.015 * max(0.0, record.start - 12.0)
    if mode == "quality_first":
        return record.ppl - 0.05 * record.size
    if mode == "mid_focus":
        return record.ppl - 0.04 * record.size + 0.02 * abs(center - 10.0)
    if mode == "probe_balance":
        return (
            record.ppl
            - 0.02 * record.size
            - 0.01 * (record.math_score + record.eq_score + record.json_score)
            + 0.02 * record.unknown_tokens
        )
    raise ValueError(f"Unknown mode: {mode}")


def choose_best_combo(
    records: list[BlockRecord],
    *,
    inserted_layers: int,
    exact_blocks: int,
    mode: str,
) -> list[BlockRecord] | None:
    if exact_blocks <= 0:
        return None
    ordered = sorted(records, key=lambda item: (item.end, item.start, item.size))
    prev_idx: list[int] = []
    for idx, item in enumerate(ordered):
        best_prev = -1
        lo = 0
        hi = idx - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ordered[mid].end < item.start:
                best_prev = mid
                lo = mid + 1
            else:
                hi = mid - 1
        prev_idx.append(best_prev)

    inf = float("inf")
    n = len(ordered)
    dp = [[[inf] * (exact_blocks + 1) for _ in range(inserted_layers + 1)] for _ in range(n + 1)]
    take = [[[False] * (exact_blocks + 1) for _ in range(inserted_layers + 1)] for _ in range(n + 1)]
    back = [[[(None, None, None)] * (exact_blocks + 1) for _ in range(inserted_layers + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0][0] = 0.0

    for i in range(1, n + 1):
        item = ordered[i - 1]
        for total in range(inserted_layers + 1):
            for blocks in range(exact_blocks + 1):
                dp[i][total][blocks] = dp[i - 1][total][blocks]
                back[i][total][blocks] = (i - 1, total, blocks)

        compatible_prefix = prev_idx[i - 1] + 1
        item_cost = block_cost(mode, item)
        block_penalty = 0.015 if mode == "quality_first" else 0.03
        for total in range(item.size, inserted_layers + 1):
            for blocks in range(1, exact_blocks + 1):
                previous = dp[compatible_prefix][total - item.size][blocks - 1]
                if previous == inf:
                    continue
                candidate = previous + item_cost + block_penalty
                if candidate < dp[i][total][blocks]:
                    dp[i][total][blocks] = candidate
                    take[i][total][blocks] = True
                    back[i][total][blocks] = (compatible_prefix, total - item.size, blocks - 1)

    if dp[n][inserted_layers][exact_blocks] == inf:
        return None

    combo: list[BlockRecord] = []
    i = n
    total = inserted_layers
    blocks = exact_blocks
    while i is not None and i > 0:
        prev_i, prev_total, prev_blocks = back[i][total][blocks]
        if take[i][total][blocks]:
            combo.append(ordered[i - 1])
        i, total, blocks = prev_i, prev_total, prev_blocks

    combo.sort(key=lambda item: item.start)
    return combo


def combo_key(combo: list[BlockRecord]) -> tuple[tuple[int, int], ...]:
    return tuple((item.start, item.size) for item in combo)


def combo_name(name_prefix: str, label: str, combo: list[BlockRecord]) -> str:
    suffix = "-".join(f"s{item.start}b{item.size}" for item in combo)
    return f"{name_prefix}-{label}-{suffix}"


def combo_to_config(name_prefix: str, label: str, combo: list[BlockRecord], target_layers: int) -> dict:
    return {
        "name": combo_name(name_prefix, label, combo),
        "strategy": "rys_repeat",
        "target_layers": target_layers,
        "alpha": 0.5,
        "selection_label": label,
        "selection_stats": {
            "block_count": len(combo),
            "sum_single_block_ppl": round(sum(item.ppl for item in combo), 6),
            "avg_single_block_ppl": round(sum(item.ppl for item in combo) / len(combo), 6),
            "max_single_block_ppl": round(max(item.ppl for item in combo), 6),
        },
        "rys_blocks": [{"start": item.start, "size": item.size, "repeat": 1} for item in combo],
    }


def _best_anchor_combo(
    candidates: list[BlockRecord],
    *,
    anchor: BlockRecord,
    inserted_layers: int,
    min_blocks: int,
    max_blocks: int,
) -> list[BlockRecord] | None:
    remaining = [
        record
        for record in candidates
        if record.end < anchor.start or record.start > anchor.end
    ]
    target_inserted = inserted_layers - anchor.size
    if target_inserted <= 0:
        return None

    best_combo: list[BlockRecord] | None = None
    best_signature: tuple[float, float, int] | None = None
    for total_blocks in range(max(min_blocks, 2), max_blocks + 1):
        rest_blocks = total_blocks - 1
        if rest_blocks <= 0:
            continue
        rest_combo = choose_best_combo(
            remaining,
            inserted_layers=target_inserted,
            exact_blocks=rest_blocks,
            mode="quality_first",
        )
        if not rest_combo:
            continue
        combo = sorted([anchor, *rest_combo], key=lambda item: item.start)
        avg_ppl = sum(item.ppl for item in combo) / len(combo)
        max_ppl = max(item.ppl for item in combo)
        signature = (avg_ppl, max_ppl, len(combo))
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_combo = combo
    return best_combo


def build_report(records: list[BlockRecord], candidates: list[BlockRecord], configs: list[dict]) -> str:
    lines: list[str] = ["# RYS Combo Selection", ""]
    lines.append(f"- Total repeat=1 records loaded: {len(records)}")
    lines.append(f"- Candidate records after filtering: {len(candidates)}")
    lines.append("")
    lines.append("## Selected combinations")
    lines.append("")
    lines.append("| Label | Blocks | Avg single-block PPL | Max single-block PPL | Segments |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for config in configs:
        stats = config.get("selection_stats", {})
        segments = ",".join(
            f"{block['start']}-{block['start'] + block['size'] - 1}"
            for block in config.get("rys_blocks", [])
        )
        lines.append(
            f"| {config.get('selection_label', '')} | {stats.get('block_count', '')} | "
            f"{stats.get('avg_single_block_ppl', '')} | {stats.get('max_single_block_ppl', '')} | {segments} |"
        )
    lines.append("")
    lines.append("## Config JSON Preview")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(configs, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    inserted_layers = args.target_layers - args.original_layers
    if inserted_layers <= 0:
        raise ValueError("target-layers must be larger than original-layers.")

    records = load_records(Path(args.summary_json))
    candidates = filter_candidates(
        records,
        min_start=args.min_start,
        max_end=args.max_end,
        candidate_ppl_threshold=args.candidate_ppl_threshold,
    )
    if not candidates:
        raise ValueError("No candidate blocks survived filtering.")

    requested_modes = [
        ("fewblocks-k3", "few_blocks", 3),
        ("fewblocks-k4", "few_blocks", 4),
        ("quality-k5", "quality_first", 5),
        ("quality-k6", "quality_first", 6),
        ("midfocus-k5", "mid_focus", 5),
        ("probe-k4", "probe_balance", 4),
    ]

    unique: dict[tuple[tuple[int, int], ...], dict] = {}
    for label, mode, exact_blocks in requested_modes:
        if exact_blocks < args.min_blocks or exact_blocks > args.max_blocks:
            continue
        combo = choose_best_combo(
            candidates,
            inserted_layers=inserted_layers,
            exact_blocks=exact_blocks,
            mode=mode,
        )
        if not combo:
            continue
        key = combo_key(combo)
        if key in unique:
            continue
        unique[key] = combo_to_config(args.name_prefix, label, combo, args.target_layers)

    if args.anchor_combo_count > 0:
        ranked_anchors = [
            record
            for record in sorted(candidates, key=lambda item: item.ppl)
            if 6 <= record.start <= 9 and record.size >= 4
        ]
        added = 0
        for anchor in ranked_anchors:
            combo = _best_anchor_combo(
                candidates,
                anchor=anchor,
                inserted_layers=inserted_layers,
                min_blocks=max(args.min_blocks, 5),
                max_blocks=args.max_blocks,
            )
            if not combo:
                continue
            key = combo_key(combo)
            if key in unique:
                continue
            label = f"anchor-s{anchor.start}b{anchor.size}"
            unique[key] = combo_to_config(args.name_prefix, label, combo, args.target_layers)
            added += 1
            if added >= args.anchor_combo_count:
                break

    configs = list(unique.values())
    if not configs:
        raise ValueError("Failed to construct any valid combo configs.")

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{output_path} ({len(configs)} combos)")

    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_report(records, candidates, configs), encoding="utf-8")
        print(report_path)


if __name__ == "__main__":
    main()
