from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from evals.rwkv_probes import BUILTIN_PROBES
from evals.rwkv_probes import ProbeSample
from evals.rwkv_probes import get_digit_score_tokens
from evals.rwkv_runtime import RWKVRNN, RuntimeConfig
from evals.rwkv_tokenizer import RWKVTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-friendly evaluation helpers for RWKV checkpoints.")
    parser.add_argument("--model-path", required=True, help="Checkpoint path, with or without .pth suffix.")
    parser.add_argument("--tokenizer-path", required=True, help="RWKV tokenizer vocab path.")
    parser.add_argument("--task", choices=["ppl", "generate", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "lambada", "textfile"])
    parser.add_argument("--dataset-path", help="Optional local dataset path. Required for textfile and lambada.")
    parser.add_argument("--max-docs", type=int, default=32, help="Maximum documents or samples to evaluate.")
    parser.add_argument("--token-budget", type=int, default=2048, help="Maximum token count for PPL evaluation.")
    parser.add_argument("--prompt", default="User: Beijing is the capital of\\n\\nAssistant: ")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--probes", default="", help="Comma-separated probes to run: math,eq,json")
    parser.add_argument("--json-out", help="Optional path to save evaluation results as JSON.")
    return parser.parse_args()


def load_eval_texts(dataset: str, dataset_path: str | None, max_docs: int) -> list[str]:
    if dataset == "lambada":
        if not dataset_path:
            raise ValueError("--dataset-path is required for lambada.")
        rows = []
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line)["text"])
                if len(rows) >= max_docs:
                    break
        return rows

    if dataset == "textfile":
        if not dataset_path:
            raise ValueError("--dataset-path is required for textfile.")
        text = Path(dataset_path).read_text(encoding="utf-8")
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        return paragraphs[:max_docs]

    if dataset == "wikitext2":
        from datasets import load_dataset

        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        rows = [row["text"] for row in data if row["text"].strip()]
        return rows[:max_docs]

    raise ValueError(f"Unsupported dataset: {dataset}")


def compute_ppl(model: RWKVRNN, tokenizer: RWKVTokenizer, texts: list[str], token_budget: int) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        if not tokens:
            continue

        state = model.zero_state()
        logits, state = model.forward(0, state)

        for token in tokens:
            log_prob = logits[token].item() - torch.logsumexp(logits, dim=-1).item()
            total_nll -= log_prob
            total_tokens += 1
            logits, state = model.forward(token, state)
            if total_tokens >= token_budget:
                break
        if total_tokens >= token_budget:
            break

    ppl = math.exp(total_nll / total_tokens) if total_tokens else float("inf")
    return {"ppl": ppl, "tokens": total_tokens, "nll": total_nll}


def sample_logits(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    if top_p < 1.0:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[min(int(cutoff_index), sorted_probs.numel() - 1)]
        probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    probs = probs / probs.sum()
    return torch.multinomial(probs, num_samples=1).item()


def repetition_metrics(text: str, tokens: list[int]) -> dict[str, float | int | str]:
    char_ngrams = [text[idx : idx + 3] for idx in range(max(0, len(text) - 2))]
    token_ngrams = [tuple(tokens[idx : idx + 3]) for idx in range(max(0, len(tokens) - 2))]

    char_repeat_ratio = 0.0
    token_repeat_ratio = 0.0
    if char_ngrams:
        char_counts = Counter(char_ngrams)
        char_repeat_ratio = sum(count - 1 for count in char_counts.values() if count > 1) / len(char_ngrams)
    if token_ngrams:
        token_counts = Counter(token_ngrams)
        token_repeat_ratio = sum(count - 1 for count in token_counts.values() if count > 1) / len(token_ngrams)

    loop_fragment = ""
    max_loop_repeats = 1
    for chunk_len in range(4, min(33, len(text) // 2 + 1)):
        for start in range(0, len(text) - chunk_len * 2 + 1):
            chunk = text[start : start + chunk_len]
            repeats = 1
            cursor = start + chunk_len
            while cursor + chunk_len <= len(text) and text[cursor : cursor + chunk_len] == chunk:
                repeats += 1
                cursor += chunk_len
            if repeats > max_loop_repeats:
                max_loop_repeats = repeats
                loop_fragment = chunk

    return {
        "text_length": len(text),
        "token_length": len(tokens),
        "unique_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "char_repeat_3gram_ratio": char_repeat_ratio,
        "token_repeat_3gram_ratio": token_repeat_ratio,
        "max_loop_repeats": max_loop_repeats,
        "loop_fragment": loop_fragment,
    }


def _logits_for_prompt(model: RWKVRNN, tokenizer: RWKVTokenizer, prompt: str) -> torch.Tensor:
    state = model.zero_state()
    logits = None
    for token in tokenizer.encode(prompt):
        logits, state = model.forward(token, state)
    if logits is None:
        logits, state = model.forward(0, state)
    return logits


def _score_probe_logits(
    logits: torch.Tensor,
    score_token_ids: list[int],
    score_values: list[float],
    correct_answer: int | None,
) -> dict[str, float | int | list[float] | bool | None]:
    restricted_logits = logits[score_token_ids].float()
    probs = torch.softmax(restricted_logits, dim=0)
    expected = float(sum(value * prob.item() for value, prob in zip(score_values, probs)))
    variance = float(sum(((value - expected) ** 2) * prob.item() for value, prob in zip(score_values, probs)))
    argmax_idx = int(torch.argmax(restricted_logits).item())
    predicted_digit = int(score_values[argmax_idx])

    log_odds = None
    is_correct = None
    if correct_answer is not None:
        correct_idx = None
        for idx, value in enumerate(score_values):
            if int(value) == correct_answer:
                correct_idx = idx
                break
        if correct_idx is not None:
            log_softmax = torch.log_softmax(restricted_logits, dim=0)
            log_p_correct = float(log_softmax[correct_idx].item())
            other_logits = torch.cat([restricted_logits[:correct_idx], restricted_logits[correct_idx + 1 :]])
            if len(other_logits) > 0:
                log_p_other = float(torch.logsumexp(other_logits, dim=0).item())
                log_odds = log_p_correct - log_p_other
            else:
                log_odds = float("inf")
            is_correct = predicted_digit == correct_answer

    return {
        "expected_score": expected,
        "uncertainty": variance,
        "predicted_digit": predicted_digit,
        "probabilities": [float(prob.item()) for prob in probs],
        "raw_logits": [float(x.item()) for x in restricted_logits],
        "log_odds": log_odds,
        "is_correct": is_correct,
    }


def run_probe(
    model: RWKVRNN,
    tokenizer: RWKVTokenizer,
    probe_name: str,
    samples: list[ProbeSample],
) -> dict[str, object]:
    score_token_ids, score_values = get_digit_score_tokens(tokenizer)
    per_sample: list[dict[str, object]] = []
    expected_scores: list[float] = []
    uncertainties: list[float] = []
    log_odds_values: list[float] = []
    correctness: list[bool] = []

    for sample in samples:
        logits = _logits_for_prompt(model, tokenizer, sample.prompt)
        scored = _score_probe_logits(logits, score_token_ids, score_values, sample.correct_answer)
        per_sample.append(
            {
                "prompt": sample.prompt,
                "category": sample.category,
                "target_score": sample.expected_score,
                **scored,
            }
        )
        expected_scores.append(float(scored["expected_score"]))
        uncertainties.append(float(scored["uncertainty"]))
        if scored["log_odds"] is not None:
            log_odds_values.append(float(scored["log_odds"]))
        if scored["is_correct"] is not None:
            correctness.append(bool(scored["is_correct"]))

    result: dict[str, object] = {
        "probe": probe_name,
        "sample_count": len(samples),
        "mean_score": sum(expected_scores) / len(expected_scores) if expected_scores else 0.0,
        "mean_uncertainty": sum(uncertainties) / len(uncertainties) if uncertainties else 0.0,
        "samples": per_sample,
    }
    if log_odds_values:
        result["mean_log_odds"] = sum(log_odds_values) / len(log_odds_values)
    if correctness:
        result["accuracy"] = sum(1.0 for flag in correctness if flag) / len(correctness)
    return result


def generate_text(
    model: RWKVRNN,
    tokenizer: RWKVTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, object]:
    state = model.zero_state()
    logits = None
    for token in tokenizer.encode(prompt):
        logits, state = model.forward(token, state)
    if logits is None:
        logits, state = model.forward(0, state)

    generated_tokens: list[int] = []
    for _ in range(max_new_tokens):
        token = sample_logits(logits, temperature=temperature, top_p=top_p)
        generated_tokens.append(token)
        logits, state = model.forward(token, state)

    text = tokenizer.decode(generated_tokens)
    metrics = repetition_metrics(text, generated_tokens)
    metrics["unknown_token_count"] = tokenizer.count_unknown(generated_tokens)
    return {"prompt": prompt, "text": text, "metrics": metrics}


def main() -> None:
    args = parse_args()
    tokenizer = RWKVTokenizer(args.tokenizer_path)
    model = RWKVRNN(RuntimeConfig(model_path=args.model_path, device=args.device, dtype=args.dtype))

    result: dict[str, object] = {
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
    }

    if args.task in {"ppl", "both"}:
        texts = load_eval_texts(args.dataset, args.dataset_path, args.max_docs)
        ppl_result = compute_ppl(model, tokenizer, texts, token_budget=args.token_budget)
        result["ppl"] = {"dataset": args.dataset, **ppl_result}
        print(f"PPL dataset : {args.dataset}")
        print(f"Token budget: {ppl_result['tokens']}")
        print(f"PPL         : {ppl_result['ppl']:.4f}")

    if args.task in {"generate", "both"}:
        generation = generate_text(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        result["generation"] = generation
        print("Generation metrics:")
        for key, value in generation["metrics"].items():
            print(f"  {key}: {value}")
        print("\nGenerated text:\n")
        print(generation["text"])

    probe_names = [item.strip() for item in args.probes.split(",") if item.strip()]
    if probe_names:
        probe_results: dict[str, object] = {}
        for probe_name in probe_names:
            if probe_name not in BUILTIN_PROBES:
                raise ValueError(f"Unsupported probe: {probe_name}")
            probe_result = run_probe(model, tokenizer, probe_name, BUILTIN_PROBES[probe_name])
            probe_results[probe_name] = probe_result
            print(f"\nProbe {probe_name}:")
            print(f"  sample_count: {probe_result['sample_count']}")
            print(f"  mean_score  : {probe_result['mean_score']:.4f}")
            print(f"  mean_uncert.: {probe_result['mean_uncertainty']:.4f}")
            if "mean_log_odds" in probe_result:
                print(f"  mean_log_odds: {probe_result['mean_log_odds']:.4f}")
            if "accuracy" in probe_result:
                print(f"  accuracy    : {probe_result['accuracy']:.4f}")
        result["probes"] = probe_results

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
