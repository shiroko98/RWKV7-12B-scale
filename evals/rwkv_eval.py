from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

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

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
