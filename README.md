# RWKV7 12B Scale Experiments

This workspace is for depth-pruning `rwkv7-g1f-13.3b-20260415-ctx8192.pth` down to an approximately 12B checkpoint, then validating the result with lightweight perplexity and repetition checks.

## Local environment

- Workspace: `D:\codes\RWKV7-12B-scale`
- Preferred conda env: `model`
- Target source checkpoint: `D:\codes\RWKV7-12B-scale\rwkv7-g1f-13.3b-20260415-ctx8192.pth`
- Tokenizer: `D:\codes\RWKV7-12B-scale\tokenizer\rwkv_vocab_v20250609.txt`
- Existing reference script: `D:\codes\RWKV7-12B-scale\demo_rnn.py`

## Working rules

- Keep large model weights out of git.
- Prefer CPU-safe validation first; add CUDA only when necessary.
- Make each meaningful iteration a git commit so pruning history stays inspectable.
- Record the current state and next steps in `todo.list`.

## Practical notes

- The original `demo_rnn.py` uses hard-coded Linux paths and CUDA defaults. Prefer the new modular scripts in this repo for repeatable experiments.
- A first-pass small perplexity benchmark is acceptable. Common choices are `WikiText-2`, `PTB`, `LAMBADA`, or a sampled subset of `C4`.
- When a step finishes, update `todo.list` before moving to the next experiment.

## Batch workflow

Generate all current pruning variants:

```bash
python tools/batch_prune.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-13.3b-20260415-ctx8192.pth ^
  --output-dir D:\codes\RWKV7-12B-scale\outputs\pruned
```

This writes:

- pruned model checkpoints in `outputs/pruned/`
- one metadata JSON per model
- a manifest file at `outputs/pruned/manifest.json`

Run batch evaluation on a GPU server:

```bash
python tools/batch_eval.py ^
  --manifest D:\codes\RWKV7-12B-scale\outputs\pruned\manifest.json ^
  --tokenizer-path D:\codes\RWKV7-12B-scale\tokenizer\rwkv_vocab_v20250609.txt ^
  --device cuda ^
  --dtype bf16 ^
  --task both ^
  --dataset wikitext2 ^
  --token-budget 2048 ^
  --max-docs 32 ^
  --max-new-tokens 512
```

Optional small-dataset alternatives:

- `--dataset lambada --dataset-path D:\codes\rwkv\misc\lambada_test.jsonl`
- `--dataset textfile --dataset-path path\to\your_eval_text.txt`
