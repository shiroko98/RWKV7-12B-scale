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

## Project layout

- `README.md`
  Workspace overview, directory guide, and common commands.

- `todo.list`
  Current progress, decisions, and next-step checklist.

- `prune_layers.py`
  Main pruning entry. Reads a checkpoint, chooses layers to drop, rewrites block indices, and writes metadata.

- `rwkv_prune/`
  Pruning implementation details.
  Contains checkpoint inspection, parameter counting, target-layer estimation, and pruning strategies such as uniform, last-layer-preserving, importance-based, and neighbor-delta.

- `rwkv_scale/`
  Expansion experiments, kept separate from pruning so the two tracks do not interfere.
  Current mainline is depth-only expansion from the 7.2B model to 56 layers.

- `evals/`
  Shared evaluation code.
  Includes tokenizer loading, RWKV runtime, PPL evaluation, long-generation sampling, and repetition metrics.

- `tools/`
  Batch and helper scripts.
  Includes batch pruning, batch evaluation, target planning, and manifest rewriting for server paths.

- `tokenizer/`
  Tokenizer files used by the evaluation scripts.

- `outputs/pruned/`
  Pruning manifests, metadata JSON, and pruned model checkpoints.

- `outputs/expanded/`
  Expansion manifests, metadata JSON, and expanded model checkpoints.

- `outputs/evals/`
  Evaluation outputs such as `*.eval.json`, `summary.json`, and text logs.

- `demo_rnn.py`
  Original reference inference script. Useful for comparison, but the newer evaluation flow is easier to reproduce.

- `check_list.txt`
  Early manual notes kept for historical reference.

## Practical notes

- The original `demo_rnn.py` uses hard-coded Linux paths and CUDA defaults. Prefer the new modular scripts in this repo for repeatable experiments.
- A first-pass small perplexity benchmark is acceptable. Common choices are `WikiText-2`, `PTB`, `LAMBADA`, or a sampled subset of `C4`.
- When a step finishes, update `todo.list` before moving to the next experiment.

## Pruning workflow

Inspect the 13.3B checkpoint:

```bash
python prune_layers.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-13.3b-20260415-ctx8192.pth ^
  --output-model D:\codes\RWKV7-12B-scale\outputs\pruned\noop.pth ^
  --inspect-only
```

Generate all configured pruning variants:

```bash
python tools/batch_prune.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-13.3b-20260415-ctx8192.pth ^
  --output-dir D:\codes\RWKV7-12B-scale\outputs\pruned
```

Current 56-layer pruning candidates:

- `rwkv7-g1f-12b-56l-uniform`
- `rwkv7-g1f-12b-56l-last6`
- `rwkv7-g1f-12b-56l-importance`
- `rwkv7-g1f-12b-56l-neighbor-delta`

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

## Expansion workflow

The expansion experiments live under `rwkv_scale/` and currently focus on depth-only expansion from the local 7.2B checkpoint to 56 layers.

Generate all configured expansion variants:

```bash
python rwkv_scale/batch_expand.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-7.2b-20260414-ctx8192.pth ^
  --output-dir D:\codes\RWKV7-12B-scale\outputs\expanded ^
  --manifest-out D:\codes\RWKV7-12B-scale\outputs\expanded\manifest_56l_expand.json
```

Current 56-layer expansion candidates:

- `rwkv7-g1f-12b-expand-56l-uniform-interp`
- `rwkv7-g1f-12b-expand-56l-uniform-copy`
- `rwkv7-g1f-12b-expand-56l-hybrid-alt`
- `rwkv7-g1f-12b-expand-56l-tail-interp`

Create a Linux-server expansion manifest if needed:

```bash
python rwkv_scale/make_server_manifest.py ^
  --input-manifest D:\codes\RWKV7-12B-scale\outputs\expanded\manifest_56l_expand.json ^
  --server-root /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale ^
  --output-manifest D:\codes\RWKV7-12B-scale\outputs\expanded\manifest_56l_expand_server.json
```

Evaluate the expanded models with the same pipeline:

```bash
python tools/batch_eval.py ^
  --manifest D:\codes\RWKV7-12B-scale\outputs\expanded\manifest_56l_expand.json ^
  --tokenizer-path D:\codes\RWKV7-12B-scale\tokenizer\rwkv_vocab_v20250609.txt ^
  --device cuda ^
  --dtype bf16 ^
  --task both ^
  --dataset wikitext2 ^
  --token-budget 8192 ^
  --max-docs 128 ^
  --max-new-tokens 1200
```

## Server commands

Pruning evaluation on Linux server:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tools/batch_eval.py \
  --manifest /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/pruned/manifest_56l_server.json \
  --tokenizer-path /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tokenizer/rwkv_vocab_v20250609.txt \
  --device cuda \
  --dtype bf16 \
  --task both \
  --dataset wikitext2 \
  --token-budget 8192 \
  --max-docs 128 \
  --max-new-tokens 1200
```

Expansion generation on Linux server:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv_scale/batch_expand.py \
  --input-model /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --output-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded \
  --manifest-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded/manifest_56l_expand.json
```

Expansion evaluation on Linux server:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tools/batch_eval.py \
  --manifest /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded/manifest_56l_expand_server.json \
  --tokenizer-path /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tokenizer/rwkv_vocab_v20250609.txt \
  --device cuda \
  --dtype bf16 \
  --task both \
  --dataset wikitext2 \
  --token-budget 8192 \
  --max-docs 128 \
  --max-new-tokens 1200
```
