# RWKV Scale

This directory contains depth-expansion tooling for growing `rwkv7-g1f-7.2b-20260414-ctx8192.pth` into approximately 12B-class checkpoints without touching the pruning pipeline.

## Directory guide

- `depth_expand.py`
  Main depth-only expansion entry.
  Reads a 32-layer checkpoint, inserts new layers by copy or interpolation, and writes a 56-layer checkpoint plus metadata.

- `batch_expand.py`
  Batch expansion driver.
  Generates the configured experiment family and writes a manifest for later evaluation.

- `make_server_manifest.py`
  Rewrites Windows paths in the expansion manifest into Linux server paths.

- `scale.py`
  Earlier research script focused on depth expansion plus a simpler width-expansion path.

- `scale_2.py`
  Earlier research script combining depth expansion, width expansion, optional `graft_init_segments`, and head-size changes.

- `scale_g0a.py`
  Variant of `scale_2.py` specialized for a different base checkpoint family.

- `Net2Net.py`
  Earlier research script using Net2Net / Net2WiderNet style width expansion.

- `compare_weights.py`
  Utility for comparing trained expanded weights against an initialization checkpoint, especially the newly added dimensions.

## Why depth-only first

The local 7.2B and 13.3B checkpoints share the same:

- `n_embd = 4096`
- `n_head = 64`
- `head_size = 64`

The main parameter difference comes from depth:

- 7.2B: 32 layers
- 13.3B: 61 layers

So the fairest comparison against the 56-layer pruning results is a 56-layer depth-expanded model family, while keeping the hidden width fixed at 4096.

## Local structure notes

Direct checkpoint inspection shows:

- `7.2B`: 32 layers, `n_embd=4096`, `n_head=64`, `head_size=64`
- `13.3B`: 61 layers, `n_embd=4096`, `n_head=64`, `head_size=64`

The main parameter increase comes from depth, not hidden width.

Shared main matrix shapes:

- attention main projections stay `(4096, 4096)`
- FFN main projections stay `(16384, 4096)` and `(4096, 16384)`

Small-rank differences between the two checkpoints:

- `att.w1 / att.w2`: `128 <-> 192`
- `att.a1 / att.a2`: `128 <-> 192`
- `att.v1 / att.v2`: `96 <-> 128`
- `att.g1 / att.g2`: `480 <-> 384`

Interpretation for the scaling experiment:

- First priority: test depth expansion to 56 layers with width fixed at 4096.
- Second priority, only if needed: test whether adjusting small-rank internal dimensions (`w/a/v/g`) improves the depth-expanded model.
- Width expansion to `6144` is a more aggressive structural change and is not the closest match to the observed 13.3B family structure.

## Current hypothesis

From the expanded-model-performance perspective, depth expansion is currently the safer first bet than width expansion:

- It matches the observed larger-model family structure more closely.
- It preserves the main tensor geometry of the pretrained 7.2B checkpoint.
- It introduces fewer new axes and fewer shape mismatches than width expansion.
- It should therefore have a lower risk of damaging the pretrained function before finetuning.

This is still a hypothesis, not a proof. The planned comparison is:

1. Compare multiple depth-only expansion strategies at 56 layers.
2. Reuse the same evaluation pipeline as pruning.
3. Only if depth-only is still weak, add a second-stage width or small-rank expansion study.

## Expansion strategies

- `uniform_interp`: uniformly insert interpolated layers between interior layers
- `uniform_copy`: uniformly insert copied layers after interior layers
- `hybrid_alt`: uniformly insert layers and alternate interpolation / copy
- `tail_interp`: bias interpolated insertions toward later layers

## Batch generation

```bash
python rwkv_scale/batch_expand.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-7.2b-20260414-ctx8192.pth ^
  --output-dir D:\codes\RWKV7-12B-scale\outputs\expanded
```

This produces:

- metadata JSON files for each expansion candidate in `outputs/expanded/`
- a manifest at `outputs/expanded/manifest_56l_expand.json`
- model checkpoints when not using `--plan-only`

## Server manifest

```bash
python rwkv_scale/make_server_manifest.py ^
  --input-manifest D:\codes\RWKV7-12B-scale\outputs\expanded\manifest_56l_expand.json ^
  --server-root /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale ^
  --output-manifest D:\codes\RWKV7-12B-scale\outputs\expanded\manifest_56l_expand_server.json
```

## Evaluation

Use the existing batch evaluation entry with the generated expansion manifest:

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

Linux server generation:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv_scale/batch_expand.py \
  --input-model /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --output-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded \
  --manifest-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded/manifest_56l_expand.json
```

Linux server evaluation:

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
