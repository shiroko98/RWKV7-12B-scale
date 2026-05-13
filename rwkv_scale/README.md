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

## Current result summary

The first full 56-layer expansion round has already produced a clear directional result:

- Copy-based insertion is much better than interpolation-based insertion.
- Pure interpolation and interpolation-heavy hybrids degrade badly and tend to loop.
- Even the best direct expansion result is still weaker than the best pruning result.

Server eval snapshot on `wikitext2` with `--token-budget 8192 --max-docs 128 --max-new-tokens 1200`:

- `uniform_copy`: `PPL ~= 16.14`, usable generation, no hard loop
- `hybrid_alt`: `PPL ~= 23.44`, strong repetition
- `uniform_interp`: `PPL ~= 60.58`, unusable
- `tail_interp`: `PPL ~= 84.79`, unusable

For comparison, the current best 56-layer pruning model (`rwkv7-g1f-12b-56l-importance`) is around `PPL ~= 10.24` on the same evaluation flow.

Working conclusion:

- If we continue direct expansion, the next round should stay on the copy side.
- The most promising immediate search space is which original layers to duplicate, not how to interpolate between layers.

Copy-only follow-up snapshot on the same evaluation flow:

- `tail_copy`: `PPL ~= 12.78`, best expansion result so far
- `boundary_delta_copy`: `PPL ~= 14.47`, second-best and fairly balanced
- `uniform_copy`: `PPL ~= 16.14`
- `importance_copy`: `PPL ~= 16.65`

Updated conclusion after the copy-focused round:

- `tail_copy` is now the strongest direct expansion variant.
- `boundary_delta_copy` is the next most promising heuristic.
- the normalized-weight-norm `importance_copy` heuristic did not beat `uniform_copy` here.
- even after this improvement, direct expansion still trails the best pruning result (`PPL ~= 10.24`).

## Expansion strategies

- `uniform_interp`: uniformly insert interpolated layers between interior layers
- `uniform_copy`: uniformly insert copied layers after interior layers
- `hybrid_alt`: uniformly insert layers and alternate interpolation / copy
- `tail_interp`: bias interpolated insertions toward later layers
- `tail_copy`: bias copied insertions toward later layers
- `importance_copy`: duplicate layers with higher normalized weight-norm scores
- `boundary_delta_copy`: duplicate layers after stronger neighbor-to-neighbor weight transitions
- `rys_repeat`: strictly duplicate a contiguous layer block in full RYS-style passes only

The current default batch is copy-only:

- `uniform_copy`
- `tail_copy`
- `importance_copy`
- `boundary_delta_copy`

The interpolation and hybrid baselines are still available, but only through an explicit config file so future runs do not accidentally mix strategies.

## RYS-style scan

To test Repeat-Your-Self style contiguous block repetition on the 7.2B to 56-layer expansion task, a dedicated scan config is included:

- `rwkv_scale/rys_scan_56l.json`
- `rwkv_scale/rys_scan_56l_full.json`
- `rwkv_scale/rys_scan_56l_half_min3.json`
- `rwkv_scale/rys_scan_56l_combo_example.json`

This config currently scans:

- start layers: `8`, `16`, `24`
- candidate block sizes: `3`, `4`, `6`, `8`, `12`, `24`
- after filtering, only start/block pairs fully inside the 32-layer source model are kept

Additional configs:

- `rys_scan_56l_full.json`: all legal strict-RYS single-block configs
- `rys_scan_56l_half_min3.json`: a reduced scan with `block_size >= 3` and `block_size <= 16`
- `rys_scan_56l_combo_example.json`: examples of multi-block strict-RYS combinations
- `rys_scan_repeat1_b2_b9.json`: single-block scan with `repeat_count = 1`, `block_size = 2..9`, and `target_layers = 32 + block_size`

The `rys_repeat` strategy is now strict RYS:

- it duplicates one contiguous block
- every duplicate pass must be a full block
- it does not allow prefix padding or partial-block fill
- for the current `32 -> 56` setup, the inserted depth is `24`, so valid block sizes must divide `24`
- by default, it does not allow blocks that start at `layer 0`, because RWKV block 0 has special `ln0` / `v_first` behavior and dormant `att.v*` parameters that are not semantically equivalent to later layers

Why `layer 0` is excluded by default:

- RWKV `block 0` is not a normal interior block
- `ln0` exists only on `block 0` and is fused into embedding normalization at inference time
- `v_first` is initialized from `layer_id == 0`, so copying layer 0 to a later position does not reproduce "run block 0 twice"
- in the 7.2B checkpoint, `blocks.0.att.v1` is all zeros and `blocks.0.att.v0` is a constant tensor, which is fine when `layer 0` ignores them, but unsafe once the copied block runs as a later layer and starts using them

If you really want to test this pathology case anyway, you can still opt in explicitly:

```bash
python rwkv_scale/depth_expand.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-7.2b-20260414-ctx8192.pth ^
  --output-model D:\codes\RWKV7-12B-scale\outputs\tmp\rys-layer0-test.pth ^
  --strategy rys_repeat ^
  --target-layers 33 ^
  --rys-start-layer 0 ^
  --rys-block-size 1 ^
  --rys-repeat-count 1 ^
  --allow-layer0-rys
```

Generate the default RYS scan pack:

```bash
python rwkv_scale/batch_expand.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-7.2b-20260414-ctx8192.pth ^
  --config D:\codes\RWKV7-12B-scale\rwkv_scale\rys_scan_56l.json ^
  --output-dir D:\codes\RWKV7-12B-scale\outputs\expanded_rys ^
  --manifest-out D:\codes\RWKV7-12B-scale\outputs\expanded_rys\manifest_rys.json
```

Build a custom scan config if needed:

```bash
python rwkv_scale/build_rys_scan_config.py ^
  --output D:\codes\RWKV7-12B-scale\rwkv_scale\rys_scan_custom.json ^
  --original-layers 32 ^
  --target-layers 56 ^
  --starts 0,4,8,12,16,20,24,28 ^
  --block-sizes 3,4,6,8,12,24
```

The config builders now skip `start=0` by default. Add `--include-layer0` only if you intentionally want to study that edge case.

Build a strict-RYS full-scan config over every legal start/block pair:

```bash
python rwkv_scale/build_rys_full_scan_config.py ^
  --output D:\codes\RWKV7-12B-scale\rwkv_scale\rys_scan_56l_full.json ^
  --original-layers 32 ^
  --target-layers 56
```

Build a variable-target single-repeat scan where each experiment duplicates exactly one block once:

```bash
python rwkv_scale/build_rys_full_scan_config.py ^
  --output D:\codes\RWKV7-12B-scale\rwkv_scale\rys_scan_repeat1_b2_b9.json ^
  --original-layers 32 ^
  --min-block-size 2 ^
  --max-block-size 9 ^
  --fixed-repeat-count 1 ^
  --name-prefix rwkv7-g1f-expand-repeat1-b2to9
```

This config keeps:

- single-block only
- `repeat_count = 1`
- `block_size = 2..9`
- `target_layers = 32 + block_size`
- no `layer 0` starts by default

If you want a more conservative search closer to the common RYS intuition, you can cap the block size to half depth:

```bash
python rwkv_scale/build_rys_full_scan_config.py ^
  --output D:\codes\RWKV7-12B-scale\rwkv_scale\rys_scan_56l_half.json ^
  --original-layers 32 ^
  --target-layers 56 ^
  --limit-half
```

Rewrite the RYS manifest for Linux server paths:

```bash
python rwkv_scale/make_server_manifest.py ^
  --input-manifest D:\codes\RWKV7-12B-scale\outputs\expanded_rys\manifest_rys.json ^
  --server-root /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale ^
  --output-manifest D:\codes\RWKV7-12B-scale\outputs\expanded_rys\manifest_rys_server.json
```

Run the full scan on server in a storage-friendly way:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tools/run_rys_scan.py \
  --input-model /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --config /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv_scale/rys_scan_56l_full.json \
  --work-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded_rys_tmp \
  --tokenizer-path /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tokenizer/rwkv_vocab_v20250609.txt \
  --summary-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_full_scan_summary.json \
  --markdown-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_full_scan_summary.md \
  --device cuda \
  --dtype bf16 \
  --task both \
  --dataset wikitext2 \
  --token-budget 8192 \
  --max-docs 128 \
  --max-new-tokens 200
```

Run the `repeat_count = 1`, `block_size = 2..9` scan on server:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tools/run_rys_scan.py \
  --input-model /mnt/data/Models/RWKV-7/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --config /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv_scale/rys_scan_repeat1_b2_b9.json \
  --work-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded_rys_repeat1_tmp \
  --tokenizer-path /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tokenizer/rwkv_vocab_v20250609.txt \
  --summary-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_repeat1_b2_b9_summary.json \
  --markdown-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_repeat1_b2_b9_summary.md \
  --log-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_repeat1_b2_b9_logs \
  --device cuda \
  --dtype bf16 \
  --task both \
  --dataset wikitext2 \
  --probes math,eq,json \
  --token-budget 8192 \
  --max-docs 128 \
  --max-new-tokens 200
```

Run the same scan on 8 GPUs:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tools/run_rys_scan_multigpu.py \
  --input-model /mnt/data/Models/RWKV-7/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --config /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv_scale/rys_scan_repeat1_b2_b9.json \
  --work-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded_rys_repeat1_tmp \
  --tokenizer-path /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tokenizer/rwkv_vocab_v20250609.txt \
  --summary-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_repeat1_b2_b9_summary.json \
  --markdown-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_repeat1_b2_b9_summary.md \
  --log-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_repeat1_b2_b9_logs \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --device cuda \
  --dtype bf16 \
  --task both \
  --dataset wikitext2 \
  --probes math,eq,json \
  --token-budget 8192 \
  --max-docs 128 \
  --max-new-tokens 200
```

This server flow does:

- build one expanded checkpoint
- run evaluation
- append the result to summary JSON
- optionally refresh a readable Markdown table
- delete the generated checkpoint and metadata by default
- delete the temporary per-model eval JSON by default, so one scan mainly leaves `summary.json`, optional `summary.md`, and logs

To include the lightweight RYS-style probes in the same run, add:

```bash
  --probes math,eq,json
```

If you explicitly want to keep each model's raw eval JSON too, add:

```bash
  --keep-per-model-json
```

Run the same scan on 8 GPUs in parallel:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tools/run_rys_scan_multigpu.py \
  --input-model /mnt/data/Models/RWKV-7/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --config /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/rwkv_scale/rys_scan_56l_full.json \
  --work-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded_rys_tmp \
  --tokenizer-path /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tokenizer/rwkv_vocab_v20250609.txt \
  --summary-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_full_scan_summary.json \
  --markdown-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_full_scan_summary.md \
  --log-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/evals/rys_full_scan_logs \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --device cuda \
  --dtype bf16 \
  --task both \
  --dataset wikitext2 \
  --probes math,eq,json \
  --token-budget 8192 \
  --max-docs 128 \
  --max-new-tokens 200
```

This multi-GPU runner:

- splits the config evenly across the listed GPUs
- launches one worker process per GPU
- writes separate temporary work dirs and shard summaries
- merges all shard summaries into one final `summary.json` and optional `summary.md`

### Full-scan result snapshot

The completed `rys_scan_56l_full.json` sweep finished with `196 / 196` records and no failures.

Main takeaway:

- simple single-layer repetition (`block_size = 1`, repeated 24 times) is clearly not the right direction
- large contiguous blocks copied a small number of times work much better
- the current best RYS-style expansion is `block_size = 24`, repeated once

Current RYS Top 5 by PPL:

| Rank | Name | Block | Repeat | PPL |
| --- | --- | --- | ---: | ---: |
| 1 | `rwkv7-g1f-12b-expand-56l-rys-s4-b24` | `4-27` | 1 | `11.33` |
| 2 | `rwkv7-g1f-12b-expand-56l-rys-s2-b24` | `2-25` | 1 | `11.36` |
| 3 | `rwkv7-g1f-12b-expand-56l-rys-s3-b24` | `3-26` | 1 | `11.38` |
| 4 | `rwkv7-g1f-12b-expand-56l-rys-s1-b24` | `1-24` | 1 | `11.59` |
| 5 | `rwkv7-g1f-12b-expand-56l-rys-s5-b12` | `5-16` | 2 | `11.72` |

Interpretation:

- compared with the earlier naive single-layer copy idea, these large-block RYS variants are much better
- within the RYS family, the best region is now `b24 x1`, followed by `b12 x2`
- even so, the current best RYS result still trails the best pruning result (`PPL ~= 10.24`), so pruning remains the stronger line for now

## Batch generation

```bash
python rwkv_scale/batch_expand.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-7.2b-20260414-ctx8192.pth
```

This produces:

- metadata JSON files for each copy-focused expansion candidate in `outputs/expanded_copy_focus/`
- a manifest at `outputs/expanded_copy_focus/manifest_copy_focus.json`
- model checkpoints when not using `--plan-only`

To generate a copy-focused batch into a separate output directory:

```bash
python rwkv_scale/batch_expand.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-7.2b-20260414-ctx8192.pth ^
  --config D:\codes\RWKV7-12B-scale\rwkv_scale\copy_focus_56l.json ^
  --output-dir D:\codes\RWKV7-12B-scale\outputs\expanded_copy_focus ^
  --manifest-out D:\codes\RWKV7-12B-scale\outputs\expanded_copy_focus\manifest_copy_focus.json
```

To regenerate the earlier interpolation / hybrid comparison pack:

```bash
python rwkv_scale/batch_expand.py ^
  --input-model D:\codes\RWKV7-12B-scale\rwkv7-g1f-7.2b-20260414-ctx8192.pth ^
  --config D:\codes\RWKV7-12B-scale\rwkv_scale\expand_baselines_56l.json ^
  --output-dir D:\codes\RWKV7-12B-scale\outputs\expanded ^
  --manifest-out D:\codes\RWKV7-12B-scale\outputs\expanded\manifest_56l_expand.json
```

## Server manifest

```bash
python rwkv_scale/make_server_manifest.py ^
  --input-manifest D:\codes\RWKV7-12B-scale\outputs\expanded_copy_focus\manifest_copy_focus.json ^
  --server-root /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale ^
  --output-manifest D:\codes\RWKV7-12B-scale\outputs\expanded_copy_focus\manifest_copy_focus_server.json
```

The manifest rewriter preserves the relative path under `outputs/`, so it also works for `outputs\expanded_copy_focus\...`.

Example:

```bash
python rwkv_scale/make_server_manifest.py ^
  --input-manifest D:\codes\RWKV7-12B-scale\outputs\expanded_copy_focus\manifest_copy_focus.json ^
  --server-root /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale ^
  --output-manifest D:\codes\RWKV7-12B-scale\outputs\expanded_copy_focus\manifest_copy_focus_server.json
```

## Evaluation

Use the existing batch evaluation entry with the generated expansion manifest:

```bash
python tools/batch_eval.py ^
  --manifest D:\codes\RWKV7-12B-scale\outputs\expanded_copy_focus\manifest_copy_focus.json ^
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
  --output-dir /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded_copy_focus \
  --manifest-out /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded_copy_focus/manifest_copy_focus.json
```

Linux server evaluation:

```bash
python /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tools/batch_eval.py \
  --manifest /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/expanded_copy_focus/manifest_copy_focus_server.json \
  --tokenizer-path /mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/tokenizer/rwkv_vocab_v20250609.txt \
  --device cuda \
  --dtype bf16 \
  --task both \
  --dataset wikitext2 \
  --token-budget 8192 \
  --max-docs 128 \
  --max-new-tokens 1200
```
