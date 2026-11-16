# Iteration 2 audit-fix status

This file records the corrections made after the review of commit `3dc22608dfb4f907d208b245b0461ff17f015599`. The research question is unchanged: how inference-time patch selection, pruning depth, and retained-token count affect accuracy and measured inference cost in a frozen CLIP-ReID encoder.

No new training, token fusion, learned scorer, Qwen run, dataset, or application feature was added. Colab was not executed from this machine.

## Verified defects and fixes

1. The encoder benchmark called the model without `torch.inference_mode()`. `model.eval()` left gradient tracking on. `time_encoder` now freezes every parameter, stays in eval mode, and runs warm-up, timed forwards, and the memory sample inside `torch.inference_mode()`. The timed forward includes L2 normalisation. The function raises if a parameter or buffer changes, or if a timed output requires grad. The encoder constructor also freezes parameters after loading.
2. The old `p90_batch_ms` was a percentile of trial averages. The new field `p90_batch_wall_ms` is the 90th percentile of individual batch wall times. Trial means are stored separately as `trial_mean_wall_ms`.
3. CUDA wall-clock time and CUDA-event time are separate fields. This machine has no CUDA, so the event fields are null. MPS `current_allocated_memory` is stored as `mps_current_allocated_bytes` and is explicitly not peak memory. `peak_cuda_bytes_allocated` is null here.
4. Method order is shuffled per trial with NumPy PCG64 and seed 553, and the order is stored.
5. New feature runs use immutable chunks. A repeated call does not rewrite a completed chunk. The cache key includes a whitelist source snapshot, package versions, the checkpoint checksum, image-byte hashes, preprocessing, descriptor construction, selection, precision, and autocast. Git revision is stored beside the key and is not part of it, so a documentation edit does not change the execution hash.
6. The first valid positive rank is taken from the full filtered gallery list. It is no longer truncated at rank 10.
7. The paper text was corrected. The historical official-test table was not changed. `paper/main.pdf` was rebuilt.

## Files changed

- `src/reidvlm/pruning/benchmark.py`
- `src/reidvlm/pruning/identity.py` (new)
- `src/reidvlm/pruning/study.py`
- `src/reidvlm/persistence.py`
- `src/reidvlm/cli.py`
- `src/reidvlm/encoders/clip_reid.py`
- `src/reidvlm/evaluation/metrics.py`
- `tests/test_iteration2.py`
- `paper/main.tex`
- `paper/main.pdf`
- `RESEARCH_ITERATION_2_STATUS.md` (timing correction note only)
- `notebooks/colab_iteration2_v2.ipynb` (new)

Preserved and not overwritten: `notebooks/colab_iteration2.ipynb`, `notebooks/colab_first_iteration.ipynb`, `results/market1501_test_20261115T182047Z.json`, `results/market1501_test_20261115T183006Z.json`, `results/market1501_test_20261115T122953Z.json`, `configs/iteration2_selected.json`, and the recorded validation grid in `RESEARCH_ITERATION_2_STATUS.md`.

## Commands actually executed

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_iteration2 -v
PYTHONPATH=src .venv/bin/python -m reidvlm prune-smoke --config configs/market1501.yaml --images 2 --output outputs/iteration2/audit_smoke.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-benchmark --config configs/market1501.yaml --batch-size 1 --warmup 1 --iterations 2 --trials 1 --prune-block 6 --keep 64 --seed 11 --output outputs/iteration2/audit_benchmark_short.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy cls_attention --prune-block 6 --keep 64 --backend reference --batch-size 2 --limit 2 --run-dir outputs/iteration2/audit_embed_limit2 --output outputs/iteration2/audit_embed_limit2.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy cls_attention --prune-block 6 --keep 64 --backend reference --batch-size 2 --limit 2 --run-dir outputs/iteration2/audit_embed_limit2 --output outputs/iteration2/audit_embed_limit2_resume.json
```

From `paper/`:

```bash
tectonic main.tex
```

The unit-test command was run again after the rank-10 fixture was corrected. The passing run reported 18 tests, OK, in 1.293 seconds, and printed `cls_row_max_abs_score_gap=1.490e-08` and `argsort_index_gap=0`.

The count check was:

```bash
PYTHONPATH=src .venv/bin/python -c "..."  # Market1501.load and build_validation_manifest
```

It printed query 3368, gallery 15913, train 12936, validation queries 751, validation gallery 12185.

## Source and checkpoint

- Git HEAD: `3dc22608dfb4f907d208b245b0461ff17f015599`
- Dirty state: `3dc22608dfb4f907d208b245b0461ff17f015599-dirty`
- Execution-source SHA-256 of the whitelist in this commit: `284a151018860cb7bb6ecad600c4d20ad604a58c47bd3694e9d6ab4af9487210`
- The local smoke logs were written before `src/reidvlm/cli.py` was added to that whitelist. They record `a51f95616457e311f400eec2eac32502f1b380bce38d6a0ec0c7baca117570e2`. They are not reused as a full validation cache.
- No whitelist file was missing.
- Checkpoint SHA-256: `bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc`
- Rebuilt PDF SHA-256: `6c2a5a1fccea36a84c564aba140d9ece2c2f9680b7fb8d64b1c75e7f3fdbe6ec`
- Local device: Apple MPS, torch 2.15.0, Python 3.12.11, float32, autocast off. CUDA is not available. The GPU is not a T4.

## Cache migration and recovery

Legacy feature tables are not relabelled. Their identity hashed paths and a dirty Git string. They do not contain image-byte hashes or the whitelist source snapshot, so compatibility with this code cannot be verified. New runs write `chunks-v1` under a new run directory.

The four-image limited embed wrote two chunk files and one identity file under `outputs/iteration2/audit_embed_limit2/chunks/9e6583d1753616fe99ab/`. The second call returned the same execution SHA-256, `672c4d4fd7b8f28d39acb123b4b8125078b34d206105bfd3de82bccd02923947`, and did not change the mtimes of those three files. That SHA belongs to the four-image limited manifest. It is not the identity of a full validation run.

The unit tests check that a changed image hash changes the execution identity, a documentation-only edit does not change the source snapshot, a seed change rejects a saved chunk identity, and an interrupted chunk resume matches an uninterrupted extraction.

## Validation

The earlier attention grid remains an MPS measurement on training identities, made before this correction. The five block-6 / 64 random controls remain T4 measurements from the previous notebook. Their original scores stay in `RESEARCH_ITERATION_2_STATUS.md` and `notebooks/colab_iteration2.ipynb`. They are not results of this corrected code, and they are not a matched table: the devices differ, and the random-run metadata does not include image-content hashes or this source snapshot. Those five runs are therefore not folded into a new table. New T4 runs are required.

No matched T4 validation result exists yet. The selected additional configuration is still block 9 / 96 patches. It was not changed.

The `--limit 2` embed printed mAP 1.0 with 1 valid query and 1 excluded query. That file is a cache smoke test. It is not a validation score.

## Official test

Not run with the corrected code. The local data counts match the protocol constants: 3,368 queries and 15,913 gallery images after junk removal. k-reciprocal was not run. The historical official-test JSON files were not modified.

## Timing

The short MPS numbers already printed in `RESEARCH_ITERATION_2_STATUS.md` stay as preliminary measurements. Their benchmark did not disable gradient tracking. They are not a corrected speedup.

The new short run is also not the planned protocol. It used batch size 1, 1 warm-up, 2 measured iterations, and 1 trial. `planned_protocol` is false. Seed 11 is the timing seed only. Device-resident scope: a preprocessed 1×3×256×128 batch already on MPS, through selection and L2 normalisation. Image loading is outside that scope. CUDA-event time is null. MPS allocated bytes were about 699,062,272 and are not peak memory. PyTorch reported its CUDA scaled-dot-product flags as true; CUDA itself was unavailable, so those flags are not evidence of a CUDA kernel.

Individual batch wall times, in milliseconds:

| Method | Batch walls | Median |
| --- | --- | ---: |
| Full encoder | 13.890, 14.397 | 14.144 |
| Attention, reference, block 6, keep 64 | 13.704, 12.547 | 13.125 |
| Attention, cls_row, block 6, keep 64 | 12.284, 12.262 | 12.273 |
| Random, seed 11, block 6, keep 64 | 16.721, 13.010 | 14.865 |

These two-sample ranges overlap. The printed ratios against the full encoder are 1.08, 1.15, and 0.95. They are not an established speedup. The planned protocol, batch sizes 1 and 32, 10 warm-up iterations, 50 measured iterations, and 3 trials, was not executed.

On the two real query images, reference and cls_row descriptors had max absolute difference 0, and the retained index tensors were equal, with 0 mismatches. The synthetic score gap remains 1.490e-8 with the same argsort. No tie case appeared on these two images. That does not certify the two backends on the whole dataset. cls_row stays an identified alternative. The reference backend remains the accuracy setting, and its extra attention pass stays inside the measured cost. Random selection does not request attention weights.

## Paper

`paper/main.tex` now describes the published comparison as the ViT result without side-information embeddings or overlapping patches (OLP). Rank-1 is stated as 95.50 − 94.98 = 0.52 percentage points, and the checkpoint is called an approximate reproduction. Cost in the historical result is the accuracy loss, not measured latency. The patch figure is one image. The text states that the class token and patch tokens have already mixed before the cut, and it no longer says that a high Rank-10 shows identity information concentrated in the retained patches. The historical table values are unchanged. PDF rebuilt with `tectonic main.tex` in `paper/`.

## Negative findings

- No T4 and no Colab session were available here, so the matched validation and official-test comparisons are unexecuted.
- The short corrected timing does not support a speedup claim.
- The five completed block-6 random results cannot be shown to match this code.
- The four-image embed score is not a Market-1501 result.
- k-reciprocal remains not run.

## Pending work and the blocker

The notebook checks out one recorded revision. It does not follow a newer `main`, and it must not be run at `3dc2260`. Paste this commit’s 40-character SHA into `REVISION` after the push. A blank value stops Cell 1. Run Cells 1–4, then one configuration. The first one is the full encoder on the training-derived validation split:

```bash
python -m reidvlm prune-embed \
  --config configs/market1501.yaml \
  --split validation \
  --strategy none \
  --prune-block 6 \
  --keep 128 \
  --backend reference \
  --batch-size 32 \
  --run-dir /content/drive/MyDrive/visibility-gated-reid/iteration2_v2/validation_none_b6_k128_reference \
  --output /content/drive/MyDrive/visibility-gated-reid/iteration2_v2/validation_none_b6_k128_reference.json
```

Then, still on that same T4 runtime and revision: attention block 6 / 64, attention block 9 / 96, and random block 9 / 96 for seeds 11, 23, 37, 53, and 71. Official-test runs come after those correctness checks. Do not start a second configuration while one is running. Do not delete `iteration2/` or `judgments/`.
