# Iteration 2 status

This file records what was implemented and what was actually run on this machine. It does not replace the historical Market-1501 files, and it does not claim that those files were reproduced here.

## Files

- `src/reidvlm/pruning/selection.py`: strategies `none`, `cls_attention`, and `random`; stable tie break; CLS-row scores; content-id seeding.
- `src/reidvlm/pruning/study.py`: experiment plan, training-derived manifest, configuration rule, resumable embedding, paired query counts.
- `src/reidvlm/pruning/benchmark.py`: encoder timing through the L2-normalised descriptor.
- `src/reidvlm/encoders/clip_reid.py`: new selection path after a complete block. The older `keep_ratio` path still uses `topk`. LayerNorm on the attention input is computed once on both paths.
- `src/reidvlm/persistence.py`: verified feature tables that can grow. A different identity is not reused.
- `src/reidvlm/cli.py`: `prune-plan`, `prune-manifest`, `prune-benchmark`, `prune-smoke`, `prune-embed`.
- `tests/test_iteration2.py`: correctness tests. Synthetic tensors are not retrieval scores.
- `configs/iteration2_plan.json`: the grid, seeds, validation rule, and timing protocol, written before the new validation runs.
- `configs/iteration2_validation_counts.json`: counts from the constructed manifest.
- `configs/iteration2_selected.json`: the additional configuration chosen from validation mAP.
- `notebooks/colab_iteration2.ipynb`: the Colab notebook for this iteration. A Tesla T4 session ran the five random seeds below. The notebook output is the record of those scores.
- `README.md`: token pruning is the current study. The Qwen notebook is labelled historical.
- `paper/main.tex` and `paper/main.pdf`: the raw top-five rate is no longer described as CMC Rank-5. No new experiment numbers were added to the paper.

`notebooks/colab_first_iteration.ipynb`, `results/market1501_test_20261115T182047Z.json`, and `results/market1501_test_20261115T183006Z.json` were not modified.

## Commands that finished

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_iteration2 -v
PYTHONPATH=src .venv/bin/python -m reidvlm prune-plan --output configs/iteration2_plan.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-manifest --config configs/market1501.yaml --output outputs/iteration2/validation_manifest.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-smoke --config configs/market1501.yaml --images 2 --output outputs/iteration2/smoke.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-benchmark --config configs/market1501.yaml --batch-size 1 --warmup 3 --iterations 10 --trials 2 --prune-block 6 --keep 64 --seed 11 --output outputs/iteration2/benchmark_b1_short.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-benchmark --config configs/market1501.yaml --batch-size 32 --warmup 2 --iterations 5 --trials 1 --prune-block 6 --keep 64 --seed 11 --output outputs/iteration2/benchmark_b32_short.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy none --batch-size 32 --run-dir outputs/iteration2/val_full --output outputs/iteration2/val_full.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy cls_attention --prune-block 6 --keep 96 --backend reference --batch-size 32 --run-dir outputs/iteration2/val_attention_b6_k96 --output outputs/iteration2/val_attention_b6_k96.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy cls_attention --prune-block 6 --keep 64 --backend reference --batch-size 32 --run-dir outputs/iteration2/val_attention_b6_k64 --output outputs/iteration2/val_attention_b6_k64.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy cls_attention --prune-block 9 --keep 96 --backend reference --batch-size 32 --run-dir outputs/iteration2/val_attention_b9_k96 --output outputs/iteration2/val_attention_b9_k96.json
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy cls_attention --prune-block 9 --keep 64 --backend reference --batch-size 32 --run-dir outputs/iteration2/val_attention_b9_k64 --output outputs/iteration2/val_attention_b9_k64.json
tectonic main.tex
```

The unit-test run reported 12 tests, OK, in 0.961 seconds. It printed `cls_row_max_abs_score_gap=1.490e-08` and `argsort_index_gap=0`.

## Checkpoint and source

- Checkpoint SHA-256: `bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc`.
- Git HEAD at the start of the run: `ffb01aabc795afe8bac1c559bdebf479a49628f5`.
- The working tree was dirty, so cache identities say `ffb01aabc795afe8bac1c559bdebf479a49628f5-dirty`.
- The code used for the Colab runs is on `main`. The T4 session reported `torch` 2.15.0+cu132 and a Tesla T4.
- Local checks before those runs used Apple MPS, `torch` 2.15.0, 16 GB memory, float32, autocast off, and SIE off.

## Validation manifest

Official training images only. Seed 553. One query image per identity that has a second camera. The remaining training images are the gallery. Query and gallery paths are disjoint.

- Eligible identities: 751
- Query images: 751
- Gallery images: 12,185
- Training images used: 12,936
- Valid queries with a cross-camera positive: 751
- Excluded queries: 0
- Cameras in both sets: 0 through 5

The released encoder was trained on these identities. This is not an identity-unseen validation set. A high score here does not transfer to the official test.

## Validation accuracy

Raw cosine retrieval. k-reciprocal was not run. All five runs used every training image in the manifest above. They are not official-test scores.

| Setting | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| Full encoder | 97.93 | 98.40 | 99.87 | 100.00 |
| Attention, block 9, 96 patches | 97.80 | 98.14 | 99.87 | 100.00 |
| Attention, block 6, 96 patches | 97.04 | 98.00 | 99.33 | 99.73 |
| Attention, block 9, 64 patches | 96.96 | 97.87 | 99.60 | 99.87 |
| Attention, block 6, 64 patches | 88.30 | 94.01 | 98.40 | 99.47 |

Against the full encoder on this same split, block 9 / 96 patches is 0.12 mAP points and 0.26 Rank-1 points lower. Block 6 / 64 patches is 9.62 mAP points and 4.39 Rank-1 points lower. Of 751 queries, block 6 / 64 improves 13 average precisions, worsens 456, and leaves 282 unchanged. Block 9 / 96 improves 29, worsens 57, and leaves 665 unchanged.

The additional configuration is class attention after block 9, keeping 96 patches, reference attention backend. It is the highest of the four validation mAP values. The Rank-1 and latency tie breaks were not used. The official test was not used. This choice is not a claim about official-test accuracy.

## Random controls on the same validation split

Colab, Tesla T4, `torch` 2.15.0+cu132. Block 6, 64 retained patches, reference backend, raw cosine. Each run uses all 751 queries and 12,185 gallery images. k-reciprocal was not run. The score files are on Drive under `visibility-gated-reid/iteration2`, and the notebook cell lists them.

| Seed | mAP | Rank-1 |
| --- | ---: | ---: |
| 11 | 78.95 | 89.48 |
| 23 | 77.58 | 89.48 |
| 37 | 79.04 | 90.41 |
| 53 | 79.04 | 89.88 |
| 71 | 81.16 | 92.41 |
| Mean | 79.15 | 90.33 |
| Sample standard deviation | 1.28 | 1.22 |

That standard deviation is variation across random patch selections. It is not uncertainty from separate training runs. Class attention at the same block and patch count, 88.30 mAP and 94.01 Rank-1, is 9.15 mAP points and 3.68 Rank-1 points above this mean. The matched random controls for block 9 with 96 patches were not run.

Raw top-five hit rate is stored separately in each JSON. It is not CMC Rank-5.

## What was not a retrieval result

- `outputs/iteration2/smoke.json`: two images. Descriptor width 1,280. After block 6 the token counts entering the blocks are six times 129 and then six times 65. Reference and CLS-row descriptors differed by 0 on those two images. Random selection was unchanged when the batch was reversed.
- A two-image `prune-embed --limit 2` run is a smoke test. One of the two queries had no valid gallery positive under the truncated gallery. Its mAP is not interpretable.

## Timing

Correction: the timings below were collected by calling the encoder without `torch.inference_mode()`, so gradient tracking was still enabled. They are preliminary. The original values and the short protocol are kept. They are not a corrected speed measurement.

Scope A is an input tensor already on the device, through the final L2-normalised descriptor, including selection. These are short local protocols, not the planned 10 warm-up / 50 measured / 3 trial protocol. The reported p90 was a percentile of trial averages, not of individual batches.

Batch 1, warm-up 3, 10 iterations, 2 trials, block 6, 64 patches:

| Method | Median batch latency | Images/s | Speedup vs full |
| --- | ---: | ---: | ---: |
| Full encoder | 12.22 ms | 81.8 | 1.00 |
| Attention, reference | 10.36 ms | 96.5 | 1.18 |
| Attention, CLS row | 11.29 ms | 88.6 | 1.08 |
| Random, seed 11 | 14.53 ms | 68.8 | 0.84 |

The two random trials were 11.50 ms and 17.56 ms, so that batch-1 random number is unstable.

Batch 32, warm-up 2, 5 iterations, 1 trial:

| Method | Median batch latency | Per image | Images/s | Speedup vs full |
| --- | ---: | ---: | ---: | ---: |
| Full encoder | 376.80 ms | 11.77 ms | 84.9 | 1.00 |
| Attention, reference | 280.28 ms | 8.76 ms | 114.2 | 1.34 |
| Attention, CLS row | 269.24 ms | 8.41 ms | 118.9 | 1.40 |
| Random, seed 11 | 270.55 ms | 8.45 ms | 118.3 | 1.39 |

CUDA peak memory was not available. After one MPS forward, current allocated memory was about 350 MB at batch 1 and 362 MB at batch 32. That is not a peak-memory measurement. No FLOP count was produced. The descriptor stays 1,280 dimensions, so cosine search over a fixed gallery does not get cheaper.

Scope B, measured once: load, preprocess, transfer, encode, and cosine against eight already computed gallery vectors took 0.124 seconds. The gallery was precomputed. This is not a Market-1501 throughput number.

## Attention backend

On a synthetic block, reference `need_weights=True` scores and the manual CLS row differed by at most 1.49e-8, and the argsort agreed. On two real images the final descriptors agreed exactly. The block residual used by the new path is the `need_weights=False` output on both backends. The reference backend still pays for a separate `need_weights=True` call.

The default remains `reference`. The short batch-1 timing did not show the CLS-row path to be faster, and the batch-32 comparison is a single trial. The historical `keep_ratio` path was not switched to this backend.

## Recovery

`tests/test_iteration2.py` builds a four-row table, saves the first two rows, resumes the rest, and matches an uninterrupted table. A mismatched identity in a verified metadata file is rejected. On real images, a third two-image embed left the verified feature-file timestamp unchanged and reproduced the same smoke scores. That smoke score is not a Market-1501 result.

## Not executed

- Random controls at block 9 with 96 patches.
- The planned 10 / 50 / 3 timing protocol, and any CUDA peak-memory figure. The T4 runs were accuracy runs, not timed benchmarks.
- k-reciprocal on every new configuration.
- Official-test embeddings for the full encoder, block 6 / 64 attention, its five random controls, block 9 / 96 attention, and its five random controls.
- Query-level success and failure figures for a new official-test ranking.

## Conclusion that the evidence supports

On the identity-seen training split, keeping 64 patches after block 6 lowers cosine mAP by 9.62 points relative to the full frozen encoder, and 456 of 751 queries get a worse average precision. The same cut with random selection is lower still: the five-seed mean is 79.15 mAP, 9.15 points below class attention, with a sample standard deviation of 1.28 points. Keeping 96 patches after block 9 lowers mAP by 0.12 points on that same split. Whether either setting holds on the official test was not measured in this iteration.

A short MPS timing at batch 32 shows the shortened sequence can be faster than the full encoder after selection work is included. The batch-1 random timing does not show a speedup, and neither protocol is the planned repeated benchmark. Half the patches were not treated as half the cost.

## Next command

Run the matched random controls for the selected configuration, block 9 and 96 patches, on the same validation manifest. One seed at a time. Do not pass `--limit`.

```bash
PYTHONPATH=src .venv/bin/python -m reidvlm prune-embed \
  --config configs/market1501.yaml \
  --split validation \
  --strategy random \
  --prune-block 9 \
  --keep 96 \
  --seed 11 \
  --backend reference \
  --batch-size 32 \
  --run-dir outputs/iteration2/val_random_b9_k96_s11 \
  --output outputs/iteration2/val_random_b9_k96_s11.json
```

Repeat with seeds 23, 37, 53, and 71. The official test is a later command. It uses `--split test` and no `--limit`.
