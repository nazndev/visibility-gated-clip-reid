# Derived summary of recorded results

These files are summaries of artifacts that already exist. They are not new experiment outputs, and they are not reconstructions of score files that are not in this repository.

Sources:

- Notebook stdout in `notebooks/colab_iteration2_v2.ipynb`, including the full-encoder validation stdout saved in commit `c2a5246` and displayed again in the current notebook. Inference was not rerun for that row.
- Historical official-test JSON files in `results/market1501_test_20261115T182047Z.json`, `results/market1501_test_20261115T183006Z.json`, and `results/market1501_test_20261115T122953Z.json`.

`official_test.csv` keeps three groups apart: `historical_official_test`, `historical_official_test_mps`, and `corrected_official_test_cosine`. k-reciprocal rows are marked `retrieval=kreciprocal`. The corrected notebook runs did not compute k-reciprocal.

`validation.csv` is the training-derived split. `identity_seen` is true on those rows. The historical JSON files do not contain an `identity_seen` field, so that column is `unknown` there.

`timing_trials.csv` is the corrected Tesla T4 benchmark at block 6 with 64 patches. Block 9 with 96 patches was not timed. Peak CUDA bytes are the benchmark-harness allocation, including the GPU-resident copies kept for the state check.

`shortlist_hit_rate` appears only on the historical files. It is a raw top-five rate. It is not CMC Rank-5.

Sample standard deviations in `summary.json` use divisor n−1 and the unrounded scores. Display rounding was checked against:

- Official full encoder: 89.41 mAP / 94.98 Rank-1.
- Attention, block 6 / 64: 74.86 / 88.93.
- Attention, block 9 / 96: 89.27 / 95.13.
- Random, block 6 / 64: 67.63 ± 0.33 mAP; 85.89 ± 0.30 Rank-1.
- Random, block 9 / 96: 85.97 ± 0.12 mAP; 94.07 ± 0.22 Rank-1.
- Validation random, block 9 / 96: 96.25 ± 0.37 mAP; 97.47 ± 0.38 Rank-1.

The Tesla T4 name for the restored full-validation row comes from the install cell in commit `c2a5246`. The score record itself has no GPU field.
