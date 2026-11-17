# Inference-time token pruning for frozen CLIP-ReID

Nazmul Hasan Nazim

## Start here

Token pruning is the subject. Person re-identification is the task used to measure it.

Read in this order:

1. [paper/main.pdf](paper/main.pdf) — the course paper.
2. [paper/slides.pdf](paper/slides.pdf) — the 12-slide explanation. Rehearse from [docs/defence_guide.md](docs/defence_guide.md).
3. [results/paper_summary/README.md](results/paper_summary/README.md) — tables derived from the recorded outputs.
4. [notebooks/colab_iteration2_v2.ipynb](notebooks/colab_iteration2_v2.ipynb) — the notebook that printed the Tesla T4 scores. The measurements were produced by code revision `d143dbdbd690f70315cb706403980eddf09279e9`.

This repository measures inference-time token pruning in a frozen CLIP-ReID ViT-B/16 encoder. Market-1501 person re-identification is the evaluation task. The current study asks how inference-time patch selection, pruning depth, and the number of retained tokens change retrieval accuracy and measured inference cost.

A 256×128 crop is 128 patches plus one class token. Selection, when enabled, runs after a complete transformer block. The class token is kept, the chosen patches are gathered into a shorter sequence in their original spatial order, and the remaining blocks run on that sequence. Positional embeddings are added once, before the transformer. There is no token fusion and no extra trained scorer.

Two selectors are compared at the same block and the same retained-patch count. Class-attention selection ranks patches by the class token's attention, averaged across heads after the softmax. Random selection draws a uniform subset without replacement. The random subset of an image is determined by the experiment seed and a hash of that image's preprocessed pixels, so it does not depend on batch size or processing order. The seeds used for that control are 11, 23, 37, 53, and 71.

The encoder is the public CLIP-ReID Market-1501 checkpoint. Parameters stay frozen, BatchNorm uses its stored statistics, and the descriptor is the 1,280-dimensional L2-normalised concatenation of the batch-normalised class token and its batch-normalised projection. The primary score is cosine retrieval. k-reciprocal re-ranking is a separate comparison and is only interpreted against the same re-ranking on the other method.

The previously reported full-test numbers, measured with the earlier pruning path, are stored in `results/market1501_test_20261115T182047Z.json` and `results/market1501_test_20261115T183006Z.json`. They are historical measurements. This iteration does not overwrite them. The experiment plan for the new grid is `configs/iteration2_plan.json`.

## Commands

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest tests.test_iteration2
.venv/bin/python -m reidvlm prune-plan
.venv/bin/python -m reidvlm prune-manifest --config configs/market1501.yaml
.venv/bin/python -m reidvlm prune-benchmark --config configs/market1501.yaml --batch-size 1 --warmup 10 --iterations 50 --trials 3
.venv/bin/python -m reidvlm prune-embed --config configs/market1501.yaml --split validation --strategy cls_attention --prune-block 6 --keep 64 --backend reference
```

`prune-embed --limit N` is a smoke test. It must not be used to choose the configuration that is later scored on the official test. The official test has already been inspected, so it is not an untouched holdout.

## Historical material

`notebooks/colab_first_iteration.ipynb` records the first measurement, including a Qwen2.5-VL judgment experiment that was not finished and is not part of the current study. `notebooks/colab_iteration2.ipynb` keeps the earlier iteration. The current recorded notebook is `notebooks/colab_iteration2_v2.ipynb`. `paper/main.pdf` is the course paper for the corrected measurements.

## Sources

CLIP-ReID inference follows Syliz517/CLIP-ReID at commit `eb1898b72c882875f478bebfc6d41644eece0a5d`. That repository has no license file. The ViT block follows OpenAI CLIP, which is MIT licensed. Class-attention token selection follows EViT (Liang et al., ICLR 2022, arXiv:2202.07800). This code does not train the pruner, does not fuse discarded tokens, and does not claim to reproduce EViT. Attention weights follow `torch.nn.MultiheadAttention`. k-reciprocal follows Zhong et al., CVPR 2017. Market-1501 follows Zheng et al., ICCV 2015.
