# Visibility and consistency gated person re-identification

Nazmul Nazim

One photo of a person is taken by one camera. The task is to rank photos from the other cameras so the same person comes first.

Market-1501 is the dataset, published by Zheng et al. in 2015. CLIP-ReID is the search model, trained on that dataset by Li, Sun, and Li in 2023. This project loads their ViT-B/16 checkpoint and does not train it again. On the standard test, 3,368 query photos against 15,913 gallery photos, that model scored 89.41% mAP and 94.98% Rank-1. The paper reports 89.6% and 95.5%. k-reciprocal re-ranking on the same features scored 94.48% mAP and 95.37% Rank-1. The result file is `results/market1501_test_20261115T122953Z.json`.

Qwen2.5-VL-3B is the judge. For each query it looks at the top 5 gallery photos. It compares upper clothing, lower clothing, footwear, a carried object, and a distinctive mark, then sees the same pair with the photos swapped. The gate keeps the answer only when both orders agree and at least two of those cues stay visible. Otherwise the CLIP-ReID order stays. The full judge run has not been done. `scripts/06_qwen_one_query.py` judges one query so the gate can be seen. That run is not a Market-1501 score.

## Mac

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/01_download_dataset.py
.venv/bin/python scripts/02_download_model.py
.venv/bin/python scripts/03_check.py
.venv/bin/python scripts/04_download_qwen.py
.venv/bin/python scripts/05_visual_rank.py
```

`./scripts/run_mac.sh` runs those five steps. Step 5 ranks every query with CLIP-ReID and does not call Qwen. The dataset Drive id is `0B8-rUzbwVRk0c054eEozWG9COHM`. The checkpoint Drive id is `1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7`. Qwen is saved under `caches/huggingface`.

## Colab

Use a GPU runtime. Open `notebooks/colab_first_iteration.ipynb` and run the cells from the top. The first cell clones this repository. The later cells download Market-1501 and the CLIP-ReID checkpoint onto the Colab disk, then rank every query. The last cell asks Qwen to judge one query. That is not a Market-1501 score. The Colab disk is wiped when the session ends, so the next session downloads the two files again.

## Files

`src/reidvlm/` has the data loader, the evaluator, the CLIP-ReID model, the gate, and the Qwen client. `configs/market1501.yaml` holds the Market-1501 settings. `requirements.txt` is for the Mac. `requirements-colab.txt` is for Linux. The dataset, the checkpoint, the Qwen cache, and `.venv` are not committed.

## Sources

CLIP-ReID inference follows Syliz517/CLIP-ReID at commit `eb1898b72c882875f478bebfc6d41644eece0a5d`. That repository has no license file. The ViT block follows OpenAI CLIP, which is MIT licensed. The re-identification changes are noted in `src/reidvlm/encoders/clip_reid.py`. k-reciprocal follows Zhong et al., CVPR 2017, through the CLIP-ReID Python code.

Qwen weights are `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, under the Qwen research license. The prompt and the gate are in this project.
