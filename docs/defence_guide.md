# Defence guide

Use this with `paper/slides.pdf`. The slides are the file to open while rehearsing.

## 60–90 second summary

This project studies token pruning inside a frozen CLIP-ReID vision transformer. Person re-identification is the test: one crop is used to rank other photos of the same person. The encoder is not retrained. After a chosen block, the class token keeps a fixed number of patches and the later blocks run on the shorter sequence. Two cuts are measured. After block 6, keeping 64 patches drops official-test mAP from 89.41 to 74.86. After block 9, keeping 96 patches stays close, at 89.27 mAP and 95.13 Rank-1. At both cuts, class attention scores above a five-seed random draw of the same number of patches. A Tesla T4 timing run at the first cut does not establish a speedup, because the full encoder’s own batch-32 medians already move from 241.5 ms to 313.7 ms. The contribution is that measurement, not a new pruning algorithm.

## Bangla

এই প্রজেক্ট frozen CLIP-ReID মডেলের ভেতরে token pruning মাপে। ছবি 128টা প্যাচে ভাগ হয়, সাথে একটা class token থাকে। মাঝপথে কিছু প্যাচ ফেলে দিলে পরের স্তরগুলো ছোট তালিকা নিয়ে চলে। মানুষ খোঁজা হলো পরীক্ষার কাজ: একটা ছবি দিয়ে অন্য ক্যামেরার ছবিতে সেই মানুষকে সামনে আনা। Block 6-এর পর 64টা প্যাচ রাখলে official test-এ mAP অনেক কমে। Block 9-এর পর 96টা প্যাচ রাখলে স্কোর পুরো মডেলের কাছে থাকে। একই সংখ্যক প্যাচ এলোমেলো রাখলে attention-এর চেয়ে স্কোর কম। সময় মাপা হয়েছে, কিন্তু দ্রুত হয়েছে এটা প্রমাণিত হয়নি।

## Slide by slide

1. Title. Say the subject is token pruning and re-identification is the score.
2. Question. Three choices: which patches, after which block, and whether the shorter run is faster on the clock.
3. Token. A patch is one cell of the 16 by 8 grid. The class token is the extra token used to build the search vector.
4. Flow figure. Point at 129, then 65, then 97. Say the class token is inside those counts and the cut is after a finished block.
5. Attention versus random. Say the two cuts change depth and count together, so they do not isolate one effect. Mention that the validation split which fixed block 9 / 96 is identity-seen, and do not quote its 97 percent scores.
6. Protocol. Define Rank-1 and mAP in the words on the slide. Give the 0.52 percentage-point Rank-1 gap.
7. Official-test figure. Read the bars left to right. Say the error bars belong only to the random means.
8. Numbers. 14.55 percentage points of mAP at block 6 / 64. Block 9 / 96 is close and is not claimed as a better operating point.
9. Photograph. One real query. Dark blocks are dropped patches. Tokens have already mixed, so a dark block is not “only background.”
10. Timing. Show the three full-encoder points at batch 32. Say block 9 / 96 was not timed.
11. Contribution. Measurement of an existing selection rule. Not a new algorithm and not a complete EViT reproduction.
12. References. If asked for the code, the measurements come from revision `d143dbd`. The checked paper record is `24e36ac`. The full-validation stdout was saved in `c2a5246` and was not rerun.

## Short definitions

- Token: one vector in the transformer sequence. Here, a patch token is one 16 by 16 cell of the resized crop.
- CLS: the class token. It is never dropped. The search vector is built from it.
- Pruning: after a complete block, dropping some patch tokens so the later blocks see a shorter sequence.
- Frozen encoder: the checkpoint weights and BatchNorm statistics stay as released. Nothing is trained for this study.
- Gallery: the set of photographs ranked against the query, after junk removal and the same-camera rule.
- mAP: the average quality of the whole list of true matches.
- Rank-1: the fraction of queries whose first allowed gallery photo is the right person.
- Sample SD: the standard deviation across the five random-selection seeds on one fixed split, with divisor n−1. It is not a confidence interval and not training uncertainty.

## Likely questions

**What did you implement, and what did you take from other work?**
The checkpoint and its re-identification training come from CLIP-ReID. The idea of scoring patches by class-token attention comes from EViT. What this project adds is the inference-time cut inside that frozen encoder, the matched random control, the official-test comparison, and the corrected timing measurement. EViT’s training and token fusion are not included.

**Why is random selection useful?**
It keeps the same block and the same number of patches, and removes only the attention rule. The recorded gap is then the difference between that rule and chance on this fixed split.

**Why did attention score higher than random?**
That is what the five-seed means show, at both cuts. A possible reason is that the attention scores use the model’s own class token, while the random draw does not. The records do not identify which individual tokens caused the gap.

**Why does a high validation score not prove performance on unseen people?**
The validation images and identities come from the training split. The encoder was trained on them. The official test uses 750 different people.

**Why is the small Rank-1 increase at block 9 / 96 not a general improvement?**
On the official test, mAP is 0.14 percentage points lower and Rank-1 is 0.15 percentage points higher than the full encoder. The setting was fixed before that test. Those two differences do not establish a better accuracy–cost operating point.

**Why does a shorter sequence not prove faster inference?**
Latency was measured only for block 6 / 64. At batch 32 the full encoder’s own trial medians are 241.5, 249.6, and 313.7 ms. That spread is larger than a claim of a stable speedup. Block 9 / 96 has no timing record.

**Why do the memory numbers include verification copies?**
The benchmark keeps GPU-resident copies of the parameters and buffers so it can check that the weights did not change. The recorded peak is the allocation in that harness, not an inference-only memory requirement.

**What is the contribution?**
A controlled measurement of two inference-time cuts, with a matched random control, on a frozen CLIP-ReID encoder and the official Market-1501 test.

**What is outside the evidence?**
No new dataset, no retraining, no learned pruner, no token fusion, no k-reciprocal on the new runs, no timing for block 9 / 96, and no claim that the one illustrated query represents the whole dataset.

**Does a dark patch in the figure mean the model removed background?**
No. By block 6 the class token and the patch tokens have already exchanged information. The figure shows which positions were dropped for one image.
