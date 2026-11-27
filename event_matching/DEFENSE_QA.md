# Individual defense: questions and answers

Every team member should be able to give these answers in their own words and open the file that backs each one. Line numbers refer to the current `match.py` and `encode.py`.

## Task and data
**What is the task; how are inputs and targets represented?**
Binary pair classification. Input: two articles, each as a headline and a headline-plus-lead view. Target: 1 if both articles have the same `event` ID in `dataset.json`, else 0 (`match.py:43`). The model outputs a sigmoid score; the decision is score ≥ 0.675.

**How was the dataset collected, labeled, cleaned, split?**
- **Collected:** public RSS and Telegram previews, Nov 26–27 2026. Every article keeps its `source` and `url`.
- **Labeled:** event IDs were assigned with AI assistance; independent human review is pending (`human_review/`).
- **Cleaned:** roundups, multi-event pieces, overlapping coverage and duplicate leads were removed. 305 records became 282 articles.
- **Split:** day one was split by event, with seed 73 and stratification by event size, into train and validation. Day two is the test set. Pairs are formed only inside a split (`match.py:39-41`).

**How did you prevent leakage between train/val/test?**
- We split by event before making pairs, so no incident is on both sides.
- Test comes from a later day.
- Checks: zero shared event IDs, zero shared normalised headlines or URLs, and no cross-split pair with cosine > 0.85.
- Standardisation uses train statistics only (`match.py:46`).
- Checkpoint and threshold are chosen on validation only (`match.py:61`, `match.py:64`).

**What class imbalance is there and how is it handled?**
Positives are 0.36% of train pairs, 1.84% of validation pairs and 0.25% of test pairs. We use class-balanced BCE (`match.py:49-52`): each class gets half the total weight, so a positive pair weighs about 274× a negative one. We report positive-class metrics and AP instead of accuracy.

## Tokenization and model
**What does your tokenizer do? Token IDs, padding, truncation, attention masks.**
- **Tokenizer:** SentencePiece subword tokenizer with the XLM-R vocabulary (250,002 tokens), wrapped with `<s>` … `</s>`.
- **Padding:** sequences are padded to the longest in a batch of 16; the attention mask is 1 for real tokens and 0 for padding.
- **Truncation:** at 128 tokens. Headlines are never truncated (max 62). 177 of 282 headline+lead views are truncated (median 152 tokens), see `benchmark.json`.
- `--demo` prints tokens, token IDs, token count and attention-mask length.

**Explain the architecture and its main components.**
1. Frozen `paraphrase-multilingual-MiniLM-L12-v2`: 12 Transformer layers, 117.7M parameters.
2. Masked mean pooling, so padding is ignored.
3. L2 normalisation to 384-d vectors (`encode.py:15`).
4. Four pair features (`match.py:20`): cosine of the headline+lead vectors, cosine of the headline vectors, Jaccard of lowercase headline words, Jaccard of headline numbers.
5. z-score, then a linear layer with 4 weights + 1 bias, then a sigmoid.

**Why this architecture instead of a simpler baseline?**
The baseline *is* the simpler model: one cosine and a threshold. We asked whether learning a weighted combination of four signals helps. Fine-tuning the encoder was ruled out because there are only 18 positive training pairs. Answer: on test it does not help.

**Which loss and optimizer, and why?**
- **Loss:** class-balanced binary cross-entropy, written as `logaddexp(0, z) − y·z` for numerical stability (`match.py:52`).
- **Optimizer:** full-batch gradient descent with LR 0.5 and L2 0.001 (`match.py:56-57`).
- **Why full batch:** the problem is convex and small (4950 × 4), so full batch is exact and deterministic. No seed is needed for training because initialisation is zeros and there is no sampling.

## Training and curves
**What do the learning curves show? How do you spot over/underfitting?**
- `learning-curves.pdf` / `history.json`: training loss falls monotonically. Validation loss reaches its minimum of 0.1188 at epoch 535 and then rises to 0.143 by epoch 2000. That is mild overfitting.
- v1 (LR 0.05, 200 epochs) had its minimum at the last epoch, which meant it was underfitting / not converged.
- The F1 curve is at a fixed 0.5 threshold, so it differs from the reported 0.780.

**Which checkpoint was selected, and how was the test set kept out of selection?**
- The epoch with minimum validation loss (535) is kept (`match.py:61`).
- The threshold is then picked on a grid to maximise validation F1 (`match.py:64`), giving 0.675.
- Test is scored once with those values. **Honest caveat:** the v2 LR and epochs were chosen after the v1 test score had been seen, so v1 is the clean result and v2 is reported as a post-hoc rerun.

**Training time, hardware, inference latency?**
- **Hardware:** Apple Silicon Mac, CPU, macOS 15.6.1.
- **Training:** the head trains in 2.2 s.
- **Inference:** scoring 9180 test pairs with the head takes 0.17 ms. The encoder takes 31 ms per article (both views) and runs at 148 articles/s batched.
- **Memory:** peak RSS is 1.1 GB with the encoder loaded and 126 MB for head training only.
- Sources: `results.json`, `benchmark.json`.

**How many experiments, seeds, variability?**
Two configurations (v1, v2), one deterministic run each. Variability is estimated with a paired bootstrap over test events: 3000 draws, seed 73 (`intervals.py`). Re-encoding and retraining from scratch reproduce every reported count exactly.

## Metrics and results
**Precision vs recall vs F1?**
- **Precision** = TP/(TP+FP): of the flagged pairs, how many are true matches.
- **Recall** = TP/(TP+FN): of the true matches, how many we flagged.
- **F1** = 2PR/(P+R): their harmonic mean. It is high only when both are high.
- For us, precision matters more: a false match mixes two stories in a broadcast.

**Why macro/micro/weighted F1? What does the confusion matrix reveal?**
- We report **positive-class** F1. With 99.75% negatives, the negative class has F1 ≈ 0.99 for every method. Macro F1 would average that in, and micro F1 would equal accuracy (≈ 0.99), so both would hide the differences.
- The confusion matrices (`figures.pdf`, slide 5) show the trade-off: v2 finds 20/23 matches but raises 126 false alarms; the baseline finds 11/23 with 7 false alarms.

**What are the headline results?**
On test:

| Method | F1 | AP | 95% CI for F1 |
|---|---:|---:|---|
| Baseline | 0.537 | 0.583 | 0.28–0.72 |
| v1 head | 0.250 | 0.467 | 0.12–0.39 |
| v2 head | 0.237 | 0.519 | 0.12–0.38 |

The interval for v2 − baseline is [−0.53, −0.08], so the baseline is better on this day. On validation v2 was better (F1 0.780 vs 0.190).

**Why did validation and test disagree?**
- The threshold was tuned at 1.84% prevalence and applied at 0.25%, so far more negatives cross it.
- Test leads are shorter (median 273 vs 600 characters).
- Validation has only 5 multi-article events, so the threshold estimate is noisy.
- These are hypotheses; causality has not been tested.

**What do the metrics not measure?**
Station-level cluster quality, label correctness (labels are provisional) and day-to-day variation.

## Errors
**Give a model failure and how you would investigate it.**
1121/1137 is a drone attack on St Petersburg vs a drone attack on Novorossiysk: v2 scores it 0.999, but it is a different incident. The live demo reproduces this with new text.

To investigate:
1. Look at the four feature values: both cosines are high (same topic) and there is word overlap ("drone", "attack").
2. Count how many false positives share this pattern. 96/126 are RU-RU, and a single generic article (1113) appears in 18.
3. Add hard negatives of this kind to training plus a location/named-entity mismatch feature.
4. Re-tune on a validation day with realistic prevalence.
5. Test on a new day. Do not re-tune on this test set.

**Where are training config, evaluation code and checkpoint defined?**

| What | Where |
|---|---|
| CLI defaults | `match.py`, last lines (`--lr 0.5 --epochs 2000`) |
| Training loop | `match.py:53-61` |
| Threshold selection | `match.py:64` |
| Metrics | `match.py:28` (`score`) |
| Test evaluation | `match.py:67-74` |
| Encoder and revision | `encode.py:6-7` |
| Selected weights | `checkpoint.json` |
| Curves | `history.json` |
| Results | `results.json` |
| Confidence intervals | `intervals.json` |
| Per-pair scores | `test_predictions.json` |
| Latency | `benchmark.json` |

## Likely follow-up questions
- **Why only 5 trainable parameters?** 18 positive training pairs cannot support more without overfitting. Even with 5, validation loss starts rising after epoch 535.
- **Is the score a probability?** No. Balanced loss shifts it upwards relative to the true prevalence. It is a ranking score with a decision threshold.
- **Why cosine?** Vectors are L2-normalised, so cosine = dot product. The encoder was trained so that paraphrases and translations have high cosine.
- **Why is the pairwise classifier non-transitive?** It scores pairs independently. 1001–1089 matches and 1001–1017 matches, but 1017–1089 does not. The station's grouping resolves this through centroids.
- **Who labeled the data?** An AI assistant, under a written rubric. This is the main limitation; the two-person review with Cohen's kappa is prepared in `human_review/` but not yet done. Any kappa value is "not measured".
