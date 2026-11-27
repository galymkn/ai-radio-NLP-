# Multilingual news event matching for NewsRadio

Task: given two news articles (English or Russian), predict whether they report the **same specific incident or decision**. Input: headline plus lead of each article. Output: a same-event score in [0, 1] and a binary decision at a validation-selected threshold. Users: the NewsRadio station, which groups incoming articles into one story per event before writing a broadcast script. A missed match repeats a story; a false match can mix unrelated facts in one broadcast. Success criterion: positive-class F1 on a held-out later day, compared with the station's fixed cosine rule.

The full project report is in [REPORT.md](REPORT.md). Slides: `Defense.pdf`. Figures (confusion matrices, precision-recall curves, pipeline diagram): `figures.pdf`. Preparation for individual questions: [DEFENSE_QA.md](DEFENSE_QA.md). Archived first run: `v1/`.

## Pipeline

```
Raw article (RSS / Telegram)
  └─> headline, lead[:600]                         (dataset.json)
       ├─> view A: headline
       └─> view B: "headline. lead"
            └─> MiniLM tokenizer: token IDs, attention mask, truncation at 128, padding per batch
                 └─> frozen multilingual MiniLM-L12 (forward pass) → masked mean pooling → L2-normalised 384-d vector
                      └─> pair features (match.py):
                            lead cosine, headline cosine, headline word Jaccard, headline digit Jaccard
                           └─> standardise with TRAIN mean/std → linear layer (4 weights + bias) → sigmoid
                                └─> balanced BCE loss → full-batch gradient descent (train)
                                └─> checkpoint + threshold chosen on VALIDATION
                                └─> single scoring of TEST → results.json, test_predictions.json
```

## Reproduce
> **Public repository copy:** article texts are removed; see [DATA_ACCESS.md](DATA_ACCESS.md) for what still runs and how to get the full dataset.

Python 3.12, macOS arm64 (the reported run). From the radio project root on the demo Mac:
```sh
source .venv/bin/activate
pip install -r event_matching/requirements.txt
python event_matching/encode.py          # optional: re-encode; needs the pinned MiniLM revision (local cache or Hugging Face)
python event_matching/match.py --lr 0.5 --epochs 2000
python event_matching/intervals.py
python event_matching/benchmark.py       # token-length statistics, encoder latency, throughput, memory -> benchmark.json
python event_matching/plot_curves.py     # writes learning-curves.pdf, figures.pdf and Defense.pdf
```
`match.py` uses the saved `embeddings.npz` and overwrites `checkpoint.json`, `history.json`, `results.json` and `test_predictions.json` deterministically (zero initialisation, full batch, no sampling) except for timing and memory fields.

**Reproduction check (Nov 27 2026, clean virtual environment with the pinned `requirements.txt`).**
1. Retraining from the shipped embeddings reproduced `checkpoint.json`, `history.json`, `results.json`, `test_predictions.json` and `intervals.json` byte-for-byte, apart from timing and memory fields.
2. Re-encoding all 282 articles from the pinned Hugging Face revision gave embeddings within 1.2e-7 of the shipped ones.
3. Retraining on those re-encoded embeddings reproduced every confusion matrix, AP value, the selected epoch (535) and the threshold (0.675). Test scores changed by at most 7e-8, and no decision flipped. The live radio is a separate application (`python -m newsradio run`) and is **not included in this package**.

### Demo / inference on new text
```sh
HF_HUB_OFFLINE=1 python event_matching/match.py --demo "Town opens new library" "В городе открылась новая библиотека" \
    --leads "The city library opened on Monday after renovation." "Библиотека открылась в понедельник после ремонта."
```
Prints the two input views, tokens, token IDs, token count and attention-mask length, the four features, the classifier score, threshold and decision, and the baseline cosine decision. `--leads` is optional, but the model was trained on headline **plus lead**. Without leads the "lead cosine" feature is computed from the headline only, so the score is out of distribution; the script prints a warning in that case. The score is a weighted classifier output, not a calibrated probability. `demo_example.txt` records three real runs: EN/RU library headlines without and with leads (both methods say same event), and a hard negative (two drone attacks in different cities) where v2 wrongly says same event (0.99999) and the baseline correctly says different (cosine 0.691). The demo inputs were written for the demonstration and are not from the dataset.

## Dataset card
- **Source.** Public RSS feeds and Telegram web previews, collected Nov 26–27 2026. Source and URL are stored per article in `dataset.json` (fields `id, title, lead, url, source, event, split`; the public repository copy keeps only `id, url, source, event, split`, see DATA_ACCESS.md). Publisher rights apply. Keep this classroom package private unless redistribution rights for the excerpts have been checked.
- **Languages.** English and Russian (detected by Cyrillic characters in the headline): train 40 EN / 60 RU, validation 22 / 24, test 60 / 76.
- **Labels.** Event ID = one specific reported incident or decision. Follow-ups describing the same incident can match. Different speeches on the same policy are different events. Labels are **assistant-assigned and pending independent human review** (see `human_review/`). Borderline cases can change the results.
- **Cleaning.** 149 first-day articles plus 156 later records with URLs absent from the first snapshot. Identified roundups, ambiguous multi-event pieces, overlapping earlier coverage and duplicate leads were excluded, leaving 282 documents. Within the matcher no further normalisation is applied: case, names, numbers and word order are kept; no stemming or stop-word removal. Leads are cut to 600 characters.
- **Split.** Day one was split by event (seed 73, stratified by whether an event has multiple articles) into train and validation; day two is the held-out test. All unordered pairs within a split are used.
- **Leakage checks.** No event ID occurs in more than one split; no exact duplicate normalised headlines or URLs across splits; no cross-split pair has headline-plus-lead cosine above 0.85. Related names and topics do recur across days.
- **Known shift.** Median lead length is 600 characters in train and validation but 273 in test (shorter next-day RSS summaries). Source names are written with `@` on day one (`@bbcrussian`) and without it on day two (`bbcrussian`); this field is not used as a feature.

| Split | Articles | EN / RU | Events | Multi-article events | Pairs | Same-event pairs (prevalence) | EN-EN / EN-RU / RU-RU positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 100 | 40 / 60 | 85 | 12 | 4950 | 18 (0.36%) | 4 / 5 / 9 |
| Validation | 46 | 22 / 24 | 36 | 5 | 1035 | 19 (1.84%) | 7 / 5 / 7 |
| Test | 136 | 60 / 76 | 120 | 12 | 9180 | 23 (0.25%) | 4 / 9 / 10 |

Pairs that share an article are statistically dependent; validation positives come from only five multi-article events.

## Model
- **Encoder.** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision pinned in `encode.py`. Tokenizer: SentencePiece subword tokenizer (XLM-R vocabulary, about 250k tokens) → token IDs with attention masks; max 128 tokens per view (truncation), dynamic padding within a batch of 16. Measured token lengths (`benchmark.json`): headlines median 22, max 62, never truncated; headline+lead views median 152, p95 191, and **177 of 282 exceed 128 tokens and lose the end of the lead**. The 600-character cut therefore rarely binds; the 128-token limit does. 117,653,760 parameters, **all frozen**. Masked mean pooling, L2 normalisation → 384-d vectors.
- **Pair features.** cosine(headline+lead views), cosine(headline views), Jaccard of lowercase headline word sets, Jaccard of digit-only headline tokens (empty union → 0).
- **Head.** Standardisation with train mean/std, linear layer with 4 weights + 1 bias (**5 trainable parameters**), sigmoid output.
- **Why this design.** Only 18 positive training pairs exist, so fine-tuning the encoder or a larger head would overfit. A frozen multilingual encoder gives cross-lingual similarity; the linear head only learns how to combine it with lexical overlap. The baseline is the rule the station already uses.

## Training configuration
| Setting | Value |
|---|---|
| Loss | Class-balanced binary cross-entropy (each class gets half the total weight) |
| Optimiser | Full-batch gradient descent (batch = all 4950 train pairs), zero initialisation |
| Learning rate / schedule | 0.5, constant (v1: 0.05) |
| Regularisation | L2 0.001 on weights (not bias) |
| Epochs | 2000 (v1: 200) |
| Checkpoint rule | Epoch with minimum balanced validation loss → epoch 535, val loss 0.11884 |
| Threshold rule | Grid 0.05–0.975 step 0.025; maximise positive-class validation F1, ties closest to 0.5 → 0.675 (validation F1 0.780) |
| Seeds | Split seed 73; bootstrap seed 73; training itself is deterministic. One run per configuration, so run-to-run variance is 0 by construction and not informative. |
| Hardware / time | Apple Silicon Mac, macOS 15.6.1 arm64, CPU. Head training incl. validation loop 2.22 s; scoring 9180 test pairs with the head 0.17 ms; peak process RSS 126 MB |
| Encoder (`benchmark.json`) | CPU, 8 threads: 31 ms median per article for both views (p95 33 ms), 148 articles/s batched, peak process RSS 1.1 GB with the encoder loaded |
| Not measured | End-to-end station throughput; GPU latency |

`history.json` stores per epoch: train and validation data loss (without the L2 term) and validation F1 at a fixed 0.5 threshold. The plotted F1 curve therefore differs from the 0.780 obtained at the selected threshold. Training loss keeps falling after epoch 535 while validation loss rises (0.119 → 0.143 at epoch 2000): mild overfitting, which is why the checkpoint is taken at the minimum.

v1's validation loss was still falling at its last epoch (200), i.e. under-converged. That observation, made after the v1 review, motivated the longer v2 run. It did not by itself diagnose insufficient model capacity.

## Metrics
All headline numbers are on the **test split** (day two, 9180 pairs, 23 positive). Implementation: `sklearn.metrics` in `match.py`.
- **Precision (positive class)** = TP / (TP + FP): share of flagged pairs that are true matches.
- **Recall (positive class)** = TP / (TP + FN): share of true matches that were flagged.
- **F1 (positive class)** = harmonic mean of the two. We report the positive class, not macro/micro averages: with 0.25% prevalence the negative class has F1 ≈ 0.99 for every method, so macro or micro averages would hide the differences. Both classes are in `results.json`.
- **AP (average precision)** summarises the precision-recall curve over all thresholds, i.e. ranking quality independent of the chosen threshold.
- **Confusion matrix** in sklearn order `[[TN, FP], [FN, TP]]`.
- **95% interval.** Paired bootstrap over test events (3000 draws, seed 73), see below.

What these metrics do **not** measure: end-to-end cluster quality in the station (which groups incrementally against centroids), label noise (labels are provisional), and day-to-day variation (one test day).

## Results
The same next-day test set was scored for v1 and again for v2; both results are disclosed. v2's learning rate and training budget were chosen after v1 had already been scored on test, so **v1 is the clean single-shot result and v2 is a post-hoc rerun**. Both thresholds and checkpoint epochs were selected on validation only. A third collection day is needed for independent confirmation.

| Method (test) | TP | FP | FN | TN | Precision | Recall | F1 | 95% F1 interval | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Baseline: cosine ≥ 0.72 | 11 | 7 | 12 | 9150 | .611 | .478 | **.537** | .276–.718 | **.583** |
| v1 head (LR 0.05, 200 ep, t = 0.825) | 18 | 103 | 5 | 9054 | .149 | .783 | .250 | .123–.392 | .467 |
| v2 head (LR 0.5, 2000 ep, t = 0.675) | 20 | 126 | 3 | 9031 | .137 | .870 | .237 | .115–.383 | .519 |

| Method (validation) | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| Baseline | 1.000 | .105 | .190 | .915 |
| v2 head | .727 | .842 | .780 | .863 |

**Interpretation.** On test the fixed cosine rule has higher F1 and AP. The learned head finds 9 more true matches than the baseline but at the cost of 119 extra false positives. The percentile interval for v2 minus baseline F1 is [−0.527, −0.075], conditional on this day's provisional labels; it does not cover annotation uncertainty or day-to-day variation. **The experiment gives no evidence for replacing the station's rule.** Likely contributors (not established causally): the threshold was tuned where prevalence was 1.84% but applied where it is 0.25%; test leads are less than half as long; only 18 positive training pairs; balanced loss weights each positive training pair about 274× a negative one (137.5 vs 0.50), pushing the head towards broad matching.

### Class-wise by language pair (test)
| Language pair | Positives | Baseline TP / FP | v2 TP / FP |
|---|---:|---:|---:|
| EN-EN | 4 | 1 / 0 | 4 / 19 |
| EN-RU | 9 | 4 / 1 | 6 / 11 |
| RU-RU | 10 | 6 / 6 | 10 / 96 |

96 of v2's 126 false positives are Russian-Russian pairs.

## Error analysis (test, v2 unless stated)
Examples are paraphrased; URLs in `dataset.json`, full text in the private dataset (DATA_ACCESS.md), all scores in `test_predictions.json`.

**Correct matches**
- 1001 / 1017: two English headlines about the same BBC interview with Zelensky. Cosine .888, v2 1.000. Both methods match.
- 1001 / 1083: same BBC interview, English vs Russian quote headline. Cosine .736, v2 .916. Cross-lingual match found by both.
- 1002 / 1016: two differently worded reports on the US Thanksgiving storms (power cuts vs storm damage). Cosine .575 (baseline misses), v2 .732. The head helps with paraphrased coverage.

**False negatives**
- 1017 / 1089: English headline vs long Russian summary of the same interview. Cosine .563, v2 .438. Note 1001 / 1089 *is* matched: a pair-level classifier is not transitive, so the station's grouping step would still join these through 1001.
- 1041 / 1149: an English explainer on Georgian police clashing with Turkish fans vs a Russian report of the Batumi match being stopped. Cosine .506, v2 .292. Different framing, few shared names.

**False positives**
- 1121 / 1137: drone attack injures a woman in St Petersburg vs drone attack damages flats in Novorossiysk. v2 .999. Same *type* of event, different incident.
- 1128 / 1156: Russian Federation Council approves the 2027 budget vs German Bundestag passes the 2027 budget. v2 .994. Shared numbers ("2027") and the digit-overlap feature push the score up.
- 1080 / 1081 / 1131: three different arrests (Kazakhstan KNB cases, Dagestan officials). Cosine .73–.79, so the baseline also fails here.
- 1017 / 1092: Zelensky interview vs Trump's later response. Cosine .727, v2 .908. Under the rubric a response is a separate event; both methods match it. Human reviewers may disagree.
- Article 1113 (Peskov on Putin–Pashinyan talks) alone appears in 18 false positives: generic Kremlin-politics vocabulary makes it look similar to many articles.

**Dominant failure pattern.** "Same event type, different incident" (attacks, budgets, arrests, weekly war summaries). How to investigate: add hard negatives of this kind to training, add entity/location overlap features, re-tune the threshold on a validation day with realistic prevalence, then confirm on a new day. Do not change the threshold based on these test examples.

## Human review
Pending. `human_review/README.txt` describes two independent reviewers covering all 136 test documents with 40 shared documents (780 shared pairs) for Cohen's kappa, plus a separate review of 49 broadcast-script sentences (a different task). Run `python human_review/agreement.py` after both forms are complete. Re-score on reconciled labels separately and keep today's results unchanged. Human values are reported as **not measured** until then.

## Where things are (individual defense)
| Question | File |
|---|---|
| Data provenance, labels, splits | `dataset.json` |
| Preprocessing, tokenizer, encoder revision | `encode.py` |
| Features, loss, gradient update, checkpoint and threshold rules, evaluation | `match.py` (`features`, `run`) |
| Selected weights, threshold, epoch | `checkpoint.json` |
| Learning curves | `history.json`, `learning-curves.pdf` |
| Confusion matrices, PR curves, pipeline diagram | `figures.pdf` (built by `plot_curves.py`) |
| Encoder latency, token lengths | `benchmark.py`, `benchmark.json` |
| Demo transcript | `demo_example.txt` |
| Prepared answers | `DEFENSE_QA.md` |
| Test metrics, validation metrics, timing | `results.json` |
| Confidence intervals | `intervals.py`, `intervals.json` |
| Per-pair test scores | `test_predictions.json` |
| Original run | `v1/` |

The earlier observation of one insufficiently supported sentence in 49 was an AI-assisted review of eight radio scripts, a different task and sample. The pair classifier changes neither the script generator nor voice synthesis.
