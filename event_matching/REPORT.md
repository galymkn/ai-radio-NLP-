# Project report: multilingual same-event matching for NewsRadio

Technical details, tables and file locations are in [README.md](README.md). This report gives the reasoning behind them.

## 1. Problem statement and objectives
NewsRadio collects English and Russian news from RSS and Telegram, groups articles that describe the same event, writes a verified script with Claude and reads it out with speech synthesis. The grouping step decides what goes into each story. If it misses a match, the station repeats a story. If it matches wrongly, one broadcast can mix facts from two different incidents, which is the more harmful error.

**NLP task.** Binary pair classification (a form of cross-lingual semantic matching). Input: two articles, each given as headline plus lead. Target: 1 if both report the same specific incident or decision, else 0. Output: a score and a decision.

**Objective.** Test whether a small supervised head on top of a frozen multilingual sentence encoder beats the station's existing rule (cosine ≥ 0.72) at the pair level on a later, unseen day, measured by positive-class F1.

**Intended use.** An internal component of a classroom news-radio prototype. It is not meant for fact-checking or for publication decisions without a human in the loop.

## 2. Related work and rationale
- **Event detection and news clustering** (Topic Detection and Tracking) traditionally compares documents with TF-IDF or embedding similarity and a threshold. The station's baseline belongs to this family.
- **Sentence-BERT-style bi-encoders** (Reimers and Gurevych, 2019) produce fixed vectors whose cosine reflects semantic similarity. The multilingual paraphrase variant was trained by knowledge distillation so that translations map to nearby vectors. That makes English-Russian comparison possible without translation.
- **Cross-encoders or fine-tuned encoders** usually match better but need far more labelled pairs than our 18 positive training pairs.

We therefore keep the encoder frozen and learn only how to weigh four similarity signals (5 parameters). The baseline is the rule already used in production, so the comparison answers the practical question: should the rule be replaced?

## 3. Data
Data comes from public RSS feeds and Telegram web previews over two days (Nov 26–27 2026). After removing roundups, multi-event pieces and duplicates, 282 articles remain: 122 English and 160 Russian. Labels (event IDs) were assigned with AI assistance and are pending independent review by two people; the protocol is in `human_review/README.txt`. Licensing: the excerpts belong to their publishers and are kept for private classroom use only.

**Split.** Day one is split by event into train (100 articles) and validation (46). All of day two is the test set (136). Splitting by event, not by pair, prevents the same incident from appearing on both sides. Using a later day as test simulates deployment. We checked that no event ID, normalised headline or URL is shared across splits and that no cross-split pair has cosine > 0.85.

**Class imbalance.** Positives are rare: 0.36% of train pairs, 1.84% of validation pairs and 0.25% of test pairs.

## 4. Preprocessing and tokenization
Leads are truncated to 600 characters. Case, numbers, names and word order are kept, because they identify specific incidents; no stemming or stop-word removal is done. Each article has two views: the headline alone, and "headline. lead". The model's SentencePiece tokenizer converts each view to token IDs with an attention mask. Views are truncated at 128 tokens and padded to the longest view in a batch of 16. Headlines never reach the limit (max 62 tokens), but 177 of 282 headline+lead views exceed it. In practice the 128-token limit, not the 600-character cut, decides how much of the lead the encoder sees. For the lexical features, headlines are lower-cased (casefold) and split into `\w+` tokens.

## 5. Architecture and training
1. **Encoder.** Frozen `paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 117.7M parameters, pinned revision), followed by masked mean pooling and L2 normalisation, giving 384-d vectors.
2. **Pair features.** Cosine of the headline+lead views, cosine of the headline views, word Jaccard and digit Jaccard.
3. **Head.** Features are z-scored with train statistics, then passed through a linear layer and a sigmoid.
4. **Training.**
   - **Loss.** Class-balanced binary cross-entropy, so that 18 positive pairs are not drowned by 4932 negatives.
   - **Optimisation.** Full-batch gradient descent with LR 0.5, L2 0.001 and 2000 epochs.
   - **Model selection.** The checkpoint is the epoch with minimum validation loss (epoch 535); the threshold maximises validation F1 (0.675).
5. **Diagnostics.** In v1 (LR 0.05, 200 epochs) validation loss was still falling at the end, so the model was under-converged. In v2, training loss keeps falling after epoch 535 while validation loss rises from 0.119 to 0.143. That is mild overfitting, and checkpoint selection handles it.

**Hardware and cost.** Apple Silicon Mac, CPU only. The head trains in 2.2 s with a peak process memory of 126 MB. The encoder needs 31 ms per article for both views (148 articles/s batched) and 1.1 GB of RSS (`benchmark.json`).

**Reproducibility.** In a clean environment with the pinned requirements, retraining reproduces all outputs byte-for-byte except timing. Re-encoding from Hugging Face gives embeddings within 1.2e-7 of the shipped ones and identical metrics.

## 6. Results
| Test (9180 pairs, 23 positive) | Precision | Recall | F1 [95% CI] | AP |
|---|---:|---:|---:|---:|
| Baseline cosine ≥ 0.72 | .611 | .478 | **.537** [.276, .718] | **.583** |
| v1 head | .149 | .783 | .250 [.123, .392] | .467 |
| v2 head | .137 | .870 | .237 [.115, .383] | .519 |

On validation, v2 had F1 0.780 against 0.190 for the baseline. On test the order reversed. The bootstrap interval for the difference in F1 (v2 minus baseline) is [−0.53, −0.08]. The test set was scored twice (v1, then v2 after changing LR and epochs), and both are reported. v1 is the clean single-shot result.

## 7. Error analysis
The head raises recall: it finds paraphrased and cross-lingual coverage the baseline misses, such as the US Thanksgiving storms pair. But it fires on many pairs describing the **same type of event in a different place**, such as drone attacks on two cities, budgets of two parliaments, or different arrests. 96 of its 126 false positives are Russian-Russian pairs, and one generic Kremlin-politics article causes 18 of them. Several false negatives pair a short English headline with a long Russian summary. Concrete pair IDs are listed in the README. The live demo reproduces the dominant failure on new text: two drone attacks in different cities score 0.99999 (`demo_example.txt`).

Probable causes, listed but not proven:
- The threshold was chosen at 1.84% prevalence and applied at 0.25%.
- Test leads are shorter: median 273 characters against 600.
- There are only 18 positive training pairs.
- The balanced loss pushes the head towards generous matching.

## 8. Limitations
- **Data size.** Two days of data and one test day; few positive events, especially in validation (5 multi-article events).
- **Labels.** They are provisional and AI-assisted; inter-annotator agreement is not yet measured.
- **Evaluation level.** We evaluate pairs, not the station's incremental centroid grouping.
- **Test reuse.** The test set was reused once, after the v1 review.

## 9. Ethics
Errors can merge facts from different incidents into one broadcast, which could misinform listeners. This is why precision matters most, and why the station keeps the higher-precision baseline. The publisher excerpts are copyrighted, so the package stays private. Labels produced with AI help may carry the model's biases; independent human review is planned before any claim is made about label quality.

## 10. Conclusion and next steps
A frozen multilingual encoder with a fixed cosine threshold remains the better pair matcher on our held-out day. The learned 5-parameter head improves recall but loses too much precision, and the experiment gives no reason to replace the station's rule. Next steps:
1. Complete human review and report kappa.
2. Collect a third day to use as a fresh test set.
3. Re-tune the threshold on a validation day with realistic prevalence.
4. Add hard negatives (same event type, different place) and named-entity overlap features.
5. Evaluate end-to-end clustering in the station.
