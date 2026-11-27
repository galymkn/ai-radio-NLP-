# Multilingual news event matching for NewsRadio

## Reproduce
From the radio project root on the demo Mac:
```sh
source .venv/bin/activate
python event_matching/match.py
HF_HUB_OFFLINE=1 python event_matching/match.py --demo "Town opens new library" "В городе открылась новая библиотека"
```
The first command uses the saved embeddings and overwrites metric outputs and checkpoint deterministically except for time and memory. For fresh encoding use `python event_matching/encode.py` with the pinned multilingual MiniLM weights available locally or through Hugging Face. The demo accepts any two new headlines. Its score is a weighted classifier output, not a calibrated probability. The live radio starts separately with `python -m newsradio run`. The research head is evaluated at the article-pair level. The station groups articles incrementally against centroids, so these results measure a matching component rather than end-to-end cluster quality.

## Dataset card
Public RSS feeds and Telegram web previews, English and Russian news, collected Nov 26–27 2026. Source and URL are recorded per article in dataset.json. Publisher rights apply. Keep this classroom package private unless redistribution rights for the excerpts have been checked. Labels are assistant-assigned and pending independent human review. Event ID denotes the specific reported incident or decision. Follow-ups describing the same incident can match. Different speeches may discuss the same policy. A broad roundup may cover multiple incidents and was excluded where identified. Borderline cases can change the results. Dataset includes shorter next-day RSS summaries, which also change input quality.

149 first-day articles plus 156 later records with URLs absent from the first snapshot. After excluding identified roundups, ambiguous multi-event pieces, overlapping earlier coverage, and duplicate leads, 282 documents remained. Single-event groups were assigned before forming pairs. Day one was divided by event using seed 73, stratifying groups by whether they had multiple articles. Day two supplied the held-out test. Within each split all unordered pairs were used. Related names and topics can recur across days. A hundred negative pairs sharing an article are statistically dependent. Validation and test contain few positive events.

| Split | Articles | Pairs | Same event |
|---|---:|---:|---:|
| Train |100|4950|18|
| Validation |46|1035|19|
| Test |136|9180|23|

## Architecture
Raw article → cleaned headline and first 600 lead characters → two input views (headline and headline plus lead) → pretrained tokenizer → token IDs, padding and attention masks → frozen Transformer → masked mean pooling → normalized 384-dimensional vectors. Maximum 128 tokens per view. The project keeps names, numbers and word order. No stemming or stop-word removal. From each article pair compute cosine for each view, Jaccard of lowercase headline word sets and Jaccard of digit-only token sets. Empty overlap yields zero. Standardize using training mean and standard deviation. A linear layer plus sigmoid learns four coefficients and one bias. MiniLM has 117,653,760 frozen parameters. The exact model revision is in encode.py. Inference on instructor-provided text uses these same functions.

## Experiment
Pair-level baseline uses the station's 0.72 threshold on the headline-plus-lead cosine. It represents the similarity rule between two articles. The supervised model uses full-batch gradient descent, zero initialization, learning rate 0.5, L2 coefficient 0.001 and 2000 epochs for v2. The original run used LR 0.05 and 200 epochs. v1's lowest validation loss was its final point. That observation justified a longer optimization run. It did not by itself diagnose insufficient model capacity. v2's minimum balanced validation loss was 0.11884 at epoch 535. Curves continue beyond checkpoint selection. Positive-class validation F1 after selecting a threshold was 0.780 at threshold 0.675. The plotted F1 curve uses a fixed threshold of 0.50 and can differ from this value.
 
## Results and interpretation
The same next-day test set was evaluated for v1 and again for v2. Both results are disclosed. LR and training budget came from the validation experiment suggested after the v1 review. Both decision thresholds and checkpoint epochs were selected on validation. A third day remains desirable for an independent confirmation.

| Method | TP | FP | FN | TN | Precision | Recall | F1 | 95% F1 interval |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fixed cosine |11|7|12|9150|.611|.478|.537|.276–.718|
| v1 |18|103|5|9054|.149|.783|.250|.123–.392|
| v2 |20|126|3|9031|.137|.870|.237|.115–.383|

Paired bootstrap resamples test events with replacement, keeping jointly scored pairs and all methods aligned. Within-event pairs receive the event multiplicity, between-event pairs the product of multiplicities. Percentile interval for v2 minus baseline F1 was [-0.527,-0.075], conditional on this day's provisional labels. This cannot quantify day-to-day variation or annotation uncertainty. Precision/recall tradeoffs and classwise results appear in results.json. Validation/test positive prevalence was 19/1035 versus 23/9180. Source composition or excerpt length might also contribute; causality has not been established.

## Reproduction
Run match.py --lr 0.5 --epochs 2000, intervals.py and plot_curves.py in that order. Plotting requires reportlab. The v1 directory contains the archived original outputs, slides and source. Human review is pending. See human_review/README.txt for full test coverage, 40 shared documents, sentence review and kappa. Re-score separately on reconciled labels, preserving today's results.

## Error analysis
See test_predictions.json for pair scores. An interview and a subsequent response can share names and topics but receive different event labels. Consult title, lead and URL to adjudicate these cases with two independent reviewers.

## Individual defense
Show dataset.json for provenance, encode.py for preprocessing, match.py for the loss and update, checkpoint.json for selected weights, history.json and learning-curves.pdf for model selection, and results.json for the comparison. Explain how a false positive could mix two events into one broadcast. The earlier observation of one insufficiently supported sentence in 49 was an AI-assisted review of eight radio scripts, a different task and sample. The pair classifier changes neither the generator nor voice synthesis.
