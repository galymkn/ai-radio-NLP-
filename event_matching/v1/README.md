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
Pair-level baseline uses the station's 0.72 threshold on the headline-plus-lead cosine. It represents the similarity rule between two articles. The supervised model uses full-batch gradient descent, zero initialization, learning rate 0.05, L2 coefficient 0.001 on weights, class-balanced binary cross entropy and 200 epochs. Both classes contribute equally to the loss despite the rarity of matches. No LR schedule. Save the epoch with minimum balanced validation loss, then select the threshold maximizing positive-class validation F1 on a preset grid (ties closest to 0.5). Final epoch 200 and threshold 0.825. One deterministic run. Training time covers the linear head and validation loop after vectors and pair features are ready. Encoder latency and full service throughput were not benchmarked here.

## Results
Held-out labels were fixed before evaluating either method. Positive-class precision measures correct matches among flagged pairs, recall measures retrieved matches among labeled matches, F1 is their harmonic mean. AP measures ranking averaged over recall increments. Results.json contains both classes and confusion matrices in sklearn order TN, FP / FN, TP. History.json stores train and validation data loss excluding the regularizer plus validation F1 at 0.5 per update.

| Method | TP | FP | FN | TN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed cosine |11|7|12|9150|.611|.478|.537|.583|
| Learned head |18|103|5|9054|.149|.783|.250|.467|

The supervised head retrieved seven more matches at the cost of 96 additional false positives. The fixed cosine baseline has higher F1 and AP in this test. The classifier received just 18 positive training pairs, and balanced loss treats a positive example as much more influential than an individual negative. Those are plausible reasons for overly broad matches. Further tests with more independent events and harder negatives are needed. The experiment provides no evidence for replacing the station's clustering rule.

## Error analysis
Test pair 1001/1017: two headlines about the same BBC interview, correctly matched. Pair 1001/1092: interview vs Trump's separate response on the same subject, model score .873, false positive under this rubric. Pair 1001/1091: short Russian excerpt from the interview, score .824, below .825, false negative. Inspect source titles and leads in dataset.json and all scores in test_predictions.json. Do not change the threshold based on these examples and reuse this test as fresh evidence.

## Individual defense
Show dataset.json for provenance, encode.py for preprocessing, match.py for the loss and update, checkpoint.json for selected weights, history.json and learning-curves.pdf for model selection, and results.json for the comparison. Explain how a false positive could mix two events into one broadcast. The earlier observation of one insufficiently supported sentence in 49 was an AI-assisted review of eight radio scripts, a different task and sample. The pair classifier changes neither the generator nor voice synthesis.
