# Data access

This public copy does **not** contain the text of the news articles. Headlines, leads and full article texts belong to their publishers (BBC, Al Jazeera, The Guardian, RIA Novosti, Interfax and others). They are kept out of public version control, as the project's root `.gitignore` also requires.

## What was removed
| File | Public copy keeps | Removed |
|---|---|---|
| `dataset.json` | `id, url, source, event, split` | `title`, `lead` |
| `human_review/events_1.csv`, `events_2.csv` | `id, url, your_event_id` | `title`, `lead` |
| `human_review/sources.json` | `id, source, story_id, url, published_at, fetched_at` | `title`, `text` |

Everything else is unchanged, including:
- `embeddings.npz` (vectors only);
- labels and splits;
- `checkpoint.json`, `history.json`, `results.json`, `test_predictions.json`, `intervals.json`, `benchmark.json`;
- the figures and slides.

The removal was applied to every commit of this package's history, so no earlier commit contains the texts either.

## What works with the public copy
- `python match.py --demo ...` runs on any new text; it needs only the checkpoint and the encoder.
- `python intervals.py` recomputes the confidence intervals; it needs only event IDs and the saved predictions.
- Every reported number can be checked against `results.json`, `test_predictions.json` and the source URLs.

## What needs the full dataset
`match.py` (training and evaluation), `encode.py`, `benchmark.py` and `plot_curves.py` read the headlines and leads. The instructor can get the full `dataset.json` from the team on request, shared privately. Replace the public `dataset.json` with it and run the commands in README.md. The reproduction check in README.md was run on the full dataset.
