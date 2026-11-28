# NewsRadio: an automated, fact-checked news radio

Final project report for Intro to NLP, KBTU, November 2026.
Team: Galymzhan Kansultan, Apabek Ysmaiyl.

NewsRadio reads ten public news sources in English and Russian, groups reports about the same event into stories, writes a short English radio bulletin for each story with an LLM, checks every sentence of that bulletin against the sources, reads it aloud and plays the result as a continuous live stream. Listeners can also ask the radio a question and get a spoken answer drawn only from today's reports.

## Contents

1. [Summary](#1-summary)
2. [Problem and goals](#2-problem-and-goals)
3. [System overview](#3-system-overview)
4. [How each NLP component works](#4-how-each-nlp-component-works)
5. [Engineering for a free, always-on station](#5-engineering-for-a-free-always-on-station)
6. [Evaluation](#6-evaluation)
7. [Limitations and ethics](#7-limitations-and-ethics)
8. [Future work](#8-future-work)
9. [Reproducing the results](#9-reproducing-the-results)
10. [Repository map](#10-repository-map)
11. [Appendix: metric definitions](#11-appendix-metric-definitions)

## 1. Summary

| What | Result |
|---|---|
| Sources | 10 (5 RSS feeds, 5 public Telegram channels), English and Russian |
| Live run, 27–28 Nov 2026 | 379 articles → 290 stories → 126 voiced bulletins; peak 107 bulletins in one hour |
| Cross-lingual grouping | 14 of 50 multi-article stories combine English and Russian reports |
| Newsworthiness filter | 12 stories rejected (promo videos, daily digests, press releases) |
| Bulletins vs their sources, BERTScore F1 | 0.711, against 0.612 for an unrelated story's sources (control); higher than control for 74 of 78 bulletins |
| Paraphrasing, ROUGE-2 F1 | 0.201 for the editor against 0.351 for copying the lead (lead-3): the editor rewrites rather than copies |
| Story grouping, pair F1 on a held-out day | 0.537 for the station's cosine rule; a learned 5-parameter matcher reached 0.237, so the rule was kept (team study, §6.3) |
| NER place veto | No change on the held-out day (F1 0.645 with and without): none of the rule's grouped pairs named different places; kept as a safeguard (§6.4) |
| Automated tests | 59 (`pytest -q`) |
| Running cost | zero: free Gemini tier, local encoders, free Edge voices |

## 2. Problem and goals

A news radio has to answer four questions every few minutes: what happened, which reports describe the same event, how to say it in about a minute of speech, and whether everything that is said is supported by a source. NewsRadio automates all four for a station aimed at listeners in Kazakhstan, using sources in English and Russian and broadcasting in English.

Goals set in the project plan (15 weeks):

| Plan | Delivered |
|---|---|
| Ingestion from RSS and Telegram, raw storage | 10 sources, raw and cleaned text kept, SQLite or Postgres |
| Embeddings and story clustering (topic detection and tracking) | Multilingual SBERT, incremental centroid clustering, NER place veto |
| LLM summarization grounded in sources, plus a separate verification call | Structured-output editor and per-sentence fail-closed fact check |
| TTS with text normalization | Edge neural voices, loudness normalization, optional number/date normalizer |
| End-to-end episode, scheduler, web player | Continuous `/stream`, web player with a 3D tuning dial, MP3 episode export |
| Broadcast memory and importance ranking | Each story plays once per listener; importance from the editor and the number of outlets |
| Evaluation: hallucinations, dedup accuracy, latency | ROUGE, BERTScore, pair-level grouping study, NER veto study, labelling export, latency measurement |
| Added beyond the plan | "Ask the radio" (retrieval-augmented answers), named entity recognition |

## 3. System overview

```
 RSS feeds ─┐                                                   ┌─> web player (3D dial, live meter, Ask the radio)
            ├─> clean text ─> embeddings ─> stories ─> editor ─> fact check ─> TTS ─> clip ─> news wheel ─┤
 Telegram ──┘   (raw kept)    + NER         (clusters)  (LLM)    (LLM, per        (Edge)                   └─> /stream (MP3 for VLC / radio apps)
                                                                  sentence)
 listener question ─> retrieve similar articles ─> LLM answer from them only ─> fact check ─> spoken answer
```

The station is one Python process (FastAPI). Inside it, an ingest loop collects and clusters news every 5 minutes, and two editor workers turn waiting stories into clips continuously, so the next bulletin is being written while the current one plays.

| Stage | Module | Technique |
|---|---|---|
| Ingestion | `ingest/rss.py`, `ingest/telegram.py` | feedparser, trafilatura full-text extraction, BeautifulSoup for Telegram web previews |
| Cleaning | `ingest/__init__.py` | HTML, URL and emoji removal, whitespace normalization |
| Representation | `cluster.py` | Sentence embeddings, `paraphrase-multilingual-MiniLM-L12-v2` (384-d) |
| Entities | `ner.py` | Token classification, `Babelscape/wikineural-multilingual-ner` (PER, ORG, LOC) |
| Grouping | `cluster.py` | Incremental single-pass clustering, cosine ≥ 0.72 to the story centroid, place veto |
| Writing | `editor.py` | LLM with a JSON schema (Gemini 3.5 Flash-Lite by default; Claude optional) |
| Checking | `editor.py` | LLM per-sentence entailment check, fail-closed |
| Speech | `tts.py`, `audio.py` | Edge neural TTS, EBU R128 loudness normalization, gapless MP3 |
| Scheduling | `wheel.py`, `server.py` | Per-listener ranking by priority, freshness and variety |
| Questions | `ask.py` | Dense retrieval + grounded generation + verification (RAG) |

## 4. How each NLP component works

### 4.1 Ingestion and text cleaning

Every 5 minutes the station reads up to 20 items per source that are younger than 24 hours. RSS entries are followed to the article page, and trafilatura extracts the main text (falling back to the feed summary). Telegram channels are read through their public web preview (`t.me/s/<channel>`), without API keys; the first line of a post becomes its headline. Posts shorter than 80 characters are skipped. HTML tags, links and emoji are removed; the original text is kept in `raw_items`. Duplicates are removed by URL. Identical headlines from different outlets are kept on purpose, because the number of outlets reporting an event is a signal of its importance.

### 4.2 Sentence embeddings

Each article is represented by its headline plus the first 600 characters of text, where reports state what happened. The multilingual paraphrase model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` maps this to a 384-dimensional unit vector. It was trained so that translations and paraphrases land close together, which lets an English BBC report and a Russian Telegram post about the same event be compared directly. Similarity is the cosine of the angle between two vectors (a dot product, since vectors are normalized).

### 4.3 Story clustering (topic detection and tracking)

News arrives as a stream and the number of events is unknown, so the station uses incremental single-pass clustering instead of k-means:

1. Each story keeps a centroid: the normalized mean vector of its articles.
2. A new article is compared with the centroids of stories updated in the last 48 hours.
3. If the best cosine is at least 0.72 (and the NER check in §4.4 does not object), the article joins that story and the centroid is updated; otherwise it starts a new story.

A story with new articles is re-edited after 30 minutes. The editor sees the previously aired script and writes a new version only if the sources contain material new facts. Stories reported by more outlets, and stories with Kazakhstani sources (Astana Times, Tengrinews), are edited first.

### 4.4 Named entity recognition and the place veto

Cosine similarity captures topic, so "drone attack on Kyiv" and "drone attack on Belgorod" look alike. The team's event-matching study (§6.3) found exactly this failure mode: the same kind of event in a different place. NewsRadio therefore runs a multilingual NER model (WikiNEuRal: mBERT fine-tuned for PER, ORG, LOC and MISC in nine languages including English and Russian) over the headline and the first 400 characters of every article. Entities with a score below 0.6 and the MISC class are dropped.

During clustering, an article may not join a story if both name places and no place matches. Names are compared in three steps, so that matches work across scripts and Russian grammatical cases:

1. Latin spelling: Russian and Kazakh letters are transliterated (Белгород → belgorod = Belgorod).
2. Shared stem: keys of at least 5 letters that differ only in a short ending (Москва ~ Москве).
3. Meaning: the multilingual encoder puts the names within cosine 0.75 (Киев ~ Kyiv).

The same entities are summarized per story and shown in the player ("People", "Places", "Organisations").

### 4.5 The editor: grounded abstractive summarization

For each story the editor receives up to six reports, one per outlet and newest first, each cut to 2,500 characters and marked as untrusted data (prompt-injection defence). The LLM must answer through a fixed JSON schema:

| Field | Meaning |
|---|---|
| `newsworthy` | false for ads, promotions, horoscopes, betting, pure opinion |
| `headline` | at most 12 words |
| `category` | one of 11, including `kazakhstan` and `central_asia` |
| `priority` | 1 (filler) to 5 (breaking) |
| `ttl_hours` | how long the clip stays relevant (1–48) |
| `has_new_info` | for re-edits: are there material new facts? |
| `script` | the bulletin |
| `sources` | outlets actually used |

The script rules: written for the ear, 120–180 words, short declarative sentences, key fact first, every claim attributed to its outlet ("according to the BBC"), only facts present in the sources, no copied sentences, numbers written as a newsreader says them, and shorter rather than padded when the sources are thin. Values outside the schema's ranges are clamped in code.

### 4.6 Fact checking: per-sentence verification

The script is split into sentences (abbreviations such as "U.S." or "Dr." do not end a sentence), numbered, and sent to the LLM again with the sources. For each sentence the checker must return exactly one boolean: supported or not. A sentence is not supported if any name, number, date, place or claim is absent from or contradicts the sources. This is an entailment (natural language inference) task.

The check fails closed: a missing, duplicated or non-boolean verdict cancels publication instead of letting unchecked text through. The story is retried with exponential backoff (2, 4 minutes) and marked failed after three attempts. Unsupported sentences are deleted; if fewer than two sentences remain, the story is rejected.

### 4.7 Text to speech and audio

Bulletins are read by Microsoft Edge neural voices (`en-US-AndrewNeural`; `ru-RU-DmitryNeural` for Russian answers). An optional normalizer spells out dates, numbers and abbreviations for English and Russian, but it is off by default because the editor already writes numbers in words. ffmpeg normalizes loudness to −16 LUFS (EBU R128), adds 0.6 s of silence and encodes every clip identically (44.1 kHz mono, 128 kbit/s, no ID3 or Xing header), so clips can be concatenated byte by byte into one continuous MP3 stream.

### 4.8 The news wheel

Each listener (web player or `/stream` connection) has its own order. A clip's score is priority × freshness (halved every 3 hours) ÷ (1 + times heard), ×1.5 if not heard yet, ×0.8 if it has the same category as the previous clip, ×0.7 per skip of its category, and ×1.5 for Kazakhstan and Central Asia. Every story plays once; when nothing new is ready, the player says so and waits, and a listener can replay any story by selecting it. A station identification plays after every five stories.

### 4.9 Ask the radio (retrieval-augmented generation)

1. Retrieve: the question is embedded with the same encoder as the articles. The six most similar articles from the last 48 hours are taken (cosine ≥ 0.25, at most two per story so one event cannot crowd out the rest). If none qualifies, the radio says the reports do not cover it and no LLM call is made.
2. Generate: the LLM answers in 40–120 words from those numbered reports only, attributes facts to outlets, returns the ids of the reports it used, or says the reports do not answer the question.
3. Verify: the answer goes through the same per-sentence check as bulletins.
4. Speak: the answer plays on air with links to its sources. A question in Russian gets a Russian answer and voice.

## 5. Engineering for a free, always-on station

- **Free LLM quota.** On the free Gemini tier, Flash models allow about 20 requests a day (about 10 stories) and Flash-Lite about 500. The station uses Flash-Lite, spaces calls by 6 seconds, caps itself at 480 calls a day, of which 60 are reserved for listener questions, and stores the count in `data/llm_usage.json` by Pacific-time day, like Google's own quota, so restarts do not reset it. A quota error (HTTP 429) pauses editing without counting against the story's retry budget.
- **Throughput.** Two editor workers share one quota and claim stories so they never take the same one. Peak production was 107 bulletins in one hour; in steady operation the daily quota (about 210 stories) is the limit, not speed.
- **Storage.** SQLite by default, Postgres through `DATABASE_URL` (Docker Compose included).
- **Interface.** The player presents the station as a radio receiver: headlines sit on a backlit 3D tuning drum that turns to each new story, a needle on the tuning scale shows clip progress, and an analog VU meter follows the real audio level. Fonts are bundled, so the player works offline; reduced motion is respected.
- **API.** `/` player, `/stream` continuous MP3, `/api/next`, `/api/clips`, `/api/status`, `/api/skip`, `/api/started`, `/api/ask`.

## 6. Evaluation

### 6.1 Live operation, 27–28 November 2026

Measured from the station's database. The station ran in sessions, not continuously.

| Measure | Value |
|---|---|
| Articles collected | 379 (225 Telegram, 154 RSS) |
| Most active sources | Al Jazeera 62, RIA Novosti 62, Interfax 52, BBC Russian 49, DW 45, BBC World 44 |
| Stories | 290: 240 single-article, 50 multi-article (35 from two or more outlets, largest: 12 articles) |
| Cross-lingual stories | 14 of the 50 multi-article stories mix English and Russian reports |
| Editor decisions | 121 stories aired, 12 rejected, 157 not yet edited when the run stopped |
| Bulletins | 126; median 104 words and 41 s of audio; 33% reach 120+ words and 34% are under 80 (the prompt asks for a shorter bulletin when sources are thin, and the first 9 bulletins used an earlier 50–110-word limit) |
| Categories | world 68, business 16, kazakhstan 14, politics 12, tech 6, sports 4, culture 4, health 2 |
| Peak throughput | 107 bulletins in one hour (Flash-Lite, two workers) |
| Editing errors | 1 recorded (handled by the retry budget) |

Rejected stories show the newsworthiness filter at work: a promotional video post about the capital, the daily digest "What happened today", an English roundup of foreign media coverage and a bank's press release about awards.

Publication-to-broadcast latency (first playback minus earliest source publication) was recorded for 21 clips, but the median of 16 hours reflects the gaps between test sessions, not the pipeline. A 24-hour continuous run (`scripts/soak.py`) is needed before quoting latency.

### 6.2 Bulletin quality: ROUGE and BERTScore

No human-written reference bulletins exist, so each of the 78 bulletins available at evaluation time was compared with the reports its editor received. Two reference rows make the numbers interpretable: lead-3 copies the first three sentences of the longest source (maximum overlap), and control compares each bulletin with the sources of a different story (the level of unrelated text). ROUGE uses only English sources because its tokenizer drops Cyrillic (51 bulletins); BERTScore uses multilingual BERT (layer 9) on all sources and takes the best-matching source.

| | ROUGE-1 P | ROUGE-1 R | ROUGE-2 F | ROUGE-L F | BERTScore P | BERTScore R | BERTScore F |
|---|---:|---:|---:|---:|---:|---:|---:|
| Editor | 0.761 | 0.275 | 0.201 | 0.254 | 0.742 | 0.683 | 0.711 |
| Lead-3 (copying) | 1.000 | 0.239 | 0.351 | 0.353 | 0.894 | 0.747 | 0.812 |
| Control (other story) | 0.325 | 0.086 | 0.012 | 0.074 | 0.632 | 0.594 | 0.612 |

What the numbers show:

- **Bulletins are about their own sources.** The editor is far above control on every metric, and its BERTScore F1 is higher than control's for 74 of 78 individual bulletins.
- **The editor paraphrases.** 76% of a bulletin's words appear in its sources (ROUGE-1 precision), but word pairs overlap much less than when copying (ROUGE-2 0.20 vs 0.35). The prompt asks for exactly this.
- **Meaning is kept while wording changes.** BERTScore precision (0.742) sits between control (0.632) and copying (0.894).
- **Low recall is expected.** A 100-word bulletin cannot cover several full articles; copying the lead has equally low recall.

BERTScore precision split by source language:

| BERTScore P | Has an English source (n = 51) | Russian sources only (n = 27) |
|---|---:|---:|
| Editor | 0.783 | 0.664 |
| Control | 0.652 | 0.593 |

The four lowest-scoring bulletins were all translations of Russian reports (for example a faithful account of RIA Novosti's report on suspected mpox cases in Saint Petersburg). Multilingual BERT aligns English and Russian tokens less well than English with English, so cross-lingual scores are lower regardless of accuracy. Even so, translated bulletins stay well above their control. Scores for the two groups should be reported separately.

Limits of these metrics: closeness to the sources is not correctness, since one invented number barely changes ROUGE or BERTScore; numbers written in words lower ROUGE against digits in the sources. The per-sentence fact check and human labelling (`export-labels`) remain the measure of factual accuracy.

### 6.3 Story grouping: learned matcher vs the cosine rule (team study)

Apabek Ysmaiyl tested whether a supervised model groups articles better than the station's rule. Data: 282 English and Russian articles collected on 26–27 November, labelled with event ids (AI-assisted, human review pending). Day one was split by event into training (100 articles) and validation (46); day two (136 articles, 9,180 pairs, 23 same-event pairs) is the held-out test. The model keeps the multilingual encoder frozen and learns how to combine four similarity features (lead cosine, headline cosine, headline word overlap, headline number overlap) with a 5-parameter logistic layer and class-balanced loss.

| Test day, pair level | Precision | Recall | F1 [95% CI] | Average precision |
|---|---:|---:|---:|---:|
| Station rule, cosine ≥ 0.72 | 0.611 | 0.478 | **0.537** [0.276, 0.718] | **0.583** |
| Learned matcher v1 | 0.149 | 0.783 | 0.250 [0.123, 0.392] | 0.467 |
| Learned matcher v2 | 0.137 | 0.870 | 0.237 [0.115, 0.383] | 0.519 |

The learned matcher finds more paraphrased and cross-lingual pairs, but it matches many pairs that describe the same kind of event in different places (two drone attacks, two parliaments' budgets), and one generic politics article causes 18 of its false positives. The validation day suggested the opposite ranking (F1 0.78 vs 0.19), a reminder that five multi-article events are not enough to choose a model. The station therefore keeps the cosine rule, and the place veto (§4.4, §6.4) targets the failure the study identified. The full study, slides and defense notes are in `event_matching/`.

### 6.4 NER place veto

`scripts/eval_ner.py` evaluates the veto on the team's labelled test day (§6.3). Article texts were taken from the station's local database by URL, which covered 113 of the 136 test articles (6,328 pairs, 16 same-event pairs). 86 of the 113 articles name at least one place. Both rules use the same texts, so they are directly comparable; the baseline numbers differ from §6.3 because the subset and the texts differ.

| Test day subset, pair level | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cosine ≥ 0.72 | 10 | 5 | 6 | 0.667 | 0.625 | 0.645 |
| Cosine ≥ 0.72 + place veto | 10 | 5 | 6 | 0.667 | 0.625 | 0.645 |

The veto did not change a single decision. None of the 15 pairs that the cosine rule grouped named two different places, so there was nothing to split: on this data the rule's five false positives are not of the "same kind of event, different place" type. That error type dominated the learned matcher in §6.3, which accepts far more pairs, but the stricter 0.72 rule already avoids it here. The veto also caused no new misses.

Conclusion: the veto is kept as a zero-cost safeguard for the error type observed in the team study, but its benefit is not demonstrated on this test day. A larger test set with more same-type events in different places is needed to measure it. Unit tests confirm the intended behaviour (Kyiv and Belgorod drone attacks are kept apart; Белгород = Belgorod and Москва ~ Москве still merge).

### 6.5 Tests

59 automated tests cover cleaning, Telegram parsing, incremental clustering, the place veto and cross-script name matching, the editor's structured output and fail-closed verification for both LLM backends, quota handling (spacing, daily cap, reserve for questions, persistence across restarts, quota errors not burning retries), parallel workers, the news wheel (priority, freshness, play-once behaviour), retrieval and grounded answers, the web API, the evaluation scripts and the metric code. One test needs the Postgres driver and is skipped without it.

## 7. Limitations and ethics

- **Hallucination is reduced, not eliminated.** The checker is itself an LLM, and Flash-Lite is weaker than Flash. No human-labelled hallucination rate has been measured yet; `export-labels` produces the labelling sheet.
- **Labels.** The event labels are AI-assisted and await independent human review; the ROUGE/BERTScore sample is 78 bulletins from two days.
- **Coverage.** Most stories have a single source (240 of 290), so cross-checking across outlets is possible only for a minority.
- **Free tier.** About 210 stories a day; Google may use free-tier prompts to improve its products, which is acceptable for public news but not for private data.
- **Copyright.** Source texts belong to their publishers. They stay in the local database; the public repository holds code, labels and metrics, not article texts.
- **Responsibility.** An error can merge facts from two incidents or misstate a fact on air. Precision is prioritized over recall throughout (the cosine rule over the learned matcher, the place veto, fail-closed checking, attribution of every claim), and listeners can always see the sources.

## 8. Future work

1. Human labelling of hallucinations and event pairs (inter-annotator agreement), then a hallucination rate per 100 sentences.
2. A local NLI model for the fact check, which would halve LLM calls and remove the checker's dependence on the same LLM family.
3. Translate Russian sources before scoring, or use a cross-lingual scorer, so BERTScore is comparable across languages.
4. A 24-hour continuous run to measure latency and stability.
5. Broadcasting in Kazakh (the editor and Edge TTS support it; pronunciation needs checking).

## 9. Reproducing the results

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[embeddings,eval,dev]'
cp .env.example .env                 # add GEMINI_API_KEY (free key from Google AI Studio)
python -m newsradio check-llm        # one test request
python -m newsradio run              # the station: http://localhost:8000
python scripts/eval_text.py          # ROUGE and BERTScore (§6.2)
python scripts/eval_ner.py           # NER place veto (§6.4)
pytest -q                            # tests (§6.5)
python -m newsradio demo             # offline demo: no keys, no downloads
```

The evaluation scripts read article texts from the local `data/newsradio.db` and write their results to `data/`, which is not published.

## 10. Repository map

| Path | Contents |
|---|---|
| `newsradio/` | The station: ingestion, clustering, NER, editor, fact check, TTS, wheel, server, RAG, player |
| `config.yaml` | Every setting with comments: sources, thresholds, models, quota |
| `scripts/eval_text.py`, `scripts/eval_ner.py` | Evaluations in §6.2 and §6.4 |
| `scripts/soak.py` | Long-run monitoring |
| `event_matching/` | The team's event-matching study (§6.3): report, slides, data card, code |
| `tests/` | Automated tests |
| `README_RU.md` | Operating manual (Russian) |

## 11. Appendix: metric definitions

- **Precision / recall / F1 (pairs).** Over all pairs of articles: precision is the share of pairs grouped together that are truly the same event; recall is the share of same-event pairs that were grouped; F1 is their harmonic mean.
- **Average precision.** Area under the precision-recall curve over all thresholds: ranking quality independent of one threshold.
- **ROUGE-N.** Overlap of word n-grams between a candidate and a reference. Precision: share of the candidate's n-grams found in the reference; recall: share of the reference's n-grams found in the candidate. ROUGE-L uses the longest common subsequence instead of n-grams. Porter stemming is applied.
- **BERTScore.** Each token of candidate and reference is embedded with a contextual model; every token is matched to its most similar token in the other text by cosine. Precision averages over candidate tokens, recall over reference tokens, F1 combines them. Raw values occupy a narrow band, so they are read against a control.
- **Cosine similarity.** The dot product of two unit vectors: 1 for the same direction, near 0 for unrelated texts.
