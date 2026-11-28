"""How close are the on-air scripts to their sources? ROUGE and BERTScore.

There are no human-written reference scripts, so every script is compared with the reports the
editor was given (articles of its story fetched before the clip was made, one per outlet).
Two reference points make the numbers readable:
  lead-3    first three sentences of the longest source: pure copying, the highest overlap possible
  control   the script against the sources of a DIFFERENT story: what unrelated text scores

Metrics (script = candidate, sources = reference):
  ROUGE-1 / ROUGE-2 / ROUGE-L   word / word-pair / longest-common-subsequence overlap.
      Only English sources: the standard ROUGE tokenizer drops Cyrillic letters.
      precision = share of the script's words found in the sources   (are we using source words?)
      recall    = share of the sources' words that made it into the script (how much did we cover?)
  BERTScore P / R / F           token-to-token similarity of contextual embeddings
      (multilingual BERT, layer 9), so paraphrases and Russian sources count. Raw values sit in a
      narrow band (~0.6-0.9); compare with the control row, not with zero. For each script the
      best-matching source is used, because the script draws on several sources.

Run from the project root:  pip install -e '.[eval]'  then  python scripts/eval_text.py
Writes data/text_eval.json (data/ is not published).
"""
import argparse
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from newsradio.editor import split_sentences  # noqa: E402

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def is_english(text: str) -> bool:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    return bool(letters) and sum(1 for c in letters if CYRILLIC.match(c)) / len(letters) < 0.2


def collect(con, max_sources=6, max_chars=2500) -> list[dict]:
    """Every voiced story clip with the sources its editor saw (one per outlet, newest first)."""
    con.row_factory = sqlite3.Row
    clips = con.execute("SELECT id, story_id, created_at, headline, script FROM clips "
                        "WHERE audio_file NOT LIKE 'ask-%' ORDER BY id").fetchall()
    out = []
    for c in clips:
        rows = con.execute("SELECT source, title, text FROM articles WHERE story_id=? AND fetched_at<=? "
                           "ORDER BY COALESCE(published_at, fetched_at) DESC", (c["story_id"], c["created_at"]))
        seen, docs = set(), []
        for r in rows:
            if r["source"] not in seen:
                seen.add(r["source"])
                docs.append({"source": r["source"], "text": f"{r['title']}. {r['text'][:max_chars]}"})
            if len(docs) >= max_sources:
                break
        if docs and c["script"].strip():
            out.append({"clip": c["id"], "headline": c["headline"], "script": c["script"], "docs": docs})
    return out


def lead3(docs) -> str:
    richest = max(docs, key=lambda d: len(d["text"]))
    return " ".join(split_sentences(richest["text"])[:3])


def rouge(cand: str, docs) -> dict | None:
    from rouge_score import rouge_scorer
    english = [d["text"] for d in docs if is_english(d["text"])]
    if not english or not is_english(cand):
        return None
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(" ".join(english), cand)
    return {f"{k}_{m}": round(getattr(v, n), 4) for k, v in sc.items()
            for m, n in (("p", "precision"), ("r", "recall"), ("f", "fmeasure"))}


def bertscore(pairs, model: str, layers: int | None, batch_size=8):
    """pairs: [(candidate, reference)] -> [(P, R, F)]."""
    from bert_score import score
    cands, refs = zip(*pairs)
    P, R, F = score(list(cands), list(refs), model_type=model, num_layers=layers, batch_size=batch_size,
                    verbose=False)
    return list(zip(P.tolist(), R.tolist(), F.tolist()))


def evaluate(items, bert_fn=None) -> dict:
    systems = {
        "editor": [it["script"] for it in items],
        "lead-3": [lead3(it["docs"]) for it in items],
        "control": [it["script"] for it in items],
    }
    # control: each script against the sources of the next story (a different event)
    docs_for = {
        "editor": [it["docs"] for it in items],
        "lead-3": [it["docs"] for it in items],
        "control": [items[(i + 1) % len(items)]["docs"] for i in range(len(items))],
    }
    per_clip = {name: [{"clip": it["clip"]} for it in items] for name in systems}
    for name in systems:
        for row, cand, docs in zip(per_clip[name], systems[name], docs_for[name]):
            r = rouge(cand, docs)
            if r:
                row.update(r)
    if bert_fn:
        for name in systems:
            pairs, owner = [], []
            for i, (cand, docs) in enumerate(zip(systems[name], docs_for[name])):
                for d in docs:
                    pairs.append((cand, d["text"]))
                    owner.append(i)
            best: dict[int, tuple] = {}
            for i, prf in zip(owner, bert_fn(pairs)):
                if i not in best or prf[2] > best[i][2]:
                    best[i] = prf
            for i, (p, r, f) in best.items():
                per_clip[name][i].update(bert_p=round(p, 4), bert_r=round(r, 4), bert_f=round(f, 4))
    summary = {}
    for name, rows in per_clip.items():
        keys = sorted({k for row in rows for k in row if k != "clip"})
        summary[name] = {k: {"mean": round(statistics.mean(v), 4),
                             "sd": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0, "n": len(v)}
                         for k in keys for v in [[row[k] for row in rows if k in row]]}
    return {"clips": len(items), "summary": summary, "per_clip": per_clip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/newsradio.db")
    ap.add_argument("--bert-model", default="bert-base-multilingual-cased")
    ap.add_argument("--bert-layers", type=int, default=9, help="layer used by bert-score for multilingual BERT")
    ap.add_argument("--no-bertscore", action="store_true", help="ROUGE only (no model download)")
    args = ap.parse_args()

    items = collect(sqlite3.connect(args.db))
    if len(items) < 2:
        sys.exit("Need at least two voiced stories in the database. Let the station run first.")
    english = sum(1 for it in items if any(is_english(d["text"]) for d in it["docs"]))
    print(f"{len(items)} on-air scripts; {english} have at least one English source (used for ROUGE)")
    bert_fn = None
    if not args.no_bertscore:
        print(f"BERTScore with {args.bert_model} (first run downloads it, ~700 MB)...")
        bert_fn = lambda pairs: bertscore(pairs, args.bert_model, args.bert_layers)  # noqa: E731
    result = evaluate(items, bert_fn)

    cols = [("rouge1_p", "R-1 P"), ("rouge1_r", "R-1 R"), ("rouge2_f", "R-2 F"), ("rougeL_f", "R-L F"),
            ("bert_p", "BERT P"), ("bert_r", "BERT R"), ("bert_f", "BERT F")]
    print("\n" + f"{'':10}" + "".join(f"{label:>9}" for _, label in cols))
    for name in ("editor", "lead-3", "control"):
        s = result["summary"][name]
        print(f"{name:10}" + "".join(f"{s[k]['mean']:>9.3f}" if k in s else f"{'-':>9}" for k, _ in cols))
    out = Path("data/text_eval.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMeans over scripts. Per-script scores and standard deviations: {out}")


if __name__ == "__main__":
    main()
