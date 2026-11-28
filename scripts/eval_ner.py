"""Does the NER place veto make story grouping better? Pair-level evaluation on the team's labels.

Labels and splits: event_matching/dataset.json (human-checked event ids, public copy without texts).
Texts: looked up by URL in the local station database (data/newsradio.db), so nothing is published.

Two rules are compared on the same article pairs:
  baseline   cosine(headline + lead) >= cluster.threshold          (the station's rule, as in REPORT.md)
  baseline+  the same, unless both articles name places and no place matches (NER veto)

Run from the project root:  python scripts/eval_ner.py [--split test]
Writes data/ner_eval.json (data/ is not published).
"""
import argparse
import json
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from newsradio.cluster import article_repr, make_embedder, places  # noqa: E402
from newsradio.config import load_config  # noqa: E402
from newsradio.ner import NameMatcher, make_ner, ner_text  # noqa: E402


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="train | validation | test | all")
    ap.add_argument("--db", default="data/newsradio.db")
    args = ap.parse_args()

    cfg = load_config()
    labels = json.loads(Path("event_matching/dataset.json").read_text(encoding="utf-8"))
    con = sqlite3.connect(args.db)
    texts = {u: (t, x) for u, t, x in con.execute("SELECT url, title, text FROM articles")}
    rows = [d | {"title": texts[d["url"]][0], "text": texts[d["url"]][1]}
            for d in labels if d["url"] in texts and (args.split == "all" or d["split"] == args.split)]
    total = sum(1 for d in labels if args.split == "all" or d["split"] == args.split)
    print(f"{len(rows)} of {total} labelled articles in split '{args.split}' found in {args.db}")
    if len(rows) < 2:
        sys.exit("Not enough articles with text. Run the station on the same days or ask the team for the full dataset.")

    print("Loading the encoder and the NER model (first run downloads them)...")
    embedder = make_embedder(cfg.cluster)
    if cfg.ner.backend == "none":
        sys.exit("ner.backend is 'none' in config.yaml; set it to 'transformers' to evaluate NER.")
    ner = make_ner(cfg.ner)
    embs = embedder.encode([article_repr(r["title"], r["text"]) for r in rows])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    ents = ner.extract([ner_text(r["title"], r["text"]) for r in rows])
    if not any(ents):
        sys.exit("The NER model found no entities at all: it probably failed to load (see the warning above).")
    locs = [places(e) for e in ents]
    matcher = NameMatcher(embedder, cfg.cluster.entity_match)

    counts = {"baseline": [0, 0, 0], "baseline+ner": [0, 0, 0]}   # tp, fp, fn
    changed = []
    for i, j in combinations(range(len(rows)), 2):
        gold = rows[i]["event"] == rows[j]["event"]
        sim = float(embs[i] @ embs[j])
        base = sim >= cfg.cluster.threshold
        veto = base and matcher.conflict(locs[i], locs[j])
        for name, pred in (("baseline", base), ("baseline+ner", base and not veto)):
            c = counts[name]
            c[0] += pred and gold
            c[1] += pred and not gold
            c[2] += (not pred) and gold
        if veto:
            changed.append({"ids": [rows[i]["id"], rows[j]["id"]], "same_event": gold, "cosine": round(sim, 3),
                            "places": [sorted(locs[i]), sorted(locs[j])],
                            "headlines": [rows[i]["title"], rows[j]["title"]]})

    result = {"split": args.split, "articles": len(rows), "pairs": len(rows) * (len(rows) - 1) // 2,
              "threshold": cfg.cluster.threshold, "entity_match": cfg.cluster.entity_match,
              "articles_with_places": sum(1 for x in locs if x),
              **{k: prf(*v) for k, v in counts.items()}, "vetoed_pairs": changed}
    out = Path("data/ner_eval.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"\n{result['pairs']} pairs, {result['articles_with_places']} of {len(rows)} articles name a place")
    print(f"{'rule':14} {'TP':>4} {'FP':>4} {'FN':>4}  precision  recall   F1")
    for name in counts:
        m = result[name]
        print(f"{name:14} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}  {fmt(m['precision']):>9}  {fmt(m['recall']):>6}  {fmt(m['f1'])}")
    right = sum(not c["same_event"] for c in changed)
    print(f"\nThe veto split {len(changed)} pairs: {right} were really different events (fixed), "
          f"{len(changed) - right} were the same event (new misses).")
    for c in changed[:15]:
        mark = "fixed" if not c["same_event"] else "MISS "
        print(f"  {mark} {c['places'][0]} vs {c['places'][1]}: {c['headlines'][0][:60]} | {c['headlines'][1][:60]}")
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    main()
