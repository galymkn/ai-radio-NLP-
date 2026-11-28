"""scripts/eval_text.py: data collection, ROUGE and the BERTScore plumbing (with a stand-in scorer)."""
import importlib.util
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rouge_score")
from newsradio.db import DB  # noqa: E402
from newsradio.ingest import RawArticle  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_text", Path(__file__).resolve().parent.parent / "scripts/eval_text.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def make_db(path):
    db = DB(path)
    stories = [
        ("Metro", "Almaty's metro opened two new stations on Monday. The line grew by three kilometres. Officials expect forty thousand riders.",
         "According to the BBC, Almaty opened two new metro stations on Monday. The line is three kilometres longer."),
        ("Bank", "The central bank kept its base rate unchanged on Friday. Inflation is slowing but above target.",
         "The central bank left its base rate unchanged on Friday, the BBC reports. Inflation is slowing."),
    ]
    for i, (title, text, script) in enumerate(stories):
        aid = db.insert_article(RawArticle(f"https://x/{i}", "BBC", "t", title, text))
        sid = db.create_story(np.ones(2), time.time())
        db.set_article_story(aid, sid, np.ones(2))
        db.insert_clip(story_id=sid, headline=title, script=script, category="world", priority=3,
                       expires_at=time.time() + 60, audio_file=f"s{sid}.mp3", duration=5, sources=["BBC"])
    db.insert_clip(story_id=None, headline="q", script="answer", category=None, priority=1,
                   expires_at=time.time(), audio_file="ask-1.mp3", duration=1, sources=[])
    db.commit()


def test_scripts_are_closer_to_their_own_sources_than_to_others(tmp_path):
    make_db(tmp_path / "db")
    items = ev.collect(sqlite3.connect(tmp_path / "db"))
    assert len(items) == 2                       # answers to listener questions are not scripts
    jaccard = lambda pairs: [(len(set(a.split()) & set(b.split())) / len(set(a.split())),) * 3 for a, b in pairs]
    res = ev.evaluate(items, jaccard)["summary"]
    assert res["editor"]["rouge1_p"]["mean"] > res["control"]["rouge1_p"]["mean"]
    assert res["lead-3"]["rouge2_f"]["mean"] > res["editor"]["rouge2_f"]["mean"]   # copying overlaps most
    assert res["editor"]["bert_f"]["mean"] > res["control"]["bert_f"]["mean"]


def test_rouge_skips_russian_only_sources():
    assert ev.rouge("Two stations opened.", [{"text": "Открылись две станции метро."}]) is None
    assert ev.is_english("The metro opened") and not ev.is_english("Открылось метро")
