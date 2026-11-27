import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from newsradio.cluster import HashEmbedder, cluster_new_articles
from newsradio.config import Cluster, Wheel, load_config
from newsradio.db import DB, Clip
from newsradio.editor import ClaudeEditor, SourceDoc, apply_verification, split_sentences
from newsradio.ingest import RawArticle, clean_text, title_key
from newsradio.ingest.telegram import parse_channel_html
from newsradio.wheel import Listener, needs_station_id, pick

ROOT = Path(__file__).resolve().parent.parent


def art(url, title, text, source="A"):
    return RawArticle(url=url, source=source, kind="test", title=title, text=text)


# ---------------------------------------------------------------- ingest helpers

def test_clean_text_strips_html_urls_emoji():
    assert clean_text("<b>Hello</b> 🔥 world https://x.y/z &amp; more") == "Hello world & more"


def test_title_key_normalizes():
    assert title_key("Breaking: Metro Opens!") == title_key("breaking metro opens")


def test_parse_telegram_preview():
    html = """
    <div class="tgme_widget_message" data-post="chan/42">
      <div class="tgme_widget_message_text">First line headline<br/>Second line with details.</div>
      <time datetime="2026-11-23T10:00:00+00:00"></time>
    </div>
    <div class="tgme_widget_message" data-post="chan/43"></div>
    """
    items = parse_channel_html("chan", html)
    assert len(items) == 1
    a = items[0]
    assert a.url == "https://t.me/chan/42"
    assert a.title == "First line headline"
    assert "Second line" in a.text
    assert a.source == "@chan"
    assert a.published_at is not None


# ---------------------------------------------------------------- clustering

def test_clustering_groups_same_story(tmp_path):
    db = DB(tmp_path / "t.db")
    for a in [
        art("u1", "Metro opens two new stations in Almaty", "The Almaty metro opened two new stations."),
        art("u2", "Almaty metro opens two new stations", "Two new stations opened in the Almaty metro.", "B"),
        art("u3", "Central bank holds interest rate", "The central bank kept the base rate unchanged."),
    ]:
        db.insert_article(a)
    res = cluster_new_articles(db, HashEmbedder(), Cluster(embedder="hash", threshold=0.45))
    assert res["clustered"] == 3 and res["new_stories"] == 2
    stories = {s["id"]: s for s in db.recent_stories(0)}
    sizes = sorted(s["n_articles"] for s in stories.values())
    assert sizes == [1, 2]
    two = next(s for s in stories.values() if s["n_articles"] == 2)
    assert two["n_sources"] == 2


def test_clustering_is_incremental(tmp_path):
    db = DB(tmp_path / "t.db")
    cfg = Cluster(embedder="hash", threshold=0.45)
    emb = HashEmbedder()
    db.insert_article(art("u1", "Metro opens two new stations in Almaty", "The Almaty metro opened two new stations."))
    cluster_new_articles(db, emb, cfg)
    db.insert_article(art("u2", "Almaty metro opens two new stations", "Two new stations opened in Almaty.", "B"))
    res = cluster_new_articles(db, emb, cfg)
    assert res["new_stories"] == 0
    assert db.stories_to_edit(time.time(), 0, 3600, 10)[0]["n_articles"] == 2


# ---------------------------------------------------------------- editor helpers

def test_split_and_verify_filter():
    s = "The metro opened. It cost 2.5 billion tenge. Fares stay the same."
    assert len(split_sentences(s)) == 3
    assert apply_verification(s, ["It cost 2.5 billion tenge."]) == "The metro opened. Fares stay the same."


def test_split_sentences_cyrillic():
    assert len(split_sentences("Открылись станции. Линия выросла.")) == 2


def test_claude_editor_parses_tool_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    ed = ClaudeEditor(load_config(ROOT / "config.yaml").editor, "Test FM")
    payload = {"newsworthy": True, "has_new_info": True, "reason": "r", "headline": "H", "category": "world",
               "priority": 4, "ttl_hours": 6, "script": "One. Two.", "sources": ["BBC"]}
    calls = []

    def fake_create(**kw):
        calls.append(kw)
        if kw["tool_choice"]["name"] == "submit_script":
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=payload)])
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={
            "verdicts": [{"index": 0, "supported": True}, {"index": 1, "supported": False}]})])

    ed.client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    docs = [SourceDoc("BBC", "t", "text", "u")]
    out = ed.write(docs, previous_script="old")
    assert out.priority == 4 and out.sources == ["BBC"]
    assert "PREVIOUSLY AIRED SCRIPT" in calls[0]["messages"][0]["content"]
    assert ed.verify("One. Two.", docs) == ["Two."]


# ---------------------------------------------------------------- wheel

def clip(i, priority=3, age_h=0.0, category="world"):
    now = time.time()
    return Clip(id=i, story_id=i, created_at=now - age_h * 3600, headline=f"h{i}", script="s", category=category,
                priority=priority, expires_at=now + 3600, audio_file=f"{i}.mp3", duration=10, play_count=0,
                last_played_at=None, sources=[])


def test_wheel_prefers_priority_and_freshness():
    cfg = Wheel(cooldown_minutes=20, half_life_hours=3, station_id_every=0)
    clips = [clip(1, priority=2), clip(2, priority=5), clip(3, priority=5, age_h=12)]
    assert pick(clips, Listener(), None, cfg).id == 2


def test_wheel_cooldown_and_rotation():
    cfg = Wheel(cooldown_minutes=20, half_life_hours=3, station_id_every=2, repeat_stories=True,
                fill_silence=False)
    clips = [clip(1, 5), clip(2, 3, category="tech")]
    lst, now = Listener(), time.time()
    order = []
    for _ in range(2):
        c = pick(clips, lst, now, cfg)
        lst.record(c, now)
        order.append(c.id)
    assert order == [1, 2]
    assert needs_station_id(lst, cfg)
    assert pick(clips, lst, now + 60, cfg) is None          # everything heard < 5 min ago
    assert pick(clips, lst, now + 21 * 60, cfg) is not None  # cooldown passed


# ---------------------------------------------------------------- end-to-end (offline)

def test_pipeline_offline(tmp_path):
    from newsradio.pipeline import Pipeline

    cfg = load_config(ROOT / "config.yaml")
    cfg.data_dir = str(tmp_path)
    cfg.clips_path.mkdir(parents=True, exist_ok=True)
    cfg.cluster.embedder, cfg.cluster.threshold = "hash", 0.45
    cfg.editor.backend, cfg.tts.backend = "fake", "dummy"
    cfg.editor.max_edits_per_cycle = 20  # all fixture stories in one cycle (live config is tuned for free tiers)
    items = [RawArticle(**x) for x in json.loads((ROOT / "fixtures/sample_articles.json").read_text())]
    pl = Pipeline.build(cfg)
    stats = pl.run_cycle(items)
    assert stats["stored"] == len(items)
    clips = pl.db.active_clips()
    assert len(clips) >= 3
    for c in clips:
        assert (cfg.clips_path / c.audio_file).exists() and c.duration > 1
    # A second cycle with the same items stores nothing and makes no new clips.
    again = pl.run_cycle(items)
    assert again["stored"] == 0 and again["clips"] == 0
