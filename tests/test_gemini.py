"""Gemini editor: request shape, JSON parsing, fail-closed verification, free-tier limits."""
import json

import httpx
import pytest

from newsradio import editor as ed_mod
from newsradio.config import Config, Editor
from newsradio.db import DB
from newsradio.editor import (GeminiEditor, RateLimited, SourceDoc, parse_gemini_json, split_sentences,
                              to_gemini_schema, WRITE_TOOL)
from newsradio.ingest import RawArticle
from newsradio.pipeline import Pipeline

DOCS = [SourceDoc("BBC", "Metro opens", "The metro opened two stations.", "https://bbc.example/1")]
SCRIPT = {"newsworthy": True, "has_new_info": True, "reason": "r", "headline": "Metro opens",
          "category": "world", "priority": 9, "ttl_hours": 6, "script": "One fact. Two facts.", "sources": ["BBC"]}


def answer(obj, status=200):
    body = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(obj)}]}}]}
    return httpx.Response(status, json=body)


@pytest.fixture
def editor(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(ed_mod.time, "sleep", lambda s: None)
    return GeminiEditor(Editor(backend="gemini", model="gemini-test"), "Test FM")


def test_write_and_verify(monkeypatch, editor):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json))
        if "SCRIPT SENTENCES" in json["contents"][0]["parts"][0]["text"]:
            return answer({"verdicts": [{"index": 0, "supported": True}, {"index": 1, "supported": False}]})
        return answer(SCRIPT)

    monkeypatch.setattr(ed_mod.httpx, "post", fake_post)
    draft = editor.write(DOCS, previous_script="old script")
    assert draft.priority == 5  # clamped from 9
    assert draft.headline == "Metro opens" and draft.sources == ["BBC"]
    url, headers, body = calls[0]
    assert url.endswith("/models/gemini-test:generateContent")
    assert headers == {"x-goog-api-key": "test-key"}  # key in a header, never in the URL
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseSchema"]["type"] == "OBJECT"
    assert "PREVIOUSLY AIRED SCRIPT" in body["contents"][0]["parts"][0]["text"]
    assert "submit_script" not in body["systemInstruction"]["parts"][0]["text"]
    assert editor.verify("One fact. Two facts.", DOCS) == ["Two facts."]


def test_incomplete_verification_fails_closed(monkeypatch, editor):
    monkeypatch.setattr(ed_mod.httpx, "post",
                        lambda *a, **kw: answer({"verdicts": [{"index": 0, "supported": True}]}))
    with pytest.raises(ValueError):
        editor.verify("One fact. Two facts.", DOCS)


def test_schema_conversion_drops_min_max():
    schema = to_gemini_schema(WRITE_TOOL["input_schema"])
    assert schema["properties"]["priority"] == {"type": "INTEGER"}
    assert schema["properties"]["sources"]["items"] == {"type": "STRING"}
    assert "category" in schema["required"]


def test_blocked_or_cut_answers_are_errors():
    with pytest.raises(ValueError):
        parse_gemini_json({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ValueError):
        parse_gemini_json({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]})


def test_daily_quota_raises_rate_limited(monkeypatch, editor):
    daily = {"error": {"code": 429, "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProject"}]}]}}
    monkeypatch.setattr(ed_mod.httpx, "post", lambda *a, **kw: httpx.Response(429, json=daily))
    with pytest.raises(RateLimited):
        editor.write(DOCS, None)


def test_short_rate_limit_is_retried(monkeypatch, editor):
    replies = [httpx.Response(429, json={"error": {"details": [{"retryDelay": "3s"}]}}), answer(SCRIPT)]
    monkeypatch.setattr(ed_mod.httpx, "post", lambda *a, **kw: replies.pop(0))
    assert editor.write(DOCS, None).headline == "Metro opens"


def test_own_daily_cap(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(ed_mod.httpx, "post", lambda *a, **kw: answer(SCRIPT))
    editor = GeminiEditor(Editor(backend="gemini", model="m", max_calls_per_day=1), "FM")
    editor.write(DOCS, None)
    with pytest.raises(RateLimited):
        editor.write(DOCS, None)


def test_quota_does_not_burn_story_attempts(tmp_path, monkeypatch):
    cfg = Config(data_dir=str(tmp_path))
    cfg.cluster.embedder, cfg.editor.backend, cfg.tts.backend = "hash", "fake", "dummy"
    cfg.clips_path.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline.build(cfg, DB(tmp_path / "db"))

    def quota(*a, **kw):
        raise RateLimited("quota")

    monkeypatch.setattr(pipeline.editor, "write", quota)
    items = [RawArticle(url=f"https://x/{i}", source="A", kind="test", title=f"Story number {i} about things",
                        text=f"Completely different text {i} " * 10) for i in range(3)]
    stats = pipeline.run_cycle(items)
    assert stats["clips"] == 0 and stats["llm_paused"] == "quota"
    with pipeline.db.lock:
        assert pipeline.db.conn.execute("SELECT COUNT(*) FROM edit_attempts").fetchone()[0] == 0
        assert pipeline.db.conn.execute("SELECT COUNT(*) FROM stories WHERE status='failed'").fetchone()[0] == 0


def test_split_keeps_abbreviations():
    assert split_sentences("The U.S. Senate passed the bill. It now goes to Dr. Smith.") == [
        "The U.S. Senate passed the bill.", "It now goes to Dr. Smith."]
    assert len(split_sentences("Открылись станции. Линия выросла.")) == 2
