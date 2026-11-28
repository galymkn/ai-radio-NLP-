"""NER-based clustering veto and 'Ask the radio' (RAG)."""
import time

import pytest
from fastapi.testclient import TestClient

from newsradio.cluster import HashEmbedder, cluster_new_articles
from newsradio.config import Cluster, Config
from newsradio.db import DB
from newsradio.ingest import RawArticle
from newsradio.ner import NameMatcher, name_key, summarize


class FakeNER:
    """Tags a few known names; enough to test the clustering rule without a 700 MB model."""
    KNOWN = {"Kyiv": "LOC", "Belgorod": "LOC", "Белгород": "LOC", "Москве": "LOC", "Moscow": "LOC",
             "Tokayev": "PER", "Putin": "PER", "Kazatomprom": "ORG"}

    def extract(self, texts):
        return [[(k, w) for w, k in self.KNOWN.items() if w in t] for t in texts]


def art(i, title, text, source="A"):
    return RawArticle(url=f"https://x/{i}", source=source, kind="test", title=title, text=text)


DRONES = [art(1, "Drone attack hits Kyiv overnight", "A drone attack hit Kyiv overnight, officials said.", "BBC"),
          art(2, "Drone attack hits Belgorod overnight", "A drone attack hit Belgorod overnight, officials said.", "TASS")]


def stories_of(db):
    return sorted(s["n_articles"] for s in db.recent_stories(0))


def test_same_kind_of_event_in_different_places_is_merged_without_ner(tmp_path):
    db = DB(tmp_path / "db")
    for a in DRONES:
        db.insert_article(a)
    cluster_new_articles(db, HashEmbedder(), Cluster(embedder="hash", threshold=0.45, entity_veto=True))
    assert stories_of(db) == [2]   # the topic is the same, so cosine alone merges them


def test_place_veto_keeps_them_apart(tmp_path):
    db = DB(tmp_path / "db")
    for a in DRONES:
        db.insert_article(a)
    res = cluster_new_articles(db, HashEmbedder(), Cluster(embedder="hash", threshold=0.45, entity_veto=True),
                               ner=FakeNER())
    assert stories_of(db) == [1, 1] and res["vetoed"] == 1


def test_same_place_still_merges_and_entities_are_stored(tmp_path):
    db = DB(tmp_path / "db")
    db.insert_article(art(1, "Drone attack hits Belgorod overnight", "A drone attack hit Belgorod overnight.", "BBC"))
    cfg = Cluster(embedder="hash", threshold=0.45, entity_veto=True)
    cluster_new_articles(db, HashEmbedder(), cfg, ner=FakeNER())
    db.insert_article(art(2, "Drone attack hits Belgorod overnight", "Белгород: a drone attack hit Belgorod.", "TASS"))
    res = cluster_new_articles(db, HashEmbedder(), cfg, ner=FakeNER())   # incremental: story places come from the DB
    assert res["new_stories"] == 0 and stories_of(db) == [2]
    sid = db.recent_stories(0)[0]["id"]
    assert ("LOC", "Belgorod") in db.story_entities([sid])[sid][0]


def test_veto_off_by_config(tmp_path):
    db = DB(tmp_path / "db")
    for a in DRONES:
        db.insert_article(a)
    cluster_new_articles(db, HashEmbedder(), Cluster(embedder="hash", threshold=0.45, entity_veto=False),
                         ner=FakeNER())
    assert stories_of(db) == [2]


def test_name_matching_across_scripts_and_cases():
    m = NameMatcher(None, 0.75)            # no encoder: spelling rules only
    assert name_key("Белгород") == "belgorod"
    assert m.same("Белгород", "Belgorod")
    assert m.same("Москва", "Москве")      # Russian case ending
    assert not m.same("Kyiv", "Belgorod")
    assert m.conflict({"Kyiv"}, {"Belgorod"})
    assert not m.conflict(set(), {"Belgorod"})   # one side names no place: no evidence, no veto


def test_summary_prefers_latin_spelling_and_counts():
    s = summarize([[("PER", "Токаев"), ("LOC", "Astana")], [("PER", "Tokayev")], [("PER", "Tokayev"), ("ORG", "UN")]])
    assert s["PER"] == ["Tokayev"] and s["LOC"] == ["Astana"] and s["ORG"] == ["UN"]


# ---------------------------------------------------------------- Ask the radio

def station(tmp_path):
    from newsradio.pipeline import Pipeline
    cfg = Config(data_dir=str(tmp_path))
    cfg.cluster.embedder, cfg.cluster.threshold = "hash", 0.45
    cfg.editor.backend, cfg.tts.backend = "fake", "dummy"
    cfg.ask.min_score = 0.05
    cfg.clips_path.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline.build(cfg)
    pipeline.ingest_once([
        art(1, "Almaty metro opens two new stations", "Almaty's metro opened two new stations on Monday. The line grew by three kilometres.", "BBC"),
        art(2, "Central bank keeps base rate unchanged", "The central bank kept its base rate at fifteen percent on Friday.", "Kursiv"),
    ])
    return cfg, pipeline


def test_retrieval_finds_the_relevant_report(tmp_path):
    from newsradio.ask import retrieve
    cfg, p = station(tmp_path)
    hits = retrieve(p.db, p.embedder, "What happened with the Almaty metro?", k=2, window_hours=48, min_score=0.05)
    assert hits[0][1]["title"] == "Almaty metro opens two new stations"


def test_answer_is_grounded_checked_and_voiced(tmp_path):
    from newsradio.ask import answer_question
    cfg, p = station(tmp_path)
    res = answer_question(p, "What happened with the Almaty metro?")
    assert res["found"] and "metro" in res["answer"]
    assert res["sources"][0]["url"] == "https://x/1" and res["audio_url"].startswith("/audio/ask-")
    assert (cfg.clips_path / res["audio_url"].removeprefix("/audio/")).exists()


def test_unsupported_answer_sentences_are_dropped(tmp_path, monkeypatch):
    from newsradio.ask import answer_question
    cfg, p = station(tmp_path)
    monkeypatch.setattr(p.editor, "answer", lambda q, docs, lang: {
        "found": True, "sources": [1], "answer": "The metro opened two stations. It cost ten billion dollars."})
    monkeypatch.setattr(p.editor, "verify", lambda text, docs: ["It cost ten billion dollars."])
    res = answer_question(p, "What happened with the Almaty metro?")
    assert res["answer"] == "The metro opened two stations."


def test_nothing_relevant_means_no_llm_call(tmp_path, monkeypatch):
    from newsradio.ask import answer_question
    cfg, p = station(tmp_path)
    cfg.ask.min_score = 0.99
    monkeypatch.setattr(p.editor, "answer", lambda *a: pytest.fail("LLM must not be called"))
    res = answer_question(p, "Кто выиграл чемпионат мира по хоккею?")
    assert not res["found"] and res["answer"].startswith("В сегодняшних")


def test_ask_endpoint_and_quota(tmp_path, monkeypatch):
    from newsradio.editor import RateLimited
    from newsradio.server import create_app
    cfg, p = station(tmp_path)
    with TestClient(create_app(cfg, run_worker=False)) as client:
        ok = client.post("/api/ask", json={"question": "What happened with the Almaty metro?"})
        assert ok.status_code == 200 and ok.json()["found"]
        assert client.post("/api/ask", json={"question": "x"}).status_code == 422
        import newsradio.ask as ask_mod
        monkeypatch.setattr(ask_mod, "answer_question", lambda *a: (_ for _ in ()).throw(RateLimited("cap")))
        assert client.post("/api/ask", json={"question": "Anything new?"}).status_code == 429
        clips = client.get("/api/clips").json()
        assert all("entities" in c for c in clips)


# ---------------------------------------------------------------- free quota

def test_editors_leave_a_reserve_for_questions(monkeypatch, tmp_path):
    from newsradio import editor as ed_mod
    from newsradio.config import Editor
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ed = ed_mod.GeminiEditor(Editor(backend="gemini", model="m", max_calls_per_day=5, reserve_for_questions=2), "FM")
    for _ in range(3):
        ed._wait_turn()                      # editors: 5 - 2 = 3 calls
    with pytest.raises(ed_mod.RateLimited):
        ed._wait_turn()
    with ed.for_questions():
        ed._wait_turn(); ed._wait_turn()     # questions may use the reserve
        with pytest.raises(ed_mod.RateLimited):
            ed._wait_turn()


def test_daily_count_survives_restart(monkeypatch, tmp_path):
    from newsradio import editor as ed_mod
    from newsradio.config import Editor
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    cfg, path = Editor(backend="gemini", model="m", max_calls_per_day=3), tmp_path / "usage.json"
    first = ed_mod.GeminiEditor(cfg, "FM", usage_path=path)
    first._wait_turn(); first._wait_turn()
    again = ed_mod.GeminiEditor(cfg, "FM", usage_path=path)   # station restarted
    again._wait_turn()
    with pytest.raises(ed_mod.RateLimited):
        again._wait_turn()


def test_quota_message_says_when_it_renews(tmp_path, monkeypatch):
    from newsradio.editor import RateLimited
    from newsradio.server import create_app
    import newsradio.ask as ask_mod
    cfg, p = station(tmp_path)
    monkeypatch.setattr(ask_mod, "answer_question", lambda *a: (_ for _ in ()).throw(RateLimited("cap")))
    with TestClient(create_app(cfg, run_worker=False)) as client:
        r = client.post("/api/ask", json={"question": "Anything new?"})
        assert r.status_code == 429 and r.json()["detail"]["reset_at"] > time.time()
