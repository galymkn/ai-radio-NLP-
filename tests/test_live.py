"""Live radio: no long silences, Kazakhstan sources first, editing separate from ingest."""
import time

from newsradio.config import Config, Wheel
from newsradio.db import DB, Clip
from newsradio.ingest import RawArticle
from newsradio.pipeline import Pipeline
from newsradio.wheel import Listener, pick


def clip(i, priority=3, category="world"):
    now = time.time()
    return Clip(id=i, story_id=i, created_at=now, headline=f"h{i}", script="s", category=category,
                priority=priority, expires_at=now + 3600, audio_file=f"{i}.mp3", duration=40, play_count=0,
                last_played_at=None, sources=[])


def test_no_silence_when_everything_was_heard():
    cfg = Wheel(cooldown_minutes=20, station_id_every=0, repeat_stories=True, fill_silence=True)
    clips = [clip(1, 5), clip(2, 3), clip(3, 2)]
    lst, now = Listener(), time.time()
    played = []
    for step in range(7):  # 3 fresh clips, then replays straight away
        c = pick(clips, lst, now + step * 40, cfg)
        assert c is not None
        lst.record(c, now + step * 40)
        played.append(c.id)
    assert played[:3] == [1, 2, 3]
    assert all(a != b for a, b in zip(played, played[1:]))  # never the same clip twice in a row
    assert played[3] == 1  # the least recently heard comes back first


def test_single_clip_still_plays():
    cfg = Wheel(cooldown_minutes=20, station_id_every=0, repeat_stories=True, fill_silence=True)
    lst, now = Listener(), time.time()
    lst.record(clip(1), now)
    assert pick([clip(1)], lst, now + 5, cfg).id == 1


def test_boosted_category_wins_a_tie():
    cfg = Wheel(station_id_every=0, boost_categories=["kazakhstan"])
    assert pick([clip(1, 3, "world"), clip(2, 3, "kazakhstan")], Listener(), None, cfg).id == 2


def article(i, source, title, text):
    return RawArticle(url=f"https://x/{i}", source=source, kind="test", title=title, text=text)


def test_priority_sources_are_edited_first(tmp_path):
    db = DB(tmp_path / "db")
    pipeline = Pipeline.build(_cfg(tmp_path, priority=["Astana Times"]), db)
    pipeline.ingest_once([
        article(1, "BBC", "Storm hits the coast of Portugal", "A strong storm hit the coast of Portugal. " * 6),
        article(2, "Guardian", "Storm hits the coast of Portugal today", "A strong storm hit the coast of Portugal. " * 6),
        article(3, "Astana Times", "Astana opens new bridge over Ishim", "A new bridge over the Ishim river opened in Astana. " * 6),
    ])
    first = db.stories_to_edit(time.time(), 0, 3600, 1, ["Astana Times"])[0]
    assert {a["source"] for a in db.story_articles(first["id"])} == {"Astana Times"}
    # Without priority the two-outlet story would come first.
    plain = db.stories_to_edit(time.time(), 0, 3600, 1)[0]
    assert plain["n_sources"] == 2


def test_ingest_and_edit_are_separate_steps(tmp_path):
    pipeline = Pipeline.build(_cfg(tmp_path), DB(tmp_path / "db"))
    stats = pipeline.ingest_once([article(1, "BBC", "Metro opens two stations", "The metro opened two stations. " * 6)])
    assert stats["stored"] == 1 and "clips" not in stats
    assert pipeline.db.active_clips() == []
    assert pipeline.edit_next() == "clip"
    assert pipeline.edit_next() == "idle"
    assert len(pipeline.db.active_clips()) == 1


def _cfg(tmp_path, priority=()):
    cfg = Config(data_dir=str(tmp_path))
    cfg.cluster.embedder, cfg.cluster.threshold = "hash", 0.45
    cfg.editor.backend, cfg.tts.backend = "fake", "dummy"
    cfg.editor.priority_sources = list(priority)
    cfg.clips_path.mkdir(parents=True, exist_ok=True)
    return cfg
