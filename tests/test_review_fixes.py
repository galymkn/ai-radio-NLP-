import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from newsradio.config import Config
from newsradio.db import DB
from newsradio.pipeline import Pipeline
from newsradio.ingest import RawArticle
from newsradio.editor import FakeEditor
from newsradio.label_export import export_labels
from newsradio.evaluate import evaluate


def test_failed_story_backoff_budget_and_no_starvation(tmp_path):
    db = DB(tmp_path / 'db')
    cfg = Config()
    cfg.editor.max_edits_per_cycle = 1
    cfg.editor.retry_backoff_seconds = 0
    cfg.editor.max_attempts = 3
    now = time.time()
    bad = db.create_story(np.ones(2), now)
    good = db.create_story(np.ones(2), now)
    db.update_story(bad, np.ones(2), 10, now)
    db.update_story(good, np.ones(2), 1, now)
    db.commit()
    pl = Pipeline(cfg, db, None, None, None)
    calls = []
    def edit(story):
        calls.append(story['id'])
        if story['id'] == bad:
            raise ValueError('Incomplete verification')
        db.mark_story_edited(good, 1, 'aired', now)
        return 42
    pl.edit_story = edit
    for _ in range(6):
        pl.edit_pending()
    assert calls == [bad, bad, bad, good]
    assert db.conn.execute('SELECT status FROM stories WHERE id=?', (bad,)).fetchone()[0] == 'failed'
    # Budget survives a new DB connection and additional articles.
    db2 = DB(tmp_path / 'db')
    db2.update_story(bad, np.ones(2), 11, now)
    db2.commit()
    assert db2.stories_to_edit(now, 0, 3600, 10) == []
    db2.retry_failed()
    assert db2.stories_to_edit(now, 0, 3600, 10)[0]['id'] == bad
    db2.record_edit_failure(bad, 3, 120, now, 'ValueError')
    assert db2.stories_to_edit(now + 119, 0, 3600, 10) == []
    assert db2.stories_to_edit(now + 121, 0, 3600, 10)[0]['id'] == bad


@pytest.mark.parametrize('normalize', [False, True])
def test_normalization_is_explicit_opt_in(tmp_path, monkeypatch, normalize):
    import newsradio.pipeline as module
    cfg = Config(data_dir=str(tmp_path))
    cfg.tts.normalize = normalize
    db = DB(tmp_path / 'db')
    aid = db.insert_article(RawArticle('u', 'A', 'test', 'Headline', 'In 2026 we opened. It cost $2.5 billion.'))
    sid = db.create_story(np.ones(2), time.time())
    db.set_article_story(aid, sid, np.ones(2))
    db.update_story(sid, np.ones(2), 1, time.time())
    db.commit()
    captured = []
    monkeypatch.setattr(module, 'render', lambda tts, text, *args: (captured.append(text) or 'fake.mp3', 10))
    pl = Pipeline(cfg, db, None, FakeEditor(cfg.editor, 'test'), None)
    pl.edit_story(db.recent_stories(0)[0])
    assert ('2026' in captured[0]) is not normalize
    if not normalize:
        assert '$2.5 billion' in captured[0]


def test_demo_recovers_from_expired_previous_run(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('data_dir: ' + str(tmp_path / 'data') + '\n')
    command = [sys.executable, '-m', 'newsradio', 'demo', '--no-serve', '--config', str(config)]
    subprocess.run(command, check=True, capture_output=True)
    first = next((tmp_path / 'data/demo').glob('run-*'))
    db = DB(first / 'newsradio.db')
    db.conn.execute('UPDATE clips SET expires_at=0')
    db.commit()
    db.conn.close()
    subprocess.run(command, check=True, capture_output=True)
    runs = list((tmp_path / 'data/demo').glob('run-*'))
    assert len(runs) == 2
    second = next(r for r in runs if r != first)
    assert (second / 'episode.mp3').stat().st_size > 0
    assert len(DB(second / 'newsradio.db').active_clips()) == 5


def test_export_unlabelled_and_first_start_not_delivery(tmp_path):
    from fastapi.testclient import TestClient
    from newsradio.server import create_app
    cfg = Config(data_dir=str(tmp_path))
    cfg.clips_path.mkdir()
    cfg.tts.backend = 'dummy'
    db = DB(cfg.db_path)
    now = time.time()
    article = RawArticle('https://example.test/1', 'A', 'test', 'Headline', 'First fact. Second fact.', now - 100)
    aid = db.insert_article(article)
    sid = db.create_story(np.ones(2), now)
    db.set_article_story(aid, sid, np.ones(2))
    db.update_story(sid, np.ones(2), 1, now)
    cid = db.insert_clip(story_id=sid, headline='H', script='First fact. Second fact.', category='world',
                         priority=3, expires_at=now+3600, audio_file='test.mp3', duration=5, sources=['A'])
    output = tmp_path / 'labels.json'
    with TestClient(create_app(cfg, run_worker=False)) as client:
        assert client.post(f'/api/started?listener=tester&clip_id={cid}').status_code == 404
        client.get('/api/next?listener=tester')
        export_labels(db, output)
        assert json.loads(output.read_text())['broadcasts'] == []
        assert client.post(f'/api/started?listener=tester&clip_id={cid}').status_code == 200
        original = db.conn.execute('SELECT started_at FROM broadcast_events').fetchone()[0]
        client.post(f'/api/started?listener=tester&clip_id={cid}')
        assert db.conn.execute('SELECT started_at FROM broadcast_events').fetchone()[0] == original
    export_labels(db, output)
    labels = json.loads(output.read_text())
    assert labels['articles'][0]['gold_story'] is None
    assert labels['sentences'][0]['source_documents'][0]['text'] == article.text
    result = evaluate(labels)
    assert result['labelled_articles'] == 0 and result['unlabelled_articles'] == 1
    assert result['unsupported_sentence_rate'] is None
    assert result['latency_samples'] == 1
    assert result['latency_p50_seconds'] >= 100
