import json
import pytest
from fastapi.testclient import TestClient
from newsradio.config import Config
from newsradio.db import DB
from newsradio.editor import ClaudeEditor
from newsradio.evaluate import evaluate
from newsradio.ingest import RawArticle
from newsradio.normalize import normalize_speech
from newsradio.pipeline import Pipeline

@pytest.mark.parametrize('verdicts', [[], [{'index': 0, 'supported': True}],
    [{'index': 0, 'supported': True}, {'index': 0, 'supported': True}],
    [{'index': 0, 'supported': 'true'}, {'index': 1, 'supported': True}]])
def test_incomplete_verification_fails_closed(verdicts):
    ed = ClaudeEditor.__new__(ClaudeEditor)
    ed._call_tool = lambda *args, **kw: {'verdicts': verdicts}
    with pytest.raises(ValueError):
        ed.verify('First fact. Second fact.', [])

def test_raw_storage_and_independent_sources(tmp_path):
    pl = Pipeline.__new__(Pipeline)
    pl.cfg, pl.db = Config(), DB(tmp_path / 'db')
    articles = [RawArticle('u' + str(i), name, 'fixture', 'A sufficiently long identical headline',
                           '<b>Original</b> source text.') for i, name in enumerate(['A', 'B'])]
    assert pl.store(articles) == 2
    assert pl.store(articles) == 0
    raw = pl.db.conn.execute('SELECT * FROM raw_items ORDER BY id').fetchall()
    assert len(raw) == 2 and raw[0]['raw_text'].startswith('<b>')

def test_normalize():
    assert normalize_speech('UN reported 12 cases.', 'en') == 'United Nations reported twelve cases.'
    assert 'двадцать' in normalize_speech('В РК 20 случаев.', 'ru')
    assert normalize_speech('12 жағдай', 'kk') == '12 жағдай'

def test_human_metrics():
    data = {'articles': [{'id': 'a', 'gold_story': 1, 'predicted_story': 1},
                         {'id': 'b', 'gold_story': 1, 'predicted_story': 2}],
            'sentences': [{'supported': True}, {'supported': False}],
            'broadcasts': [{'published_at': 10, 'broadcast_at': 40}]}
    result = evaluate(data)
    assert result['dedup_recall'] == 0
    assert result['dedup_precision'] is None
    assert result['unsupported_sentence_rate'] == .5
    assert result['latency_p95_seconds'] == 30
    assert evaluate({})['unsupported_sentence_rate'] is None

def test_player_api_and_episode(tmp_path):
    from newsradio.server import create_app
    from newsradio.episode import export_episode
    cfg = Config(data_dir=str(tmp_path))
    cfg.clips_path.mkdir()
    cfg.cluster.embedder = 'hash'
    cfg.editor.backend = 'fake'
    cfg.tts.backend = 'dummy'
    pl = Pipeline.build(cfg)
    stats = pl.run_cycle([RawArticle('https://example.test/1', 'Example', 'fixture',
        'Metro opens stations', 'Two metro stations opened today. The new line serves the northern district.')])
    assert stats['clips'] == 1
    result = export_episode(pl.db, cfg.clips_path, tmp_path / 'episode.mp3')
    assert result['seconds'] > 1
    manifest = json.loads((tmp_path / 'episode.json').read_text())
    assert manifest[0]['source_urls'] == ['https://example.test/1']
    with TestClient(create_app(cfg, run_worker=False)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/api/status').json()['clips_total'] == 1
        first = client.get('/api/next?listener=tester').json()
        assert first['type'] == 'story'
        assert client.get(first['audio_url']).status_code == 200
        assert client.get('/api/next?listener=tester').json()['type'] == 'idle'


def test_delivery_memory_survives_station_restart(tmp_path):
    from newsradio.server import Station
    cfg = Config(data_dir=str(tmp_path))
    db = DB(tmp_path / 'db')
    db.remember_delivery('listener', 42, 100)
    station = Station(cfg, db)
    assert station.listener('listener').last_played[42] == 100


def test_postgres_sql_adapter(monkeypatch):
    from types import SimpleNamespace
    psycopg = pytest.importorskip("psycopg")
    from newsradio.postgres import PostgresConnection, Row, row_factory
    statements = []
    class Connection:
        def execute(self, sql, params=()):
            statements.append((sql, params))
            return SimpleNamespace(rowcount=1, fetchone=lambda: Row(id=7))
        def commit(self):
            pass
    monkeypatch.setattr(psycopg, 'connect', lambda *a, **kw: Connection())
    from newsradio.db import SCHEMA
    db = PostgresConnection('postgresql://unused', SCHEMA)
    result = db.execute('INSERT OR IGNORE INTO articles(url) VALUES (?)', ('url',))
    assert result.lastrowid == 7
    assert statements[-1] == ('INSERT INTO articles(url) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id', ('url',))
    assert any('BIGSERIAL PRIMARY KEY' in sql for sql, _ in statements)
    assert row_factory(SimpleNamespace(description=None))([]) == {}
