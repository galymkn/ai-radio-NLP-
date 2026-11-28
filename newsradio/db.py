"""SQLite storage: articles, stories (clusters) and clips (voiced segments)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS edit_attempts (
    id INTEGER PRIMARY KEY,
    story_id INTEGER UNIQUE NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_retry_at REAL NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS article_entities (
    id INTEGER PRIMARY KEY,
    article_id INTEGER UNIQUE NOT NULL,
    entities TEXT NOT NULL            -- JSON [[kind, name], ...], kind = PER | ORG | LOC
);
CREATE TABLE IF NOT EXISTS broadcast_events (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER UNIQUE NOT NULL,
    started_at REAL NOT NULL,
    event_kind TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS listener_history (
    id INTEGER PRIMARY KEY,
    listener TEXT NOT NULL,
    clip_id INTEGER NOT NULL,
    played_at REAL NOT NULL,
    UNIQUE(listener, clip_id)
);
CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    raw_title TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY,
    url           TEXT UNIQUE NOT NULL,
    source        TEXT NOT NULL,
    kind          TEXT NOT NULL,           -- rss | telegram | fixture
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    title_key     TEXT,                    -- normalized title for cheap exact dedup
    published_at  REAL,
    fetched_at    REAL NOT NULL,
    story_id      INTEGER,
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS ix_articles_story ON articles(story_id);
CREATE INDEX IF NOT EXISTS ix_articles_title_key ON articles(title_key);

CREATE TABLE IF NOT EXISTS stories (
    id              INTEGER PRIMARY KEY,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    centroid        BLOB NOT NULL,
    n_articles      INTEGER NOT NULL DEFAULT 0,
    n_sources       INTEGER NOT NULL DEFAULT 0,
    last_edited_at  REAL,
    last_edited_n   INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'new'   -- new | aired | rejected
);

CREATE TABLE IF NOT EXISTS clips (
    id              INTEGER PRIMARY KEY,
    story_id        INTEGER,
    created_at      REAL NOT NULL,
    headline        TEXT NOT NULL,
    script          TEXT NOT NULL,
    category        TEXT,
    priority        INTEGER NOT NULL DEFAULT 3,
    expires_at      REAL NOT NULL,
    audio_file      TEXT NOT NULL,
    duration        REAL NOT NULL DEFAULT 0,
    play_count      INTEGER NOT NULL DEFAULT 0,
    last_played_at  REAL,
    sources         TEXT NOT NULL DEFAULT '[]',
    active          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_clips_active ON clips(active);
"""


@dataclass
class Clip:
    id: int
    story_id: int | None
    created_at: float
    headline: str
    script: str
    category: str | None
    priority: int
    expires_at: float
    audio_file: str
    duration: float
    play_count: int
    last_played_at: float | None
    sources: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "story_id": self.story_id,
            "headline": self.headline,
            "script": self.script,
            "category": self.category,
            "priority": self.priority,
            "duration": round(self.duration, 1),
            "sources": self.sources,
            "created_at": self.created_at,
            "audio_url": f"/audio/{self.audio_file}",
        }


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class DB:
    """Thin thread-safe wrapper. One connection, guarded by a lock (fine for an MVP)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path.startswith(("postgresql://", "postgres://")):
            from .postgres import PostgresConnection
            self.conn = PostgresConnection(self.path, SCHEMA)
        else:
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def listener_history(self, listener):
        with self.lock:
            return self.conn.execute("SELECT clip_id, played_at FROM listener_history WHERE listener=?", (listener,)).fetchall()

    def remember_delivery(self, listener, clip_id, now):
        with self.lock:
            self.conn.execute("INSERT OR IGNORE INTO listener_history(listener, clip_id, played_at) VALUES (?,?,?)",
                              (listener, clip_id, now))
            self.conn.commit()

    # ---------- articles ----------
    def has_url(self, url: str) -> bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM articles WHERE url=?", (url,)).fetchone() is not None

    def has_title_key(self, key: str) -> bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM articles WHERE title_key=?", (key,)).fetchone() is not None

    def insert_article(self, a) -> int | None:
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO raw_items(url, source, kind, raw_title, raw_text, fetched_at) VALUES (?,?,?,?,?,?)",
                (a.url, a.source, a.kind, a.raw_title, a.raw_text, time.time()),
            )
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO articles(url, source, kind, title, text, title_key, published_at, fetched_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (a.url, a.source, a.kind, a.title, a.text, a.title_key, a.published_at, time.time()),
            )
            self.conn.commit()
            return cur.lastrowid if cur.rowcount else None

    def unclustered_articles(self) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                "SELECT * FROM articles WHERE story_id IS NULL ORDER BY COALESCE(published_at, fetched_at)"
            ).fetchall()

    def set_article_story(self, article_id: int, story_id: int, emb: np.ndarray) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE articles SET story_id=?, embedding=? WHERE id=?", (story_id, to_blob(emb), article_id)
            )

    def save_entities(self, article_id: int, entities) -> None:
        with self.lock:
            self.conn.execute("INSERT OR IGNORE INTO article_entities(article_id, entities) VALUES (?,?)",
                              (article_id, json.dumps([list(e) for e in entities], ensure_ascii=False)))

    def story_entities(self, story_ids) -> dict[int, list[list[tuple[str, str]]]]:
        """Entities of every article in the given stories: {story_id: [[(kind, name), ...] per article]}."""
        ids, out = list(story_ids), {}
        with self.lock:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                rows = self.conn.execute(
                    "SELECT a.story_id, e.entities FROM article_entities e JOIN articles a ON a.id = e.article_id "
                    f"WHERE a.story_id IN ({','.join('?' * len(chunk))})", tuple(chunk)).fetchall()
                for r in rows:
                    out.setdefault(r[0], []).append([tuple(e) for e in json.loads(r[1])])
        return out

    def recent_articles_with_embeddings(self, since: float) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                "SELECT id, url, source, title, text, published_at, story_id, embedding FROM articles "
                "WHERE embedding IS NOT NULL AND COALESCE(published_at, fetched_at) >= ?", (since,)).fetchall()

    def story_articles(self, story_id: int) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                "SELECT * FROM articles WHERE story_id=? ORDER BY COALESCE(published_at, fetched_at) DESC",
                (story_id,),
            ).fetchall()

    # ---------- stories ----------
    def recent_stories(self, since: float) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("SELECT * FROM stories WHERE updated_at>=?", (since,)).fetchall()

    def create_story(self, centroid: np.ndarray, now: float) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO stories(created_at, updated_at, centroid, n_articles) VALUES (?,?,?,0)",
                (now, now, to_blob(centroid)),
            )
            return cur.lastrowid

    def update_story(self, story_id: int, centroid: np.ndarray, n_articles: int, now: float) -> None:
        with self.lock:
            n_sources = self.conn.execute(
                "SELECT COUNT(DISTINCT source) FROM articles WHERE story_id=?", (story_id,)
            ).fetchone()[0]
            self.conn.execute(
                "UPDATE stories SET centroid=?, n_articles=?, n_sources=?, updated_at=? WHERE id=?",
                (to_blob(centroid), n_articles, n_sources, now, story_id),
            )

    def stories_to_edit(self, now: float, cooldown_s: float, window_s: float, limit: int,
                        priority_sources: list[str] | tuple = ()) -> list[sqlite3.Row]:
        """Stories with new articles. Stories that include a priority outlet come first, then the
        ones covered by more outlets."""
        marks = ",".join("?" * len(priority_sources)) or "NULL"
        with self.lock:
            return self.conn.execute(
                f"""SELECT * FROM stories
                   WHERE n_articles > last_edited_n AND status != 'failed'
                     AND NOT EXISTS (SELECT 1 FROM edit_attempts a
                         WHERE a.story_id=stories.id AND a.next_retry_at > ?)
                     AND updated_at >= ?
                     AND (last_edited_at IS NULL OR last_edited_at <= ?)
                   ORDER BY (SELECT COUNT(*) FROM articles p
                             WHERE p.story_id=stories.id AND p.source IN ({marks})) > 0 DESC,
                            n_sources DESC, n_articles DESC, updated_at DESC
                   LIMIT ?""",
                (now, now - window_s, now - cooldown_s, *priority_sources, limit),
            ).fetchall()

    def mark_story_edited(self, story_id: int, n_articles: int, status: str, now: float) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM edit_attempts WHERE story_id=?", (story_id,))
            self.conn.execute(
                "UPDATE stories SET last_edited_at=?, last_edited_n=?, status=? WHERE id=?",
                (now, n_articles, status, story_id),
            )
            self.conn.commit()

    def record_edit_failure(self, story_id, max_attempts, backoff_seconds, now, error):
        with self.lock:
            self.conn.execute("INSERT OR IGNORE INTO edit_attempts(story_id) VALUES (?)", (story_id,))
            row = self.conn.execute("SELECT attempts FROM edit_attempts WHERE story_id=?", (story_id,)).fetchone()
            attempts = row[0] + 1
            self.conn.execute("UPDATE edit_attempts SET attempts=?, next_retry_at=?, last_error=? WHERE story_id=?",
                              (attempts, now + backoff_seconds * 2 ** min(attempts-1, 10), error, story_id))
            if attempts >= max_attempts:
                self.conn.execute("UPDATE stories SET status='failed' WHERE id=?", (story_id,))
            self.conn.commit()

    def retry_failed(self):
        """Explicit operator action; new articles alone do not restart an exhausted retry budget."""
        with self.lock:
            self.conn.execute("DELETE FROM edit_attempts WHERE story_id IN (SELECT id FROM stories WHERE status='failed')")
            self.conn.execute("UPDATE stories SET status='new' WHERE status='failed'")
            self.conn.commit()

    def record_broadcast(self, clip_id, now, event_kind):
        with self.lock:
            if self.conn.execute("SELECT 1 FROM clips WHERE id=?", (clip_id,)).fetchone() is None:
                return False
            self.conn.execute("INSERT OR IGNORE INTO broadcast_events(clip_id, started_at, event_kind) VALUES (?,?,?)",
                              (clip_id, now, event_kind))
            self.conn.commit()
            return True

    def commit(self) -> None:
        with self.lock:
            self.conn.commit()

    # ---------- clips ----------
    def latest_clip_for_story(self, story_id: int) -> Clip | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM clips WHERE story_id=? ORDER BY created_at DESC LIMIT 1", (story_id,)
            ).fetchone()
        return _clip(row) if row else None

    def insert_clip(self, *, story_id, headline, script, category, priority, expires_at,
                    audio_file, duration, sources) -> int:
        now = time.time()
        with self.lock:
            # A new version of a story supersedes the old clip.
            if story_id is not None:
                self.conn.execute("UPDATE clips SET active=0 WHERE story_id=? AND active=1", (story_id,))
            cur = self.conn.execute(
                "INSERT INTO clips(story_id, created_at, headline, script, category, priority, expires_at,"
                " audio_file, duration, sources) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (story_id, now, headline, script, category, priority, expires_at, audio_file, duration,
                 json.dumps(sources, ensure_ascii=False)),
            )
            self.conn.commit()
            return cur.lastrowid

    def active_clips(self, now: float | None = None) -> list[Clip]:
        now = now or time.time()
        with self.lock:
            self.conn.execute("UPDATE clips SET active=0 WHERE active=1 AND expires_at<?", (now,))
            self.conn.commit()
            rows = self.conn.execute("SELECT * FROM clips WHERE active=1").fetchall()
        return [_clip(r) for r in rows]

    def mark_played(self, clip_id: int, now: float) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE clips SET play_count=play_count+1, last_played_at=? WHERE id=?", (now, clip_id)
            )
            self.conn.commit()

    def stats(self) -> dict:
        with self.lock:
            q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
            return {
                "articles": q("SELECT COUNT(*) FROM articles"),
                "stories": q("SELECT COUNT(*) FROM stories"),
                "stories_aired": q("SELECT COUNT(*) FROM stories WHERE status='aired'"),
                "clips_active": q("SELECT COUNT(*) FROM clips WHERE active=1"),
                "clips_total": q("SELECT COUNT(*) FROM clips"),
                "audio_seconds_active": round(q("SELECT COALESCE(SUM(duration),0) FROM clips WHERE active=1"), 1),
            }


def _clip(row: sqlite3.Row) -> Clip:
    return Clip(
        id=row["id"], story_id=row["story_id"], created_at=row["created_at"], headline=row["headline"],
        script=row["script"], category=row["category"], priority=row["priority"],
        expires_at=row["expires_at"], audio_file=row["audio_file"], duration=row["duration"],
        play_count=row["play_count"], last_played_at=row["last_played_at"],
        sources=json.loads(row["sources"] or "[]"),
    )
