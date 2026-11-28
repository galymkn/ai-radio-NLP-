"""One processing cycle: ingest -> cluster -> edit -> verify -> voice -> store clip."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass

from .cluster import Embedder, cluster_new_articles, make_embedder
from .config import Config
from .db import DB
from .editor import EditorBackend, RateLimited, SourceDoc, apply_verification, make_editor, split_sentences
from .ingest import RawArticle
from .ingest.rss import fetch_feed
from .ingest.telegram import fetch_channel
from .tts import TTSBackend, make_tts, render

log = logging.getLogger(__name__)


@dataclass
class Pipeline:
    cfg: Config
    db: DB
    embedder: Embedder
    editor: EditorBackend
    tts: TTSBackend

    @classmethod
    def build(cls, cfg: Config, db: DB | None = None) -> "Pipeline":
        pipeline = cls(cfg=cfg, db=db or DB(cfg.database_url or cfg.db_path), embedder=make_embedder(cfg.cluster),
                       editor=make_editor(cfg.editor, cfg.station_name, cfg.language), tts=make_tts(cfg.tts))
        pipeline._claims()
        return pipeline

    def _claims(self):
        """Stories being edited right now, so parallel editor workers never take the same one."""
        if not hasattr(self, "_busy"):
            self._busy, self._busy_lock = set(), threading.Lock()
        return self._busy, self._busy_lock

    # ------------------------------------------------------------ 1. ingest
    def collect(self) -> list[RawArticle]:
        ic = self.cfg.ingest
        items: list[RawArticle] = []
        for src in self.cfg.sources.rss:
            items += fetch_feed(src.name, src.url, max_items=ic.max_items_per_feed,
                                max_age_hours=ic.max_age_hours, fetch_fulltext=ic.fetch_fulltext,
                                is_known=self.db.has_url)
        if ic.telegram_backend == "telethon":
            from .ingest.mtproto import fetch_channels
            items += fetch_channels(self.cfg.sources.telegram, ic, self.db.has_url)
            return items
        if ic.telegram_backend != "web":
            raise ValueError("telegram_backend must be web or telethon")
        for ch in self.cfg.sources.telegram:
            items += fetch_channel(ch, max_age_hours=ic.max_age_hours, min_text_chars=ic.min_text_chars,
                                   is_known=self.db.has_url)
        return items

    def store(self, items: list[RawArticle]) -> int:
        n = 0
        for a in items:
            if len(a.text) < self.cfg.ingest.min_text_chars and len(a.title) < 20:
                continue
            if self.db.insert_article(a):
                n += 1
        return n

    # ------------------------------------------------------------ 3-4. edit + voice
    def _docs(self, story_id: int) -> list[SourceDoc]:
        ec = self.cfg.editor
        rows = self.db.story_articles(story_id)
        seen, docs = set(), []
        for r in rows:  # newest first, one doc per outlet
            if r["source"] in seen:
                continue
            seen.add(r["source"])
            docs.append(SourceDoc(source=r["source"], title=r["title"], text=r["text"][: ec.max_chars_per_source],
                                  url=r["url"]))
            if len(docs) >= ec.max_sources_per_story:
                break
        return docs

    def edit_story(self, story) -> int | None:
        now = time.time()
        sid, n = story["id"], story["n_articles"]
        docs = self._docs(sid)
        prev = self.db.latest_clip_for_story(sid)
        draft = self.editor.write(docs, prev.script if prev else None)

        if not draft.newsworthy:
            log.info("story %d rejected: %s", sid, draft.reason)
            self.db.mark_story_edited(sid, n, "rejected", now)
            return None
        if prev and not draft.has_new_info:
            log.info("story %d: no new info", sid)
            self.db.mark_story_edited(sid, n, "aired", now)
            return None

        script = draft.script
        if self.cfg.editor.verify:
            unsupported = self.editor.verify(script, docs)
            if unsupported:
                script = apply_verification(script, unsupported)
        if len(split_sentences(script)) < 2:
            log.info("story %d: too little verified content left, skipping", sid)
            self.db.mark_story_edited(sid, n, "rejected", now)
            return None

        name = f"s{sid}-{uuid.uuid4().hex[:8]}"
        from .normalize import normalize_speech
        spoken = normalize_speech(script, self.cfg.language) if self.cfg.tts.normalize else script
        audio_file, dur = render(self.tts, spoken, self.cfg.clips_path, name)
        clip_id = self.db.insert_clip(
            story_id=sid, headline=draft.headline, script=script, category=draft.category,
            priority=draft.priority, expires_at=now + draft.ttl_hours * 3600, audio_file=audio_file,
            duration=dur, sources=draft.sources or [d.source for d in docs],
        )
        self.db.mark_story_edited(sid, n, "aired", now)
        log.info("clip %d [p%d %s] %s (%.0fs)", clip_id, draft.priority, draft.category, draft.headline, dur)
        return clip_id

    def edit_next(self) -> str:
        """Edit the most important waiting story.
        Returns "clip", "no-clip" (rejected / nothing new), "idle" (queue empty), "paused" (quota) or "error"."""
        ec = self.cfg.editor
        busy, lock = self._claims()
        with lock:
            stories = self.db.stories_to_edit(time.time(), ec.reedit_cooldown_minutes * 60,
                                              self.cfg.cluster.window_hours * 3600, 1 + len(busy),
                                              ec.priority_sources)
            s = next((row for row in stories if row["id"] not in busy), None)
            if s is None:
                return "idle"
            busy.add(s["id"])
        try:
            return "clip" if self.edit_story(s) else "no-clip"
        except RateLimited as exc:  # free quota used up: not the story's fault, try later
            self.llm_paused = str(exc)[:200]
            log.warning("LLM quota reached, editing paused: %s", exc)
            return "paused"
        except Exception as exc:  # one bad story must not stop the station
            self.db.record_edit_failure(s["id"], ec.max_attempts, ec.retry_backoff_seconds,
                                        time.time(), type(exc).__name__)
            log.exception("editing story %d failed", s["id"])
            return "error"
        finally:
            with lock:
                busy.discard(s["id"])

    def edit_pending(self) -> int:
        made = 0
        self.llm_paused = ""
        for _ in range(self.cfg.editor.max_edits_per_cycle):
            outcome = self.edit_next()
            if outcome == "clip":
                made += 1
            elif outcome in ("idle", "paused"):
                break
        return made

    # ------------------------------------------------------------ full cycle
    def ingest_once(self, items: list[RawArticle] | None = None) -> dict:
        """Fetch, store and cluster. The live server runs this every poll_seconds; editing runs separately."""
        t0 = time.time()
        items = self.collect() if items is None else items
        stored = self.store(items)
        cl = cluster_new_articles(self.db, self.embedder, self.cfg.cluster)
        return {"fetched": len(items), "stored": stored, **cl, "seconds": round(time.time() - t0, 1)}

    def run_cycle(self, items: list[RawArticle] | None = None) -> dict:
        t0 = time.time()
        stats = self.ingest_once(items)
        stats["clips"] = self.edit_pending()
        stats["seconds"] = round(time.time() - t0, 1)
        if getattr(self, "llm_paused", ""):
            stats["llm_paused"] = self.llm_paused
        log.info("cycle done: %s", stats)
        return stats
