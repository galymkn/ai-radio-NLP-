"""HTTP server: web player, personal pseudo-stream (/api/next) and a continuous MP3 stream (/stream)."""
from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import audio
from .config import Config
from .db import DB
from .editor import RateLimited
from .ner import summarize
from .pipeline import Pipeline
from .tts import make_tts, render
from .wheel import Listener, needs_station_id, pick

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
STATION_ID = "station_id.mp3"
SILENCE = "silence.mp3"
MAX_LISTENERS = 10_000


def prepare_assets(cfg: Config) -> None:
    clips = cfg.clips_path
    if not (clips / SILENCE).exists():
        audio.silence(clips / SILENCE, 2.0)
    if not (clips / STATION_ID).exists():
        text = f"You're listening to {cfg.station_name}. Automated news, around the clock."
        try:
            render(make_tts(cfg.tts), text, clips, STATION_ID.removesuffix(".mp3"))
        except Exception as e:  # noqa: BLE001
            log.warning("station ID TTS failed (%s); using silence", e)
            audio.silence(clips / STATION_ID, 1.0)


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=300)


def next_quota_reset() -> float:
    """Gemini daily quotas reset at midnight Pacific time; returns that moment as a Unix time."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()


def story_entities(db: DB, story_id) -> dict[str, list[str]]:
    if story_id is None:
        return {}
    return summarize(db.story_entities([story_id]).get(story_id, []))


class Station:
    def __init__(self, cfg: Config, db: DB):
        self.cfg, self.db = cfg, db
        self.lock = threading.RLock()
        self.listeners: OrderedDict[str, Listener] = OrderedDict()

    def listener(self, lid: str) -> Listener:
        if lid not in self.listeners:
            self.listeners[lid] = Listener()
            if not self.cfg.wheel.repeat_stories:
                for row in self.db.listener_history(lid):
                    self.listeners[lid].last_played[row['clip_id']] = row['played_at']
                    self.listeners[lid].plays[row['clip_id']] = 1
            while len(self.listeners) > MAX_LISTENERS:
                self.listeners.popitem(last=False)
        self.listeners.move_to_end(lid)
        return self.listeners[lid]

    def next_for_listener(self, lid):
        with self.lock:
            item = self.next_item(self.listener(lid))
            if item['type'] == 'story':
                self.db.remember_delivery(lid, item['id'], time.time())
            return item

    def next_item(self, lst: Listener) -> dict:
        with self.lock:
            return self._next_item(lst)

    def _next_item(self, lst: Listener) -> dict:
        """Decide what this listener hears next."""
        now = time.time()
        if needs_station_id(lst, self.cfg.wheel):
            lst.stories_since_id = 0
            return {"type": "station_id", "headline": self.cfg.station_name, "audio_file": STATION_ID}
        clip = pick(self.db.active_clips(now), lst, now, self.cfg.wheel)
        if clip is None:
            return {"type": "idle", "retry_after": 10, "audio_file": SILENCE}
        lst.record(clip, now)
        self.db.mark_played(clip.id, now)
        return {"type": "story", **clip.to_dict(), "entities": story_entities(self.db, clip.story_id),
                "audio_file": clip.audio_file}


def create_app(cfg: Config, run_worker: bool | None = None) -> FastAPI:
    run_worker = cfg.server.run_worker if run_worker is None else run_worker
    db = DB(cfg.database_url or cfg.db_path)
    station = Station(cfg, db)

    worker_state = {"enabled": run_worker, "last_success": None, "last_error": None, "editor": None}
    # One pipeline (encoder, NER, editor, TTS) shared by the worker and by listener questions.
    shared = {"pipeline": None}
    build_lock = threading.Lock()

    def get_pipeline() -> Pipeline:
        with build_lock:
            if shared["pipeline"] is None:
                shared["pipeline"] = Pipeline.build(cfg, db)
            return shared["pipeline"]

    # How long the editor waits before looking again, by outcome of the last story.
    editor_pause = {"clip": 0, "no-clip": 0, "error": 5, "idle": 10, "paused": 120}

    async def editor_loop(pipeline, worker_no):
        """Writes and voices stories one after another, while the current clip is on air."""
        while True:
            try:
                outcome = await asyncio.to_thread(pipeline.edit_next)
            except Exception:  # noqa: BLE001
                outcome = "error"
                log.exception("editor step failed")
            worker_state["editor"] = {"last": outcome, "at": time.time(), "worker": worker_no,
                                      "paused": getattr(pipeline, "llm_paused", "") if outcome == "paused" else ""}
            await asyncio.sleep(editor_pause.get(outcome, 5))

    async def worker():
        pipeline, editor_tasks = None, []
        try:
            while True:
                try:
                    if pipeline is None:
                        pipeline = await asyncio.to_thread(get_pipeline)
                        editor_tasks = [asyncio.create_task(editor_loop(pipeline, i))
                                        for i in range(max(1, cfg.editor.workers))]
                    result = await asyncio.to_thread(pipeline.ingest_once)
                    worker_state.update(last_success=time.time(), last_error=None, last_cycle=result)
                except Exception as exc:  # noqa: BLE001
                    worker_state["last_error"] = type(exc).__name__
                    log.exception("ingest cycle failed")
                await asyncio.sleep(cfg.ingest.poll_seconds)
        finally:
            for t in editor_tasks:
                t.cancel()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(prepare_assets, cfg)
        task = asyncio.create_task(worker()) if run_worker else None
        yield
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title=cfg.station_name, lifespan=lifespan)
    app.state.station = station
    app.mount("/audio", StaticFiles(directory=str(cfg.clips_path)), name="audio")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")  # fonts for the player

    @app.get("/")
    def player():
        return FileResponse(STATIC / "player.html")

    @app.get("/api/status")
    def status():
        return {"station": cfg.station_name, **db.stats(), "listeners": len(station.listeners), "worker": worker_state}

    @app.get("/api/clips")
    def clips():
        items = sorted(db.active_clips(), key=lambda c: (-c.priority, -c.created_at))
        return [{**c.to_dict(), "entities": story_entities(db, c.story_id)} for c in items]

    @app.post("/api/ask")
    def ask(body: AskBody):
        """Answer a listener's question from recent reports (retrieve, answer, fact-check, voice)."""
        from .ask import answer_question
        try:
            return answer_question(get_pipeline(), body.question.strip())
        except RateLimited:
            raise HTTPException(429, {"message": "Today's free Gemini quota is used up.",
                                      "reset_at": next_quota_reset()})
        except ValueError:
            raise HTTPException(502, "The answer could not be checked against the reports. Try again.")

    @app.get("/api/next")
    def next_item(listener: str = Query(..., min_length=4, max_length=64)):
        item = station.next_for_listener(listener)
        item["audio_url"] = f"/audio/{item.pop('audio_file')}"
        return JSONResponse(item)

    @app.post("/api/skip")
    def skip(listener: str = Query(..., min_length=4, max_length=64), category: str | None = None):
        with station.lock:
            lst = station.listener(listener)
            if category:
                lst.skipped_categories[category] = lst.skipped_categories.get(category, 0) + 1
        return {"ok": True}

    @app.post("/api/started")
    def started(clip_id: int = Query(..., ge=1), listener: str = Query(..., min_length=4, max_length=64)):
        history = db.listener_history(listener)
        if not any(row['clip_id'] == clip_id for row in history):
            raise HTTPException(404, "Clip has not been delivered to this listener")
        db.record_broadcast(clip_id, time.time(), "player_started")
        return {"ok": True}

    @app.get("/stream")
    async def stream(request: Request):
        """Continuous MP3 for VLC / radio apps. Paced to real time, own wheel state per connection."""
        lst = Listener()
        lead_seconds, chunk = 3.0, 8192

        async def gen():
            start, sent = time.monotonic(), 0
            while not await request.is_disconnected():
                item = await asyncio.to_thread(station.next_item, lst)
                data = await asyncio.to_thread((cfg.clips_path / item["audio_file"]).read_bytes)
                for i in range(0, len(data), chunk):
                    while sent / audio.BYTES_PER_SEC - (time.monotonic() - start) > lead_seconds:
                        await asyncio.sleep(0.2)
                    if i == 0 and item["type"] == "story":
                        await asyncio.to_thread(db.record_broadcast, item["id"], time.time(), "stream_sent")
                    yield data[i:i + chunk]
                    sent += len(data[i:i + chunk])

        return StreamingResponse(gen(), media_type="audio/mpeg",
                                 headers={"Cache-Control": "no-cache", "icy-name": cfg.station_name})

    return app
