"""The news wheel: decides what plays next.

Each story is voiced once and kept as a clip. The wheel rotates active clips by
score = priority x freshness / (1 + times this listener heard it), with a per-listener
cooldown. Listener state is per connection, so every listener gets their own order
while TTS cost stays flat (clips are shared).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .config import Wheel as WheelCfg
from .db import Clip


@dataclass
class Listener:
    last_played: dict[int, float] = field(default_factory=dict)
    plays: dict[int, int] = field(default_factory=dict)
    stories_since_id: int = 0
    last_category: str | None = None
    skipped_categories: dict[str, int] = field(default_factory=dict)

    def record(self, clip: Clip, now: float) -> None:
        self.last_played[clip.id] = now
        self.plays[clip.id] = self.plays.get(clip.id, 0) + 1
        self.stories_since_id += 1
        self.last_category = clip.category


def score(clip: Clip, listener: Listener, now: float, cfg: WheelCfg) -> float:
    age_h = max(0.0, (now - clip.created_at) / 3600)
    freshness = math.pow(0.5, age_h / max(cfg.half_life_hours, 0.1))
    s = clip.priority * freshness / (1 + listener.plays.get(clip.id, 0))
    if clip.id not in listener.last_played:
        s *= 1.5
    if clip.category and clip.category == listener.last_category:
        s *= 0.8  # mix topics
    s *= 0.7 ** listener.skipped_categories.get(clip.category or "", 0)
    if clip.category in cfg.boost_categories:
        s *= 1.5
    return s


def pick(clips: list[Clip], listener: Listener, now: float | None, cfg: WheelCfg) -> Clip | None:
    now = now or time.time()
    if not clips:
        return None
    if not cfg.repeat_stories:
        clips = [c for c in clips if c.id not in listener.last_played]
        if not clips:
            return None
    cooldown = cfg.cooldown_minutes * 60
    fresh = [c for c in clips if now - listener.last_played.get(c.id, -1e12) >= cooldown]
    if fresh:
        return max(fresh, key=lambda c: (score(c, listener, now, cfg), c.created_at))
    # Everything was heard recently: repeat the least recently heard one, but not back-to-back.
    if cfg.fill_silence:
        # Live radio never goes quiet: replay at once, skipping only the clip that has just played.
        just_played = max(listener.last_played, key=listener.last_played.get) if listener.last_played else None
        candidates = [c for c in clips if c.id != just_played] or clips
        return min(candidates, key=lambda c: listener.last_played.get(c.id, 0))
    oldest = min(clips, key=lambda c: listener.last_played.get(c.id, 0))
    if now - listener.last_played.get(oldest.id, 0) >= min(cooldown, 300):
        return oldest
    return None


def needs_station_id(listener: Listener, cfg: WheelCfg) -> bool:
    return cfg.station_id_every > 0 and listener.stories_since_id >= cfg.station_id_every
