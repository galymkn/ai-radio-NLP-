"""Configuration loading (YAML + env overrides)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RssSource:
    name: str
    url: str


@dataclass
class Sources:
    rss: list[RssSource] = field(default_factory=list)
    telegram: list[str] = field(default_factory=list)


@dataclass
class Ingest:
    telegram_backend: str = "web"
    poll_seconds: int = 120
    max_items_per_feed: int = 20
    max_age_hours: float = 24
    fetch_fulltext: bool = True
    min_text_chars: int = 80


@dataclass
class Cluster:
    embedder: str = "sbert"
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    threshold: float = 0.72
    window_hours: float = 48
    entity_veto: bool = False       # don't join a story whose places differ from the article's (needs NER)
    entity_match: float = 0.75      # name similarity that counts as the same place ("Киев" ~ "Kyiv")


@dataclass
class Ner:
    backend: str = "none"           # transformers | none
    model: str = "Babelscape/wikineural-multilingual-ner"
    min_score: float = 0.6


@dataclass
class Editor:
    backend: str = "claude"
    model: str = "claude-haiku-5-5"
    verify: bool = True
    max_edits_per_cycle: int = 8
    workers: int = 1                        # live server: stories written in parallel
    min_seconds_between_calls: float = 0    # free tiers limit requests per minute
    max_calls_per_day: int = 0              # own daily cap on LLM calls (0 = no cap); resets on restart
    max_attempts: int = 3
    retry_backoff_seconds: float = 120
    reedit_cooldown_minutes: float = 30
    max_sources_per_story: int = 6
    max_chars_per_source: int = 1800
    script_words: str = "120-180"           # length of one on-air script
    priority_sources: list[str] = field(default_factory=list)  # stories with these outlets are edited first


@dataclass
class TTS:
    backend: str = "edge"
    normalize: bool = False
    voice: str = "en-US-AndrewNeural"
    rate: str = "+0%"


@dataclass
class Wheel:
    repeat_stories: bool = False
    cooldown_minutes: float = 20
    half_life_hours: float = 3
    station_id_every: int = 5
    fill_silence: bool = True               # everything heard recently -> replay the oldest clip, never go silent
    boost_categories: list[str] = field(default_factory=list)  # e.g. kazakhstan: played a bit more often


@dataclass
class Ask:
    top_k: int = 6                  # reports given to the LLM
    min_score: float = 0.25         # cosine below this is not relevant enough to use
    max_chars_per_source: int = 1500
    verify: bool = True             # fact-check the answer like a broadcast script
    voice_ru: str = "ru-RU-DmitryNeural"


@dataclass
class Server:
    host: str = "0.0.0.0"
    port: int = 8000
    run_worker: bool = True


@dataclass
class Config:
    station_name: str = "NewsRadio"
    language: str = "en"
    data_dir: str = "data"
    database_url: str = ""
    sources: Sources = field(default_factory=Sources)
    ingest: Ingest = field(default_factory=Ingest)
    cluster: Cluster = field(default_factory=Cluster)
    ner: Ner = field(default_factory=Ner)
    editor: Editor = field(default_factory=Editor)
    tts: TTS = field(default_factory=TTS)
    wheel: Wheel = field(default_factory=Wheel)
    ask: Ask = field(default_factory=Ask)
    server: Server = field(default_factory=Server)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def clips_path(self) -> Path:
        return self.data_path / "clips"

    @property
    def db_path(self) -> Path:
        return self.data_path / "newsradio.db"


def _build(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        ftype = f.type if isinstance(f.type, type) else globals().get(str(f.type))
        if ftype is not None and is_dataclass(ftype):
            value = _build(ftype, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = path or os.getenv("NEWSRADIO_CONFIG", "config.yaml")
    raw: dict[str, Any] = {}
    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _build(Config, raw)
    cfg.sources.rss = [s if isinstance(s, RssSource) else RssSource(**s) for s in cfg.sources.rss]
    cfg.database_url = os.getenv("DATABASE_URL", cfg.database_url)
    if os.getenv("EDITOR_MODEL"):
        cfg.editor.model = os.environ["EDITOR_MODEL"]
    cfg.clips_path.mkdir(parents=True, exist_ok=True)
    return cfg
