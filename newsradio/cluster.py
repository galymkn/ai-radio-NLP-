"""Embedding + incremental clustering of articles into stories.

Each story keeps a running centroid. A new article joins the most similar story
(cosine >= threshold) within the time window, otherwise it starts a new story.
With a multilingual encoder, a Russian Telegram post and an English BBC article
about the same event end up in the same story.
"""
from __future__ import annotations

import logging
import time
from typing import Protocol

import numpy as np

from .config import Cluster
from .db import DB, from_blob
from .ner import NER, NameMatcher, ner_text

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SbertEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False).astype(np.float32)


class HashEmbedder:
    """Char n-gram hashing — no model download. Monolingual and crude; for tests/offline only."""

    def __init__(self, n_features: int = 4096):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=n_features,
                                     alternate_sign=False, norm="l2")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)


def make_embedder(cfg: Cluster) -> Embedder:
    if cfg.embedder == "sbert":
        return SbertEmbedder(cfg.model)
    if cfg.embedder == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedder: {cfg.embedder}")


def article_repr(title: str, text: str) -> str:
    # Title + lead carries most of the "what happened" signal.
    return f"{title}. {text[:600]}"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def places(entities) -> set[str]:
    return {name for kind, name in entities if kind == "LOC"}


def cluster_new_articles(db: DB, embedder: Embedder, cfg: Cluster, now: float | None = None,
                         ner: NER | None = None) -> dict:
    now = now or time.time()
    rows = db.unclustered_articles()
    if not rows:
        return {"clustered": 0, "new_stories": 0}

    embs = embedder.encode([article_repr(r["title"], r["text"]) for r in rows])
    ents = [[] for _ in rows]
    if ner is not None:
        try:
            ents = ner.extract([ner_text(r["title"], r["text"]) for r in rows])
        except Exception:  # noqa: BLE001 — a NER failure must not stop the station; cluster without it
            log.exception("NER failed; clustering this batch without entities")
            ner = None
    for row, e in zip(rows, ents):
        db.save_entities(row["id"], e)

    stories = db.recent_stories(since=now - cfg.window_hours * 3600)
    ids: list[int] = [s["id"] for s in stories]
    counts: list[int] = [s["n_articles"] for s in stories]
    centroids = [from_blob(s["centroid"]).copy() for s in stories]
    dim = embs.shape[1]
    centroids = [c for c in centroids if c.shape[0] == dim] if centroids else []
    if len(centroids) != len(ids):  # embedder changed -> old centroids are incompatible
        ids, counts, centroids = [], [], []

    # Entity veto: the story's places (from all its articles) must not contradict the article's places.
    matcher = NameMatcher(embedder, cfg.entity_match) if (cfg.entity_veto and ner is not None) else None
    story_places: list[set[str]] = []
    if matcher:
        known = db.story_entities(ids)
        story_places = [set().union(*(places(a) for a in known.get(sid, []))) for sid in ids]

    touched: set[int] = set()
    new_stories = vetoed = 0
    for row, e, art_ents in zip(rows, embs, ents):
        chosen = -1
        if centroids:
            sims = np.stack(centroids) @ e
            for j in np.argsort(-sims):          # most similar story first
                if sims[j] < cfg.threshold:
                    break
                if matcher and matcher.conflict(places(art_ents), story_places[j]):
                    vetoed += 1                  # similar topic, different place: keep looking
                    continue
                chosen = int(j)
                break
        if chosen >= 0:
            i = chosen
            counts[i] += 1
            centroids[i] = _normalize(centroids[i] * (counts[i] - 1) + e)
        else:
            sid = db.create_story(e, now)
            ids.append(sid)
            counts.append(1)
            centroids.append(e.copy())
            if matcher:
                story_places.append(set())
            i = len(ids) - 1
            new_stories += 1
        if matcher:
            story_places[i] |= places(art_ents)
        db.set_article_story(row["id"], ids[i], e)
        touched.add(i)

    for i in touched:
        db.update_story(ids[i], centroids[i], counts[i], now)
    db.commit()
    log.info("clustered %d articles (%d new stories, %d touched, %d merges vetoed by places)",
             len(rows), new_stories, len(touched), vetoed)
    return {"clustered": len(rows), "new_stories": new_stories, "touched": len(touched), "vetoed": vetoed}
