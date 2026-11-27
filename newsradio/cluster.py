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


def cluster_new_articles(db: DB, embedder: Embedder, cfg: Cluster, now: float | None = None) -> dict:
    now = now or time.time()
    rows = db.unclustered_articles()
    if not rows:
        return {"clustered": 0, "new_stories": 0}

    embs = embedder.encode([article_repr(r["title"], r["text"]) for r in rows])

    stories = db.recent_stories(since=now - cfg.window_hours * 3600)
    ids: list[int] = [s["id"] for s in stories]
    counts: list[int] = [s["n_articles"] for s in stories]
    centroids = [from_blob(s["centroid"]).copy() for s in stories]
    dim = embs.shape[1]
    centroids = [c for c in centroids if c.shape[0] == dim] if centroids else []
    if len(centroids) != len(ids):  # embedder changed -> old centroids are incompatible
        ids, counts, centroids = [], [], []

    touched: set[int] = set()
    new_stories = 0
    for row, e in zip(rows, embs):
        best, best_sim = -1, -1.0
        if centroids:
            sims = np.stack(centroids) @ e
            best = int(np.argmax(sims))
            best_sim = float(sims[best])
        if best >= 0 and best_sim >= cfg.threshold:
            i = best
            counts[i] += 1
            centroids[i] = _normalize(centroids[i] * (counts[i] - 1) + e)
        else:
            sid = db.create_story(e, now)
            ids.append(sid)
            counts.append(1)
            centroids.append(e.copy())
            i = len(ids) - 1
            new_stories += 1
        db.set_article_story(row["id"], ids[i], e)
        touched.add(i)

    for i in touched:
        db.update_story(ids[i], centroids[i], counts[i], now)
    db.commit()
    log.info("clustered %d articles (%d new stories, %d touched)", len(rows), new_stories, len(touched))
    return {"clustered": len(rows), "new_stories": new_stories, "touched": len(touched)}
