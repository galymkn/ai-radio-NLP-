"""RSS/Atom collector with optional full-text extraction (trafilatura)."""
from __future__ import annotations

import calendar
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import feedparser
import httpx

from . import USER_AGENT, RawArticle, clean_text

log = logging.getLogger(__name__)


def _published(entry) -> float | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return float(calendar.timegm(t))
    return None


def _fulltext(client: httpx.Client, url: str) -> str | None:
    try:
        import trafilatura

        r = client.get(url)
        r.raise_for_status()
        return trafilatura.extract(r.text, include_comments=False, include_tables=False, favor_precision=True)
    except Exception as e:  # noqa: BLE001 — a single bad page must not break the cycle
        log.debug("fulltext failed for %s: %s", url, e)
        return None


def fetch_feed(
    name: str,
    url: str,
    *,
    max_items: int,
    max_age_hours: float,
    fetch_fulltext: bool,
    is_known: Callable[[str], bool],
    timeout: float = 15.0,
) -> list[RawArticle]:
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        try:
            resp = client.get(url)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            log.warning("feed %s failed: %s", name, e)
            return []
        feed = feedparser.parse(resp.content)

        cutoff = time.time() - max_age_hours * 3600
        fresh = []
        for entry in feed.entries[:max_items]:
            link = entry.get("link")
            if not link or is_known(link):
                continue
            pub = _published(entry)
            if pub and pub < cutoff:
                continue
            fresh.append((entry, link, pub))

        texts: list[str | None] = [None] * len(fresh)
        if fetch_fulltext and fresh:
            with ThreadPoolExecutor(max_workers=6) as pool:
                texts = list(pool.map(lambda x: _fulltext(client, x[1]), fresh))

    out = []
    for (entry, link, pub), full in zip(fresh, texts):
        summary = clean_text(entry.get("summary", ""))
        text = full if full and len(full) > len(summary) else summary
        out.append(RawArticle(url=link, source=name, kind="rss", title=entry.get("title", ""), text=text,
                              published_at=pub))
    log.info("rss %-14s +%d", name, len(out))
    return out
