"""Telegram public-channel collector.

Uses the public web preview (https://t.me/s/<channel>), which needs no API keys and
works for any public channel. For private channels or higher volume, swap this for
Telethon (MTProto) — the function signature stays the same.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

import httpx
from bs4 import BeautifulSoup

from . import USER_AGENT, RawArticle

log = logging.getLogger(__name__)


def parse_channel_html(channel: str, html: str) -> list[RawArticle]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for msg in soup.select("div.tgme_widget_message"):
        post = msg.get("data-post")  # "channel/12345"
        body = msg.select_one("div.tgme_widget_message_text")
        if not post or body is None:
            continue
        for br in body.find_all("br"):
            br.replace_with("\n")
        text = body.get_text()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        title = lines[0][:200]
        published = None
        t = msg.select_one("time[datetime]")
        if t is not None:
            try:
                published = datetime.fromisoformat(t["datetime"]).timestamp()
            except ValueError:
                pass
        out.append(RawArticle(url=f"https://t.me/{post}", source=f"@{channel}", kind="telegram",
                              title=title, text=" ".join(lines), published_at=published))
    return out


def fetch_channel(
    channel: str,
    *,
    max_age_hours: float,
    min_text_chars: int,
    is_known: Callable[[str], bool],
    timeout: float = 15.0,
) -> list[RawArticle]:
    channel = channel.lstrip("@")
    try:
        r = httpx.get(f"https://t.me/s/{channel}", headers={"User-Agent": USER_AGENT}, timeout=timeout,
                      follow_redirects=True)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        log.warning("telegram @%s failed: %s", channel, e)
        return []
    cutoff = time.time() - max_age_hours * 3600
    items = [
        a for a in parse_channel_html(channel, r.text)
        if len(a.text) >= min_text_chars
        and not is_known(a.url)
        and (a.published_at is None or a.published_at >= cutoff)
    ]
    log.info("tg  @%-13s +%d", channel, len(items))
    return items
