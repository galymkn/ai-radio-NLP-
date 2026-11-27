"""Source collectors. Each returns a list of RawArticle."""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

USER_AGENT = "Mozilla/5.0 (compatible; NewsRadioBot/0.1; +https://github.com/)"


@dataclass
class RawArticle:
    url: str
    source: str
    kind: str
    title: str
    text: str
    published_at: float | None = None
    title_key: str = field(init=False)
    raw_title: str = field(init=False)
    raw_text: str = field(init=False)

    def __post_init__(self):
        self.raw_title, self.raw_text = self.title, self.text
        self.title = clean_text(self.title)[:300]
        self.text = clean_text(self.text)
        self.title_key = title_key(self.title)


_ws = re.compile(r"\s+")
_tags = re.compile(r"<[^>]+>")
_urls = re.compile(r"https?://\S+")
_emoji = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️]+", flags=re.UNICODE
)


def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = _tags.sub(" ", s)
    s = _urls.sub(" ", s)
    s = _emoji.sub(" ", s)
    return _ws.sub(" ", s).strip()


def title_key(title: str) -> str:
    """Normalized title used for cheap exact-duplicate detection."""
    s = unicodedata.normalize("NFKC", title.lower())
    s = re.sub(r"[^\w\s]", "", s)
    return _ws.sub(" ", s).strip()[:120]
