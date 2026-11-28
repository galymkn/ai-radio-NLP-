"""Named entity recognition: people, organisations and places in each article.

Used for two things:
  * clustering: two reports about the same kind of event in different places
    ("drone attack on Kyiv" / "drone attack on Belgorod") must not be merged;
  * the player: show who and where a story is about.

The default model, Babelscape/wikineural-multilingual-ner (mBERT fine-tuned on
WikiNEuRal), tags PER / ORG / LOC / MISC in nine languages including English and
Russian. It is downloaded from Hugging Face on first use (about 700 MB).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

KINDS = ("PER", "ORG", "LOC")   # MISC (events, nationalities, products) is too noisy to use
Entity = tuple[str, str]        # (kind, surface text as written in the article)


class NER(Protocol):
    def extract(self, texts: list[str]) -> list[list[Entity]]: ...


class TransformersNER:
    def __init__(self, model: str, min_score: float = 0.6):
        from transformers import pipeline

        self.pipe = pipeline("token-classification", model=model, aggregation_strategy="simple")
        self.min_score = min_score

    def extract(self, texts: list[str]) -> list[list[Entity]]:
        if not texts:
            return []
        out = []
        for found in self.pipe(texts, batch_size=16):
            ents = []
            for e in found:
                kind, word = e.get("entity_group"), clean_name(e.get("word", ""))
                if kind in KINDS and e.get("score", 0) >= self.min_score and len(word) >= 2:
                    ents.append((kind, word))
            out.append(list(dict.fromkeys(ents)))  # dedupe, keep order
        return out


class NoNER:
    """Switched off: no entities, clustering works as before."""

    def extract(self, texts: list[str]) -> list[list[Entity]]:
        return [[] for _ in texts]


def make_ner(cfg) -> NER:
    if cfg.backend == "transformers":
        try:
            return TransformersNER(cfg.model, cfg.min_score)
        except Exception as exc:  # noqa: BLE001 — no model, no network: run the station without NER
            log.warning("NER model %s unavailable (%s); continuing without entities", cfg.model, exc)
            return NoNER()
    if cfg.backend == "none":
        return NoNER()
    raise ValueError(f"unknown ner backend: {cfg.backend}")


def ner_text(title: str, text: str) -> str:
    # The place and the actors of an event are almost always in the headline and the first lines.
    return f"{title}. {text[:400]}"


# ---------------------------------------------------------------- comparing names across languages

def clean_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("##", "")).strip(" .,;:«»\"'()")


_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya", "ә": "a", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
    "һ": "h", "і": "i",
})


def name_key(s: str) -> str:
    """Lower-case Latin spelling: 'Москве' -> 'moskve', 'Tokayev' -> 'tokayev'."""
    s = unicodedata.normalize("NFKC", s).lower().translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9 ]", "", "".join(c for c in s if not unicodedata.combining(c))).strip()


class NameMatcher:
    """Decides whether two names refer to the same thing, also across EN/RU and Russian cases.

    Same if the Latin keys are equal, if one key is a 5+ letter prefix of the other up to a short
    case ending ('moskva' ~ 'moskve' via the shared stem 'moskv'), or if the multilingual
    encoder puts the names close together ('Киев' ~ 'Kyiv').
    """

    def __init__(self, embedder, threshold: float):
        self.embedder, self.threshold = embedder, threshold
        self._vec: dict[str, np.ndarray] = {}

    def _vectors(self, names: list[str]) -> None:
        todo = [n for n in dict.fromkeys(names) if n not in self._vec]
        if todo and self.embedder is not None:
            for n, v in zip(todo, self.embedder.encode(todo)):
                self._vec[n] = v / (np.linalg.norm(v) or 1)

    def same(self, a: str, b: str) -> bool:
        ka, kb = name_key(a), name_key(b)
        if not ka or not kb:
            return False
        if ka == kb:
            return True
        stem = min(len(ka), len(kb)) - 1
        if stem >= 5 and ka[:stem] == kb[:stem] and abs(len(ka) - len(kb)) <= 2:
            return True
        if self.embedder is None:
            return False
        self._vectors([a, b])
        return float(self._vec[a] @ self._vec[b]) >= self.threshold

    def conflict(self, places_a: set[str], places_b: set[str]) -> bool:
        """Both sides name places and none of them match: different events."""
        if not places_a or not places_b:
            return False
        return not any(self.same(a, b) for a in places_a for b in places_b)


def summarize(entity_lists: list[list[Entity]], per_kind: int = 3) -> dict[str, list[str]]:
    """Most mentioned names of a story by kind, one spelling per name (Latin spelling preferred).

    Spellings are grouped loosely for display: same kind, same first 4 letters of the Latin key and
    almost the same length ('Токаев' -> 'tokaev' ~ 'tokayev')."""
    groups: list[dict] = []   # {"kind", "key", "forms": [...]}
    for ents in entity_lists:
        for kind, word in ents:
            key = name_key(word) or word.lower()
            for g in groups:
                if g["kind"] == kind and g["key"][:4] == key[:4] and abs(len(g["key"]) - len(key)) <= 2:
                    g["forms"].append(word)
                    break
            else:
                groups.append({"kind": kind, "key": key, "forms": [word]})
    out: dict[str, list[str]] = {k: [] for k in KINDS}
    for g in sorted(groups, key=lambda g: -len(g["forms"])):
        if len(out[g["kind"]]) < per_kind:
            latin = [w for w in g["forms"] if w.isascii()] or g["forms"]
            out[g["kind"]].append(max(set(latin), key=latin.count))
    return out
