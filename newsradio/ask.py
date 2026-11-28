"""Ask the radio: answer a listener's question from recent reports (retrieval-augmented generation).

1. Retrieve: embed the question with the same multilingual encoder that clusters the articles
   and take the most similar recent articles (cosine), at most two per story so one event
   cannot crowd out the rest.
2. Generate: the LLM answers ONLY from those numbered reports and says which ones it used.
3. Check: the answer goes through the same sentence-by-sentence fact check as broadcast scripts;
   unsupported sentences are dropped, and an unverifiable answer is not given.
4. Speak: the answer is voiced (English or Russian voice, following the question's language).
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import nullcontext

import numpy as np

from .config import TTS as TTSCfg
from .db import from_blob
from .editor import SourceDoc, apply_verification, split_sentences
from .tts import make_tts, render

log = logging.getLogger(__name__)

NOT_COVERED = {
    "English": "Today's reports don't cover that yet. Try asking about a story from the list.",
    "Russian": "В сегодняшних сообщениях об этом пока ничего нет. Спросите о новости из списка.",
}
NOT_CONFIRMED = {
    "English": "I couldn't confirm an answer from today's reports.",
    "Russian": "Не удалось подтвердить ответ по сегодняшним сообщениям.",
}


def question_language(question: str) -> str:
    return "Russian" if re.search(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", question) else "English"


def retrieve(db, embedder, question: str, *, k: int, window_hours: float, min_score: float,
             per_story: int = 2) -> list[tuple[float, dict]]:
    q = embedder.encode([question])[0].astype(np.float32)
    q /= np.linalg.norm(q) or 1
    scored = []
    for r in db.recent_articles_with_embeddings(time.time() - window_hours * 3600):
        v = from_blob(r["embedding"])
        if v.shape == q.shape:
            scored.append((float(v @ q), dict(r)))
    scored.sort(key=lambda x: -x[0])
    picked, per = [], {}
    for score, row in scored:
        if score < min_score or len(picked) >= k:
            break
        if per.get(row["story_id"], 0) >= per_story:
            continue
        per[row["story_id"]] = per.get(row["story_id"], 0) + 1
        picked.append((score, row))
    return picked


def answer_question(pipeline, question: str) -> dict:
    # Listener questions may use the part of the daily LLM quota that the editors leave free.
    with getattr(pipeline.editor, "for_questions", nullcontext)():
        return _answer(pipeline, question)


def _answer(pipeline, question: str) -> dict:
    cfg = pipeline.cfg
    ac = cfg.ask
    language = question_language(question)
    hits = retrieve(pipeline.db, pipeline.embedder, question, k=ac.top_k, window_hours=cfg.cluster.window_hours,
                    min_score=ac.min_score)
    log.info("ask %r: %d reports retrieved", question[:80], len(hits))
    if not hits:
        return {"found": False, "answer": NOT_COVERED[language], "sources": [], "audio_url": None}

    docs = [SourceDoc(source=r["source"], title=r["title"], text=r["text"][: ac.max_chars_per_source], url=r["url"])
            for _, r in hits]
    result = pipeline.editor.answer(question, docs, language)
    text = result.get("answer", "").strip()
    if not result.get("found") or not text:
        return {"found": False, "answer": text or NOT_COVERED[language], "sources": [], "audio_url": None}

    if ac.verify:
        unsupported = pipeline.editor.verify(text, docs)
        if unsupported:
            text = apply_verification(text, unsupported)
        if not split_sentences(text):
            return {"found": False, "answer": NOT_CONFIRMED[language], "sources": [], "audio_url": None}

    used = [i for i in result.get("sources", []) if isinstance(i, int) and 1 <= i <= len(docs)]
    sources = [{"n": i, "source": docs[i - 1].source, "title": docs[i - 1].title, "url": docs[i - 1].url,
                "score": round(hits[i - 1][0], 3)} for i in dict.fromkeys(used or range(1, len(docs) + 1))]

    audio_url = None
    try:
        voice = ac.voice_ru if language == "Russian" else cfg.tts.voice
        tts = pipeline.tts if voice == cfg.tts.voice else make_tts(
            TTSCfg(backend=cfg.tts.backend, normalize=False, voice=voice, rate=cfg.tts.rate))
        name, _ = render(tts, text, cfg.clips_path, f"ask-{uuid.uuid4().hex[:10]}")
        audio_url = f"/audio/{name}"
    except Exception:  # noqa: BLE001 — the text answer is still useful without audio
        log.exception("could not voice the answer")
    return {"found": True, "answer": text, "sources": sources, "audio_url": audio_url, "language": language}
