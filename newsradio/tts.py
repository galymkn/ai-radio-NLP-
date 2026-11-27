"""Text-to-speech backends. Every backend writes a raw audio file; audio.normalize()
then converts it to the common stream format."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol

from . import audio
from .config import TTS as TTSCfg

log = logging.getLogger(__name__)


class TTSBackend(Protocol):
    ext: str

    def synthesize(self, text: str, out: Path) -> None: ...


class EdgeTTS:
    """Microsoft Edge neural voices via the `edge-tts` package. Free, no key, needs internet."""

    ext = "mp3"

    def __init__(self, cfg: TTSCfg):
        import edge_tts  # noqa: F401

        self.voice, self.rate = cfg.voice, cfg.rate

    def synthesize(self, text: str, out: Path) -> None:
        import edge_tts

        async def _go():
            await edge_tts.Communicate(text, self.voice, rate=self.rate).save(str(out))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_go())
        else:
            # A second event loop cannot run in a thread with an active loop.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(lambda: asyncio.run(_go())).result()



class KokoroTTS:
    """Local open-source TTS (Kokoro-82M, Apache-2.0). Runs on CPU / Apple Silicon."""

    ext = "wav"

    def __init__(self, cfg: TTSCfg):
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code="a")  # 'a' = American English, 'b' = British
        self.voice = cfg.voice if not cfg.voice.startswith("en-") else "af_heart"

    def synthesize(self, text: str, out: Path) -> None:
        import numpy as np
        import soundfile as sf

        chunks = [a for _, _, a in self.pipeline(text, voice=self.voice)]
        sf.write(str(out), np.concatenate(chunks), 24000)


class DummyTTS:
    """No speech — a quiet tone whose length matches the text. For tests/offline demo."""

    ext = "mp3"

    def __init__(self, cfg: TTSCfg):
        pass

    def synthesize(self, text: str, out: Path) -> None:
        words = max(1, len(text.split()))
        audio.tone(out, seconds=min(30.0, words / 2.6))  # ~156 wpm newsreader pace


def make_tts(cfg: TTSCfg) -> TTSBackend:
    backends = {"edge": EdgeTTS, "kokoro": KokoroTTS, "dummy": DummyTTS}
    if cfg.backend not in backends:
        raise ValueError(f"unknown tts backend: {cfg.backend}")
    return backends[cfg.backend](cfg)


def render(tts: TTSBackend, text: str, clips_dir: Path, name: str) -> tuple[str, float]:
    """Synthesize + normalize. Returns (file name inside clips_dir, duration seconds)."""
    raw = clips_dir / f"{name}.raw.{tts.ext}"
    final = clips_dir / f"{name}.mp3"
    tts.synthesize(text, raw)
    audio.normalize(raw, final)
    raw.unlink(missing_ok=True)
    return final.name, audio.duration(final)
