"""ffmpeg helpers. All clips are normalized to one format so they can be
concatenated byte-wise into a continuous MP3 stream."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 44100
CHANNELS = 1
BITRATE = "128k"
BYTES_PER_SEC = 128_000 // 8


def _run(args: list[str]) -> None:
    args[0] = os.getenv("FFMPEG_BINARY", args[0])
    proc = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{proc.stderr[-2000:]}")


def _encode_args(out: Path) -> list[str]:
    # No ID3 tag and no Xing header -> files concatenate cleanly into one stream.
    return ["-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "libmp3lame", "-b:a", BITRATE,
            "-map_metadata", "-1", "-id3v2_version", "0", "-write_xing", "0", str(out)]


def normalize(src: Path, dst: Path, pad_seconds: float = 0.6) -> None:
    """EBU R128 loudness normalization + a short tail of silence between items."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
          "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur={pad_seconds}",
          *_encode_args(dst)])


def silence(dst: Path, seconds: float = 2.0) -> None:
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono", "-t", str(seconds), *_encode_args(dst)])


def tone(dst: Path, seconds: float) -> None:
    """Quiet placeholder tone (used by the dummy TTS backend)."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"sine=frequency=440:sample_rate={SAMPLE_RATE}:duration={seconds}",
          "-af", "volume=0.1", *_encode_args(dst)])


def duration(path: Path) -> float:
    if not shutil.which("ffprobe"):
        return path.stat().st_size / BYTES_PER_SEC
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return path.stat().st_size / BYTES_PER_SEC
