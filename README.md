# NewsRadio

RSS / Telegram → storage → story clustering (SBERT embeddings) → LLM editor (Gemini free tier by default) → per-sentence fact check → TTS → MP3 / web player / live stream.

See [README_RU.md](README_RU.md) for setup, Docker/Postgres, Telegram credentials, demo commands, evaluation and limitations.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m newsradio demo
```

Requires ffmpeg. Open http://localhost:8000. Offline demo audio is a placeholder tone, not speech. For spoken audio use `python -m newsradio demo --tts edge` with network access.

Production extras: `pip install -e '.[embeddings,postgres,telegram,dev]'`.

Live station: put `GEMINI_API_KEY=...` into `.env` (free key from Google AI Studio), then `pip install -e '.[embeddings,dev]'`, `python -m newsradio check-llm`, `python -m newsradio run`.
