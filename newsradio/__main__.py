"""CLI.

  python -m newsradio run      # server + background worker (the full station)
  python -m newsradio cycle    # run one ingest->clip cycle and exit
  python -m newsradio ingest   # only fetch sources and print what was found
  python -m newsradio demo     # offline demo on fixtures/sample_articles.json, then serve
  python -m newsradio check-llm  # one test request to the configured LLM (checks key and model)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="newsradio")
    p.add_argument("command", choices=["run", "cycle", "ingest", "demo", "episode", "export-labels", "retry-failed",
                                          "check-llm"])
    p.add_argument("--config", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--no-serve", action="store_true", help="demo: build clips but don't start the server")
    p.add_argument("--editor", choices=["gemini", "claude", "fake"], help="override editor.backend")
    p.add_argument("--tts", choices=["edge", "kokoro", "dummy"], help="override tts.backend")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "trafilatura", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.editor:
        cfg.editor.backend = args.editor
    if args.tts:
        cfg.tts.backend = args.tts

    if args.command == "run":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port)

    elif args.command == "episode":
        from .db import DB
        from .episode import export_episode
        print(json.dumps(export_episode(DB(cfg.database_url or cfg.db_path), cfg.clips_path, args.output or "data/episode.mp3")))

    elif args.command == "check-llm":
        from .editor import SourceDoc, make_editor

        editor = make_editor(cfg.editor, cfg.station_name, cfg.language)
        doc = SourceDoc("Demo Wire", "Almaty metro opens two new stations",
                        "Almaty's metro opened two new stations on its western line on Monday. City officials "
                        "said the extension adds three kilometres of track. About forty thousand passengers a day "
                        "are expected.", "https://example.com/demo")
        print(f"backend={cfg.editor.backend} model={getattr(editor, 'model', '-')}")
        draft = editor.write([doc], None)
        print(json.dumps(draft.__dict__, ensure_ascii=False, indent=2))
        print("Unsupported sentences:", editor.verify(draft.script, [doc]))
        print("LLM works.")

    elif args.command == "retry-failed":
        from .db import DB
        DB(cfg.database_url or cfg.db_path).retry_failed()
        print("Failed stories unlocked for retry")

    elif args.command == "export-labels":
        from .db import DB
        from .label_export import export_labels
        print(export_labels(DB(cfg.database_url or cfg.db_path), args.output or "data/annotations.json", args.limit))

    elif args.command == "cycle":
        from .pipeline import Pipeline

        print(json.dumps(Pipeline.build(cfg).run_cycle(), indent=2))

    elif args.command == "ingest":
        from .db import DB
        from .pipeline import Pipeline

        pl = Pipeline.__new__(Pipeline)
        pl.cfg, pl.db = cfg, DB(cfg.database_url or cfg.db_path)
        for a in Pipeline.collect(pl):
            print(f"[{a.source}] {a.title[:100]}  ({len(a.text)} chars)")

    elif args.command == "demo":
        from .ingest import RawArticle
        from .pipeline import Pipeline

        cfg.database_url = ""  # demo must never write to the production database
        from tempfile import mkdtemp
        demo_root = Path(cfg.data_dir) / "demo"
        demo_root.mkdir(parents=True, exist_ok=True)
        cfg.data_dir = mkdtemp(prefix="run-", dir=demo_root)
        print(f"Fresh demo directory: {cfg.data_dir}")
        cfg.wheel.repeat_stories = True
        cfg.wheel.cooldown_minutes = 0  # continuous playback for a short demo queue
        cfg.clips_path.mkdir(parents=True, exist_ok=True)
        cfg.cluster.embedder, cfg.cluster.threshold = "hash", 0.45
        cfg.ner.backend = "none"  # the offline demo downloads no models
        cfg.editor.backend = args.editor or "fake"
        cfg.tts.backend = args.tts or "dummy"
        cfg.editor.reedit_cooldown_minutes = 0
        cfg.editor.max_edits_per_cycle = 20  # voice every fixture story in one go
        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_articles.json"
        items = [RawArticle(**x) for x in json.loads(fixtures.read_text(encoding="utf-8"))]
        pipeline = Pipeline.build(cfg)
        print(json.dumps(pipeline.run_cycle(items), indent=2))
        from .episode import export_episode
        print(json.dumps(export_episode(pipeline.db, cfg.clips_path, Path(cfg.data_dir) / "episode.mp3")))
        if not args.no_serve:
            import uvicorn

            from .server import create_app

            uvicorn.run(create_app(cfg, run_worker=False), host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
