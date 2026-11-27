"""Export normalized MP3 clips and an inspectable source manifest."""
import json
from pathlib import Path


def export_episode(db, clips_dir, destination, limit=10):
    clips = sorted(db.active_clips(), key=lambda c: (-c.priority, -c.created_at))[:limit]
    if not clips:
        raise ValueError('No active clips. Run demo or cycle first.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.tmp')
    try:
        with temp.open('wb') as out:
            for clip in clips:
                out.write((Path(clips_dir) / clip.audio_file).read_bytes())
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    manifest = []
    offset = 0.0
    for clip in clips:
        docs = db.story_articles(clip.story_id)
        manifest.append({**clip.to_dict(), 'start_seconds': round(offset, 3),
                         'source_urls': [d['url'] for d in docs]})
        offset += clip.duration
    destination.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {'file': str(destination), 'clips': len(clips), 'seconds': round(offset, 2)}
