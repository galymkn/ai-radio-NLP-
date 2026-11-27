"""Export inspectable source text and blank human labels; no synthetic gold labels."""
import json
from pathlib import Path
from .editor import split_sentences


def export_labels(db, destination, limit=50):
    if limit < 1:
        raise ValueError('limit must be positive')
    with db.lock:
        articles = db.conn.execute(
            'SELECT id, url, source, title, text, published_at, story_id FROM articles '
            'WHERE story_id IS NOT NULL ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        story_ids = {a['story_id'] for a in articles}
        clips = db.conn.execute('SELECT * FROM clips ORDER BY id').fetchall()
        events = {r['clip_id']: dict(r) for r in db.conn.execute('SELECT * FROM broadcast_events').fetchall()}
        sentences, broadcasts = [], []
        for clip in clips:
            if clip['story_id'] not in story_ids:
                continue
            docs = db.story_articles(clip['story_id'])
            evidence = [{k: d[k] for k in ('id', 'url', 'source', 'title', 'text', 'published_at')} for d in docs]
            for index, sentence in enumerate(split_sentences(clip['script'])):
                sentences.append({'clip_id': clip['id'], 'index': index, 'text': sentence,
                                  'supported': None, 'note': '', 'source_documents': evidence})
            event = events.get(clip['id'])
            # Use only articles already available when this clip was created.
            dates = [d['published_at'] for d in docs if d['published_at'] is not None
                     and d['fetched_at'] <= clip['created_at'] and d['published_at'] <= clip['created_at']]
            if event and dates:
                broadcasts.append({'clip_id': clip['id'], 'published_at': min(dates),
                                   'broadcast_at': event['started_at'], 'event_kind': event['event_kind']})
    result = {
        'instructions': 'Assign gold_story labels across ALL sampled articles; set supported true/false from evidence. Null means unlabelled, never a negative label.',
        'sampling': 'Most recent clustered articles; convenience sample, not representative quality estimate.',
        'latency_definition': 'Earliest known publication in the story at clip creation to first playback-start ACK or stream first-chunk send. Neither proves audible playback.',
        'articles': [{**dict(a), 'predicted_story': a['story_id'], 'gold_story': None} for a in articles],
        'sentences': sentences, 'broadcasts': broadcasts,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(destination)
