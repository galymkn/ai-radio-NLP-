"""Compute metrics from HUMAN labels: python -m newsradio.evaluate annotations.json."""
import argparse
import json
import math
from itertools import combinations


def evaluate(data):
    all_rows = data.get('articles', [])
    rows = [r for r in all_rows if r.get('gold_story') is not None and r.get('predicted_story') is not None]
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Article IDs must be unique')
    tp = fp = fn = 0
    for a, b in combinations(rows, 2):
        actual = a['gold_story'] == b['gold_story']
        predicted = a['predicted_story'] == b['predicted_story']
        tp += actual and predicted
        fp += predicted and not actual
        fn += actual and not predicted
    all_verdicts = data.get('sentences', [])
    verdicts = [r for r in all_verdicts if r.get('supported') is not None]
    if any(type(r['supported']) is not bool for r in verdicts):
        raise ValueError('Sentence support must be manually labelled true/false')
    latency = []
    for row in data.get('broadcasts', []):
        delta = row['broadcast_at'] - row['published_at']
        if not math.isfinite(delta) or delta < 0:
            raise ValueError('Invalid publication-to-broadcast latency')
        latency.append(delta)
    latency.sort()
    return {
        'unlabelled_articles': len(all_rows) - len(rows),
        'unlabelled_sentences': len(all_verdicts) - len(verdicts),
        'labelled_articles': len(rows), 'pair_tp': tp, 'pair_fp': fp, 'pair_fn': fn,
        'dedup_precision': tp / (tp + fp) if tp + fp else None,
        'dedup_recall': tp / (tp + fn) if tp + fn else None,
        'labelled_sentences': len(verdicts),
        'unsupported_sentence_rate': sum(not r['supported'] for r in verdicts) / len(verdicts) if verdicts else None,
        'latency_samples': len(latency),
        'latency_p50_seconds': latency[math.ceil(len(latency) * .5)-1] if latency else None,
        'latency_p95_seconds': latency[math.ceil(len(latency) * .95)-1] if latency else None,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotations')
    args = parser.parse_args()
    print(json.dumps(evaluate(json.load(open(args.annotations))), indent=2))
