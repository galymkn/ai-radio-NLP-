"""Measure tokenization statistics, encoder latency, throughput and peak memory. Writes benchmark.json."""
from pathlib import Path
import json, platform, resource, statistics, time
import numpy as np
import torch
from encode import encoder, vectors, MODEL, REVISION

ROOT = Path(__file__).resolve().parent
torch.manual_seed(0)
rows = json.loads((ROOT/'dataset.json').read_text())
model = encoder()
tok = model.tokenizer
limit = model.max_seq_length

def lengths(texts):
    return [len(tok(t, truncation=False)['input_ids']) for t in texts]

views = {'headline': [r['title'] for r in rows], 'headline_plus_lead': [f"{r['title']}. {r['lead'][:600]}" for r in rows]}
token_stats = {}
for name, texts in views.items():
    n = lengths(texts)
    token_stats[name] = dict(median=float(np.median(n)), p95=float(np.percentile(n, 95)), max=int(max(n)),
                             truncated_at_limit=int(sum(x > limit for x in n)), documents=len(n))

vectors(model, rows[:16])  # warm-up
single = []
for r in rows[:50]:
    start = time.perf_counter(); vectors(model, [r]); single.append(time.perf_counter() - start)
start = time.perf_counter(); vectors(model, rows); full = time.perf_counter() - start

out = dict(model=MODEL, revision=REVISION, device='cpu', torch_threads=torch.get_num_threads(), platform=platform.platform(),
           max_seq_length=limit, vocabulary_size=tok.vocab_size, token_lengths=token_stats,
           single_article_both_views_ms=dict(median=1000*statistics.median(single), p95=1000*float(np.percentile(single, 95)), n=len(single)),
           corpus_282_articles_both_views_seconds=full, throughput_articles_per_second=len(rows)/full,
           peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(ROOT/'benchmark.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
