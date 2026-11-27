"""Draw learning curves, evaluation figures and the defense slides from saved outputs.

Writes learning-curves.pdf, figures.pdf and Defense.pdf. Every number comes from
history.json, checkpoint.json, results.json, v1/results.json, intervals.json and benchmark.json.
Validation scores are recomputed from the saved checkpoint (no training happens here).
"""
import json, textwrap
from itertools import combinations
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
from sklearn.metrics import precision_recall_curve

from match import features, sigmoid

r = Path(__file__).resolve().parent
load = lambda name: json.loads((r/name).read_text())
h, ck, res, v1, iv = load('history.json'), load('checkpoint.json'), load('results.json'), load('v1/results.json'), load('intervals.json')
bench = load('benchmark.json') if (r/'benchmark.json').exists() else None
INK, GREY, GREEN, ORANGE, BLUE = (HexColor(c) for c in ('#183734', '#666666', '#008476', '#c05226', '#404bb0'))
size = (960, 540)

# ---------- scores per split, recomputed from the checkpoint ----------
rows = load('dataset.json'); z = np.load(r/'embeddings.npz')
def split_scores(split):
    ids = [i for i, a in enumerate(rows) if a['split'] == split]
    pairs = list(combinations(ids, 2))
    x = np.array([features(i, j, rows, z['title'], z['full']) for i, j in pairs])
    y = np.array([int(rows[i]['event'] == rows[j]['event']) for i, j in pairs])
    p = sigmoid(((x - ck['mean']) / np.array(ck['scale'])) @ np.array(ck['weights']) + ck['bias'])
    return y, p, x[:, 0]
scores = {s: split_scores(s) for s in ('validation', 'test')}

# ---------- drawing helpers ----------
def text(c, x, y, s, size=10, font='Helvetica', color=INK, right=False):
    c.setFillColor(color); c.setFont(font, size)
    (c.drawRightString if right else c.drawString)(x, y, s)

def frame(c, x, y, w, ht, xlabel, ylabel, xmax_label):
    c.setStrokeColor(GREY); c.setLineWidth(1); c.rect(x, y, w, ht)
    text(c, x - 14, y, '0'); text(c, x - 14, y + ht - 8, '1')
    text(c, x, y - 14, xlabel.split('|')[0]); text(c, x + w, y - 14, xmax_label, right=True)
    text(c, x + w / 2, y - 14, xlabel.split('|')[-1]) if '|' in xlabel else None
    c.saveState(); c.translate(x - 22, y + ht / 2); c.rotate(90); text(c, -len(ylabel) * 2.5, 0, ylabel); c.restoreState()

def polyline(c, pts, color, width=2):
    c.setStrokeColor(color); c.setLineWidth(width); p = c.beginPath(); p.moveTo(*pts[0])
    for pt in pts[1:]: p.lineTo(*pt)
    c.drawPath(p)

def curve(c, x, y, w, ht, keys, colors, ylabel):
    frame(c, x, y, w, ht, 'epoch 1|epoch', ylabel, str(len(h)))
    for key, color in zip(keys, colors):
        polyline(c, [(x + w * i / (len(h) - 1), y + ht * row[key]) for i, row in enumerate(h)], color)
    m = x + w * (ck['epoch'] - 1) / (len(h) - 1)
    c.setStrokeColor(GREY); c.setDash(2, 2); c.line(m, y, m, y + ht); c.setDash()
    text(c, m + 3, y + ht - 13, f"selected epoch {ck['epoch']}")

def learning_curves(c):
    curve(c, 70, 100, 385, 205, ['train_loss', 'val_loss'], [GREEN, ORANGE], 'balanced BCE loss')
    curve(c, 530, 100, 340, 205, ['val_f1_at_half'], [BLUE], 'positive-class F1')
    text(c, 70, 323, 'Loss per epoch   green: training   orange: validation', 14)
    text(c, 530, 323, 'Validation F1 at fixed threshold 0.50', 14)

def matrix(c, x, y, title, cm, cell=46):
    (tn, fp), (fn, tp) = cm
    text(c, x, y + 2 * cell + 26, title, 12, 'Helvetica-Bold')
    text(c, x + cell, y + 2 * cell + 8, 'pred 0', 9, right=True); text(c, x + 2 * cell, y + 2 * cell + 8, 'pred 1', 9, right=True)
    for k, (v, label) in enumerate([(tn, 'TN'), (fp, 'FP'), (fn, 'FN'), (tp, 'TP')]):
        cx, cy = x + (k % 2) * cell, y + (1 - k // 2) * cell
        shade = HexColor('#dfeee9') if label in ('TN', 'TP') else HexColor('#f6ddd2')
        c.setFillColor(shade); c.setStrokeColor(GREY); c.rect(cx, cy, cell, cell, fill=1)
        text(c, cx + 4, cy + cell - 12, label, 8, color=GREY); text(c, cx + cell / 2 - 3 * len(str(v)), cy + 12, str(v), 13, 'Helvetica-Bold')
    text(c, x - 4, y + cell + 18, 'true 0', 9, right=True); text(c, x - 4, y + 18, 'true 1', 9, right=True)

def confusion(c, x, y):
    for k, (name, cm) in enumerate([('Baseline cos >= 0.72', res['baseline']['cm']),
                                    (f"v1 head, t={v1['threshold']:.3f}", v1['classifier']['cm']),
                                    (f"v2 head, t={ck['threshold']:.3f}", res['classifier']['cm'])]):
        matrix(c, x + k * 160, y, name, cm)

def pr(c, x, y, w, ht, split):
    yy, p, base = scores[split]
    frame(c, x, y, w, ht, 'recall 0|recall', 'precision', '1')
    for s, color, t in ((base, GREY, .72), (p, BLUE, ck['threshold'])):
        prec, rec, _ = precision_recall_curve(yy, s)
        polyline(c, [(x + w * a, y + ht * b) for a, b in zip(rec, prec)], color, 1.6)
        pred = s >= t; tp = (pred & (yy == 1)).sum()
        px, py = tp / yy.sum(), tp / max(1, pred.sum())
        c.setFillColor(color); c.circle(x + w * px, y + ht * py, 4, fill=1, stroke=0)
    text(c, x, y + ht + 8, f"{split.capitalize()}: {int(yy.sum())} positives / {len(yy)} pairs", 12, 'Helvetica-Bold')

def pipeline(c, y=300):
    steps = ['Raw article', 'Headline +\nlead[:600]', 'Tokenizer\nIDs, mask, 128', 'MiniLM-L12\nfrozen 117.7M', 'Mean pool\n384-d, L2', '4 pair\nfeatures', 'z-score +\nlinear (5)', 'Sigmoid\nscore']
    bw, gap = 98, 14
    for k, s in enumerate(steps):
        bx = 52 + k * (bw + gap); trainable = k == 6
        c.setFillColor(HexColor('#fbe6d8') if trainable else HexColor('#e4efec')); c.setStrokeColor(INK); c.roundRect(bx, y, bw, 52, 6, fill=1)
        for j, line in enumerate(s.split('\n')): text(c, bx + 7, y + 33 - 15 * j, line, 11.5, 'Helvetica-Bold' if j == 0 else 'Helvetica')
        if k: c.setStrokeColor(INK); c.line(bx - gap, y + 26, bx - 2, y + 26); c.line(bx - 7, y + 30, bx - 2, y + 26); c.line(bx - 7, y + 22, bx - 2, y + 26)
    text(c, 52 + 6 * (bw + gap), y - 18, 'only trainable part', 10, color=ORANGE)
    text(c, 52, y - 18, 'Training: balanced BCE -> full-batch gradient descent updates 4 weights + bias. Inference: score >= threshold -> same event.', 11)

# ---------- standalone figures ----------
c = Canvas(str(r/'learning-curves.pdf'), pagesize=size); learning_curves(c); c.save()
c = Canvas(str(r/'figures.pdf'), pagesize=size)
text(c, 50, 490, 'Test confusion matrices (sklearn order: rows true 0/1, columns predicted 0/1)', 16, 'Helvetica-Bold'); confusion(c, 90, 280)
text(c, 50, 200, 'Generated by plot_curves.py from results.json and v1/results.json.', 11); c.showPage()
text(c, 50, 490, 'Precision-recall curves. Blue: v2 head. Grey: baseline cosine. Dots: operating thresholds.', 16, 'Helvetica-Bold')
pr(c, 90, 120, 330, 300, 'validation'); pr(c, 540, 120, 330, 300, 'test'); c.showPage()
text(c, 50, 490, 'Pipeline', 16, 'Helvetica-Bold'); pipeline(c); c.save()

# ---------- slides ----------
def row(name, m, key):
    (tn, fp), (fn, tp) = m['cm']
    return [name, tp, fp, fn, f"{m['precision'][1]:.3f}", f"{m['recall'][1]:.3f}", f"{m['f1'][1]:.3f}", f"{m['ap']:.3f}", f"{iv[key][0]:.2f}-{iv[key][1]:.2f}"]
table = [['Test, 9180 pairs, 23 positive', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'AP', 'F1 95% CI'],
         row('Baseline cosine >= 0.72', res['baseline'], 'baseline'),
         row('v1 head (single-shot)', v1['classifier'], 'v1'),
         row('v2 head (post-hoc rerun)', res['classifier'], 'v2')]
lat = (f"Encoder on CPU: {bench['single_article_both_views_ms']['median']:.0f} ms per article (both views), "
       f"{bench['throughput_articles_per_second']:.0f} articles/s batched, peak RSS {bench['peak_process_rss_bytes']/2**30:.1f} GB."
       if bench else 'Encoder latency: not measured.')
trunc = (f"{bench['token_lengths']['headline_plus_lead']['truncated_at_limit']} of 282 headline+lead views exceed 128 tokens and are truncated."
         if bench else '')

slides = [
 ('One event, several headlines', '01:00', ['RSS and Telegram > event matching > Claude script and verification > speech synthesis > web radio',
   'Task: binary pair classification. Input: two articles (headline + lead) in English or Russian. Output: same-event score and decision.',
   'Label 1 = same specific incident or decision. Success: positive-class F1 on a later, unseen day vs the station rule.',
   'A missed match repeats a story. A false match can mix unrelated facts in one broadcast.']),
 ('282 articles across two collection days', '01:30', ['Public RSS and Telegram news, 122 English / 160 Russian. AI-assisted event labels, human review pending.',
   'Train 100 articles, 18 positive pairs (0.36%). Validation 46, 19 (1.84%). Test: 136 next-day articles, 23 in 9,180 (0.25%).',
   'Split by event before forming pairs. No shared events, headlines or URLs across splits; no cross-split pair with cosine > 0.85.',
   'Keep case, names, numbers; no stemming or stop-word removal. Leads cut to 600 chars. Test leads are shorter: median 273 vs 600.']),
 ('From text to five learned parameters', '02:00', ['PIPELINE',
   'Tokenizer: SentencePiece, 250k vocabulary, token IDs + attention mask, padding per batch of 16. ' + trunc,
   'Features: cosine of both views, headline word Jaccard, headline digit Jaccard. Standardized with train statistics.',
   'Why: only 18 positive training pairs, so fine-tuning would overfit. Baseline = the rule the station already uses.']),
 ('Training and checkpoint selection', '02:00', None),
 ('Test results: the baseline wins', '00:50', ['TABLE', 'CONFUSION']),
 ('Error analysis', '00:40', ['Correct: 1002/1016 US Thanksgiving storms, paraphrased. Cosine .575 (baseline misses), v2 .732.',
   'False positive: 1121/1137 drone attack in St Petersburg vs in Novorossiysk. v2 .999. Same type, different incident.',
   'False positive: 1128/1156 Russian vs German 2027 budget. Shared number pushes the score up.',
   'False negative: 1041/1149 Georgian police vs Turkish fans / Batumi match stopped. Cosine .506, v2 .292.',
   '96 of 126 v2 false positives are Russian-Russian pairs. Fix: hard negatives, entity overlap features.']),
 ('Why did validation and test disagree?', '00:30', ['PR']),
 ('Live demonstration', '00:50', ['match.py --demo "headline A" "headline B" --leads "lead A" "lead B"',
   'Shows input views, tokens, token IDs, attention mask, four features, score, threshold, decision and baseline cosine.',
   'Try a hard negative: two drone attacks in different cities. v2 says same event (.99999), baseline says different (.691).',
   lat]),
 ('Conclusion and limitations', '00:40', ['Main result: the fixed cosine rule stays. The learned head raises recall but loses too much precision.',
   'Reproducible: re-encoding and retraining from scratch reproduce every reported count and AP exactly (score differences < 1e-7).',
   'Limitations: two days, one test day, few positive events, provisional AI-assisted labels, pair-level only, test reused once.',
   'Next: human review and kappa, a third day as fresh test, realistic-prevalence validation, hard negatives, end-to-end clustering.'])]

def draw_table(c, y):
    xs = [52, 330, 380, 430, 480, 545, 610, 670, 735]
    for k, cells in enumerate(table):
        for x, cell in zip(xs, cells): text(c, x, y, str(cell), 15, 'Helvetica-Bold' if k == 0 else 'Helvetica')
        if k == 0: c.setStrokeColor(INK); c.line(52, y - 8, 900, y - 8)
        y -= 28
    return y - 10

c = Canvas(str(r/'Defense.pdf'), pagesize=size)
for title, dur, lines in slides:
    c.setFillColor(HexColor('#f8f7f2')); c.rect(0, 0, *size, stroke=0, fill=1)
    text(c, 50, 493, 'NEWSRADIO    /    NLP DEFENSE', 14); text(c, 908, 493, dur, 14, right=True)
    text(c, 50, 428, title, 32, 'Helvetica-Bold')
    if lines is None:
        learning_curves(c)
        text(c, 50, 378, f"Full-batch gradient descent, balanced BCE, LR {res['lr']}, L2 0.001, {res['epochs']} epochs, zero init, deterministic.", 15)
        text(c, 50, 356, f"Checkpoint: min validation loss, epoch {ck['epoch']}. Threshold {ck['threshold']:.3f} maximizes validation F1. Apple Silicon CPU, head {res['train_seconds']:.1f} s.", 15)
        text(c, 50, 58, f"v1 (LR 0.05, 200 epochs) was still improving at its last epoch. v2 overfits mildly after {ck['epoch']}: validation loss rises.", 15)
        text(c, 50, 32, 'Checkpoint and threshold selection used validation only. Both test evaluations are reported.', 15)
    else:
        y = 372
        for line in lines:
            if line == 'TABLE': y = draw_table(c, y); continue
            if line == 'CONFUSION':
                confusion(c, 120, 40)
                text(c, 640, 150, 'Paired event bootstrap, 3,000 draws:', 13); text(c, 640, 132, 'v2 minus baseline F1 = [-0.53, -0.08].', 13)
                text(c, 640, 106, 'Test scored twice: v1 is the clean', 13); text(c, 640, 88, 'single-shot result, v2 a post-hoc rerun.', 13)
                continue
            if line == 'PR':
                pr(c, 90, 60, 250, 250, 'validation'); pr(c, 420, 60, 250, 250, 'test')
                for k, s in enumerate(['Blue: v2 head. Grey: baseline.', 'Dots: chosen thresholds.', '', 'Threshold 0.675 picked at', '1.84% prevalence; test has 0.25%.', 'Test leads half as long.', 'Only 5 multi-article', 'validation events.']):
                    text(c, 712, 300 - 20 * k, s, 13)
                continue
            if line == 'PIPELINE': pipeline(c, 300); y = 250; continue
            for l in textwrap.wrap(line, 100):
                text(c, 52, y, l, 17); y -= 24
            y -= 14
    c.showPage()
c.save()
print('Wrote learning-curves.pdf, figures.pdf, Defense.pdf')
