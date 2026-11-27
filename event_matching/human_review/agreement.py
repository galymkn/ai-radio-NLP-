"""Run after two independent reviewers complete both forms."""
import csv,itertools,json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
r=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader((r/name).open()))
def agreement(a,b):
 if len(set(a+b))<2:return 'undefined (constant ratings)'
 return round(cohen_kappa_score(a,b),4)
shared=json.loads((r/'shared.json').read_text())
def labels(n):
 rows=read(f'events_{n}.csv')
 if any(not a['your_event_id'].strip() for a in rows):raise SystemExit(f'Finish events_{n}.csv first')
 events={int(a['id']):a['your_event_id'].strip() for a in rows}
 return [int(events[i]==events[j]) for i,j in itertools.combinations(shared,2)]
print('Event agreement on 780 shared same/different pairs:',agreement(labels(1),labels(2)))
def verdicts(n):
 rows=sorted(read(f'sentences_{n}.csv'),key=lambda a:(int(a['clip_id']),int(a['index'])))
 if any(a['supported'].strip() not in ('0','1') for a in rows):raise SystemExit(f'Finish sentences_{n}.csv with 0 or 1')
 return [int(a['supported']) for a in rows]
a,b=verdicts(1),verdicts(2)
print('Sentence agreement:',agreement(a,b),'unsupported counts:',a.count(0),b.count(0),'out of',len(a))
