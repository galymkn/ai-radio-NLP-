"""Paired event-block bootstrap, conditional on this day's observed gold events."""
from pathlib import Path
import json
import numpy as np
r=Path(__file__).resolve().parent
def load(name):return json.loads((r/name).read_text())
rows={a['id']:a['event'] for a in load('dataset.json') if a['split']=='test'}
events=sorted(set(rows.values()))
pred=load('test_predictions.json');old=load('v1/test_predictions.json')
assert [a['ids'] for a in pred]==[a['ids'] for a in old]
i,j=np.array([[events.index(rows[k]) for k in a['ids']] for a in pred]).T
y=np.array([a['label'] for a in pred],bool)
methods=[]
for key,records,t in [('baseline',pred,.72),('score',old,load('v1/checkpoint.json')['threshold']),('score',pred,load('checkpoint.json')['threshold'])]:
 methods.append(np.array([a[key]>=t for a in records]))
rng=np.random.default_rng(73);scores=[]
for _ in range(3000):
 n=rng.multinomial(len(events),np.full(len(events),1/len(events)))
 weights=np.where(i==j,n[i],n[i]*n[j])
 if weights[y].sum()==0:continue
 scores.append([2*weights[y&m].sum()/(weights[y].sum()+weights[m].sum()) for m in methods])
a=np.array(scores)
out={k:np.quantile(vals,[.025,.975]).tolist() for k,vals in zip(['baseline','v1','v2','v2_minus_baseline'],[*a.T,a[:,2]-a[:,0]])}
out.update(valid_draws=len(a),seed=73,unit='test event',draws=3000)
(r/'intervals.json').write_text(json.dumps(out,indent=2));print(out)
