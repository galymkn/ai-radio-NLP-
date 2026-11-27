"""Train and evaluate an event-pair classifier, or score two new headlines."""
import argparse
import json
import platform
import re
import resource
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
FEATURES = ['lead cosine', 'headline cosine', 'word overlap', 'digit overlap']
def words(row):
    return set(re.findall(r'\w+', row['title'].casefold()))
def overlap(a,b):
    return len(a & b) / max(1,len(a | b))
def features(i,j,rows,title,full):
    a,b=words(rows[i]),words(rows[j])
    return [float(full[i] @ full[j]),float(title[i] @ title[j]),overlap(a,b),
            overlap({w for w in a if w.isdigit()},{w for w in b if w.isdigit()})]
def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-50,50)))
def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def score(y,p,t):
    labels=p>=t
    prec,rec,f,_=precision_recall_fscore_support(y,labels,labels=[0,1],zero_division=0)
    return dict(n=len(y),positive=int(y.sum()),cm=confusion_matrix(y,labels,labels=[0,1]).tolist(),
                precision=prec.tolist(),recall=rec.tolist(),f1=f.tolist(),ap=float(average_precision_score(y,p)))
def run():
    rows=json.loads((ROOT/'dataset.json').read_text())
    z=np.load(ROOT/'embeddings.npz')
    assert list(z['ids'])==[a['id'] for a in rows]
    title,full=z['title'],z['full']
    data={}
    for split in ('train','validation','test'):
        ids=[i for i,a in enumerate(rows) if a['split']==split]
        pairs=list(combinations(ids,2))
        x=np.array([features(i,j,rows,title,full) for i,j in pairs])
        y=np.array([int(rows[i]['event']==rows[j]['event']) for i,j in pairs])
        data[split]=(x,y,pairs)
    x,y,_=data['train'];v,vy,_=data['validation']
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-6)
    x=(x-mu)/sd;v=(v-mu)/sd
    # Balanced binary cross-entropy gives each class half the total weight.
    def weights(labels):
        return np.where(labels,len(labels)/(2*labels.sum()),len(labels)/(2*(len(labels)-labels.sum())))
    def loss(logits,labels):
        return float(np.mean(weights(labels)*(np.logaddexp(0,logits)-labels*logits)))
    w=np.zeros(x.shape[1]);b=0.;hist=[];best=None
    start=time.perf_counter()
    for epoch in range(1,201):
        residual=weights(y)*(sigmoid(x@w+b)-y)
        w-=.05*(x.T@residual/len(y)+.001*w);b-=.05*residual.mean()
        val_logits=v@w+b
        hist.append(dict(epoch=epoch,train_loss=loss(x@w+b,y),val_loss=loss(val_logits,vy),
                         val_f1_at_half=score(vy,sigmoid(val_logits),.5)['f1'][1]))
        if best is None or hist[-1]['val_loss']<best[0]:best=(hist[-1]['val_loss'],epoch,w.copy(),b)
    _,epoch,w,b=best
    prob=sigmoid(v@w+b)
    t=float(max(np.arange(.05,1,.025),key=lambda t:(score(vy,prob,t)['f1'][1],-abs(t-.5))))
    train_s=time.perf_counter()-start
    model=dict(features=FEATURES,mean=mu.tolist(),scale=sd.tolist(),weights=w.tolist(),bias=float(b),threshold=t,epoch=epoch)
    save('checkpoint.json',model);save('history.json',hist)
    x,y,pairs=data['test'];start=time.perf_counter();p=sigmoid(((x-mu)/sd)@w+b);infer=time.perf_counter()-start
    res=dict(split={k:dict(articles=sum(a['split']==k for a in rows),pairs=len(d[1]),positive=int(d[1].sum())) for k,d in data.items()},
             baseline=score(y,x[:,0],.72),classifier=score(y,p,t),train_seconds=train_s,head_batch_seconds=infer,platform=platform.platform(),
             peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,epoch=epoch,threshold=t)
    save('results.json',res)
    save('test_predictions.json',[dict(ids=[rows[i]['id'],rows[j]['id']],label=int(yy),baseline=float(xx[0]),score=float(pp)) for (i,j),yy,xx,pp in zip(pairs,y,x,p)])
    print(json.dumps(res,indent=2))
def demo(a,b):
    from encode import encoder,vectors
    rows=[dict(title=s,lead='') for s in (a,b)]
    model=encoder();title,full=vectors(model,rows);x=np.array(features(0,1,rows,title,full))
    head=json.loads((ROOT/'checkpoint.json').read_text())
    p=float(sigmoid(((x-head['mean'])/head['scale'])@head['weights']+head['bias']))
    print(json.dumps(dict(inputs=[a,b],tokens=[model.tokenizer.tokenize(s)[:24] for s in (a,b)],features=dict(zip(FEATURES,x)),score=p,threshold=head['threshold'],same_event=p>=head['threshold']),ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--demo',nargs=2);args=parser.parse_args()
    if args.demo:demo(*args.demo)
    else:run()
