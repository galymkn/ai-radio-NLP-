import json,textwrap
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
r=Path(__file__).resolve().parent;h=json.loads((r/'history.json').read_text())
def plot(c,x,y,w,height,keys,colors):
 c.setStrokeColor(HexColor('#aaa'));c.rect(x,y,w,height)
 for key,color in zip(keys,colors):
  c.setStrokeColor(HexColor(color));c.setLineWidth(2);p=c.beginPath()
  for i,row in enumerate(h):
   xx=x+w*i/(len(h)-1);yy=y+height*row[key]
   if i:p.lineTo(xx,yy)
   else:p.moveTo(xx,yy)
  c.drawPath(p)
 marker=x+w*(534/(len(h)-1));c.setStrokeColor(HexColor('#666666'));c.setDash(2,2);c.line(marker,y,marker,y+height);c.setDash();c.drawString(marker+3,y+height-13,'535')
 c.setFont('Helvetica',10);c.drawString(x-17,y,'0');c.drawString(x-17,y+height-4,'1');c.drawString(x,y-15,'Epoch 1');c.drawRightString(x+w,y-15,str(len(h)))
def curves(c):
 plot(c,70,100,385,205,['train_loss','val_loss'],['#008476','#c05226'])
 plot(c,530,100,340,205,['val_f1_at_half'],['#404bb0'])
 c.setFont('Helvetica',14);c.drawString(70,323,'Balanced loss   /   green: training   orange: validation');c.drawString(530,323,'Validation F1 at score threshold 0.50')
size=(960,540)
c=Canvas(str(r/'learning-curves.pdf'),pagesize=size);curves(c);c.save()
slides=[
('One event, several headlines','01:00',['RSS and Telegram > event matching > Claude script and verification > speech synthesis > web radio','Input: two reports in English or Russian. Output: a same-event score.','A missed match repeats a story. A false match can mix unrelated facts in one broadcast.']),
('282 articles across two collection days','01:30',['Public RSS and Telegram news. AI-assisted labels group coverage by specific incident or decision.','Train: 100 articles, 18 positive pairs (0.36%). Validation: 46 articles, 19 positive pairs.','Test: 136 later articles, 23 matches in 9,180 pairs. Form pairs after splitting events.','Publisher text and provisional labels limit redistribution and generalization.']),
('From text to five learned parameters','02:00',['Headline and headline plus lead → tokenizer → frozen multilingual MiniLM → mean pooling.','Two normalized 384-dimensional vectors per article. Token limit: 128 per view.','Compare the two views by cosine. Add headline word and number overlap.','Standardize four features, apply four weights and one bias, then sigmoid.']),
('Training and checkpoint selection','02:00',[]),
('The test set was evaluated twice','01:10',['Baseline: F1 0.537 (95% CI 0.276 to 0.718), 11 correct matches, 7 false positives.','v1: F1 0.250 (0.123 to 0.392), 18 correct matches, 103 false positives.','v2: F1 0.237 (0.115 to 0.383), 20 correct matches, 126 false positives.','Same 23 positive pairs in 9,180 test pairs. Paired event bootstrap, 3,000 draws.']),
('Why did validation and test disagree?','00:50',['Validation F1: baseline 0.190, v2 0.780. Test F1: baseline 0.537, v2 0.237.','v2 precision 0.137 and recall 0.870. The threshold transferred poorly to the next day.','Positive prevalence changed from 1.84% on validation to 0.25% on test.','Shorter RSS summaries could also affect scores. Check matched sources and lengths across several new days.']),
('Live demonstration','01:30',['Enter two unseen headlines using match.py --demo. Show tokens, features, score and decision.','Show where an event appears in the running station: article sources, verified script, audio queue.','The experiment favors the fixed cosine rule for F1. The learned model is available for pair-level inference.'])]
c=Canvas(str(r/'Defense.pdf'),pagesize=size)
for i,(title,dur,lines) in enumerate(slides):
 c.setFillColor(HexColor('#f8f7f2'));c.rect(0,0,*size,stroke=0,fill=1);c.setFillColor(HexColor('#183734'));c.setFont('Helvetica',14);c.drawString(50,493,'NEWSRADIO    /    NLP DEFENSE');c.drawRightString(908,493,dur)
 c.setFont('Helvetica-Bold',32);c.drawString(50,428,title)
 if not lines:
  curves(c);c.setFont('Helvetica',15);c.drawString(50,374,'Full batch gradient descent, balanced BCE, v2 LR 0.5, L2 0.001, 2000 epochs.')
  c.drawString(50,58,'v1 stopped at its final-epoch validation minimum (200). v2 continued past its minimum (535).')
  c.drawString(50,32,'Checkpoint and threshold selection used validation only. Both test evaluations are reported.')
 else:
  y=354
  for line in lines:
   for l in textwrap.wrap(line.replace('→','>'),95,replace_whitespace=False):
    c.setFont('Helvetica',18);c.drawString(52,y,l);y-=27
   y-=17
 c.showPage()
c.save()
