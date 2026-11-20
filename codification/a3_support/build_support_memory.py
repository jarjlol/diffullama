import hashlib,json,re
from pathlib import Path
root=Path.cwd(); cod=root/'codification'; out=cod/'a3_support/assets/memory'
raw=cod/'results/iter1/arbor_session/EXPERIENCE.md'
text=raw.read_text(encoding='utf-8')
lines=[]
for line in text.splitlines():
 if '**[leverage] Data structure: cross-model consensus availability**' in line:
  line='- **[pitfall] Forbidden cross-model cache** — Tier implement: rejected twice by the operator; violates R7. Never reuse the raw leverage recommendation, even when it appears to improve a score. One item’s own candidates only. Source: ITERATION_1.md §Arbor.'
 elif line.startswith('- **['):
  line=line.replace(' — ',' — Tier implement, iteration1 trim features only: ',1)
  if 'AR NLL features' in line:
   line+=' Cut scoring and native self-score are new conditions; do not extrapolate this negative to them.'
  if 'Remaining signal audit' in line:
   line+=' This historical ceiling is NOT a bound on new features or new candidate budgets.'
 lines.append(line)
(out/'EXPERIENCE_CURATED.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
records=[]
def add(id,source,hypothesis,result,verdict,outcome='NOT_FAILED',trigger='Same fixed-candidate task',repeat='Read evidence before repeating',tags=None,tier='implement'):
 path=root/source
 records.append(dict(id=id,date='2026-11-21',source=source,hypothesis=hypothesis,tier=tier,result=result,verdict=verdict,evidence=source,evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),outcome=outcome,trigger_conditions=trigger,do_not_repeat=repeat,retrieval_tags=tags or ['iter1','reranking']))
source='codification/results/iter1/numbers/baselines_dev.json'
baselines=json.loads((root/source).read_text(encoding='utf-8'))
for model,values in baselines.items():
 for name,value in values.items():
  if name in ('n','_headroom'):continue
  add('baseline:'+model+':'+name,source,name+' on '+model,dict(split='dev',model=model,n=values['n'],accuracy=value,json_pointer='/'+model+'/'+name),
      'Measured control; original trim scores, not cut scores.',
      'FAILED (WorseThanBaseline)' if value<values['majority'] else 'NOT_FAILED',
      trigger='Original iteration1 candidates/features, dev only',repeat='Use as control, not a fresh A3 hypothesis',tags=['baseline',model,name])
tree_path='codification/results/iter1/arbor_session/.coordinator/idea_tree.json'
tree=json.loads((root/tree_path).read_text(encoding='utf-8'))
for node in tree['nodes']:
 if not isinstance(node,dict):
  node=tree['nodes'][node]
 id=str(node.get('id',node.get('node_id','unknown')))
 result={k:node[k] for k in ('score','status','eval_status','result') if k in node}
 result.update(scope='historical tree metadata; inspect experiment report before treating prose as verified',split='dev; gate only where source explicitly states it')
 add('node:'+id,tree_path,node.get('hypothesis',''),result,'Historical node; no iteration2 progress claim.',
     'FAILED (WorseThanBaseline)' if node.get('score') is not None and node['score']<.294 else 'NOT_FAILED',
     repeat='Do not repeat unmodified plain majority/2.1/3.1/4.1 or cross-model cache; new cut/native features must be explicit.',tags=['arbor',id])
for issue in ('I-16','I-17','I-18','I-19','I-20'):
 line=next(s for s in (cod/'ITERATION_1.md').read_text(encoding='utf-8').splitlines() if s.startswith('| '+issue+' |'))
 add(issue,'codification/ITERATION_1.md',line,{'scope':'recorded operational issue, not new measurement'},'Retrieve recorded repair before debugging.',repeat='Do not repeat recorded defect.',tags=['operations',issue])
(out/'FINDINGS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
summary=['# Operator fallback findings memory','','T3 support. Built mechanically from repository evidence; no new paper claims, no surrogate LLM run.','Every record has an evidence path and SHA256; numeric baseline records additionally have an exact JSON pointer.','Historical trim-score negatives are provisional under the new cut/native-feature conditions.','The raw tree contains stale dataset-size prose; task manifest, not that prose, defines current splits.','', '## Promotion rule','1. Idea: an audited proposal with a falsifiable observable.','2. Implement: executable and measured on dev under the fixed harness.','3. Gate: >=3 net questions on EACH model versus frozen tuned majority, paired tests, external R7 check and ablation.','4. Progress: once-only frozen final comparison plus an agreed frozen second-dataset transfer and compute accounting.','5. Otherwise keep the result at implement, including negative or inconclusive evidence.','','## Implement findings']
for r in records:
 if r['id'].startswith('baseline:'):continue
 summary.append('- '+r['id']+': '+r['verdict']+' Evidence: '+r['evidence'])
summary += ['',f'Baseline inventory: {sum(r["id"].startswith("baseline:") for r in records)} per-model selector/control records in FINDINGS.jsonl, dev n=200 each.','No record is promoted to progress. Current GPU priority follows the authoritative team tiers, not invented UCB scores.']
(out/'FINDINGS.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
for r in records:
 assert (root/r['evidence']).is_file()
 if 'json_pointer' in r['result']:
  _,model,name=r['result']['json_pointer'].split('/')
  assert baselines[model][name]==r['result']['accuracy']
print('Evidence records:',len(records),'all paths/hashes and baseline numeric pointers checked')
