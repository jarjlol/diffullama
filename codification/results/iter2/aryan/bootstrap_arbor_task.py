from pathlib import Path
import collections,hashlib,json,os,shutil,subprocess
home=Path.home(); base=home/'codify/iter2/aryan'; code=base/'codification'
source=home/'arbor-rerank'; pack=base/'task-pack-v2'; dest=home/'arbor-rerank-iter2'
if pack.exists() or dest.exists():
    raise SystemExit('Refusing an existing pack or task destination')
pack.mkdir()
for name in ('task.py','eval.py','eval.sh','baselines.py','research_config.yaml','README.md','PROVENANCE.md'):
    shutil.copy2(code/'rerank_task'/name,pack/name)
for split in ('dev','test'):
    target=pack/'data'/split;target.mkdir(parents=True)
    # Already separated dev and merge-gate gold only. No final path is opened.
    source_items=[json.loads(s) for s in (source/'data'/split/'items.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    by_model={m:{r['uid'] for r in source_items if r['model']==m} for m in ('dream-base','llada-base')}
    assert all(len(uids)==200 for uids in by_model.values()) and by_model['dream-base']==by_model['llada-base']
    approved=by_model['dream-base']
    gold=[json.loads(s) for s in (source/'data'/split/'gold.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    gold=[r for r in gold if r['uid'] in approved]
    (target/'gold.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gold),encoding='utf-8')
    if len(gold)!=200 or len({r['uid'] for r in gold})!=200:
        raise SystemExit(f'{split}: expected 200 unique dev/gate questions')
(pack/'solution.py').write_text('''"""A3 bootstrap seed: ordinary majority; replace with N4 frozen tuned majority before search."""\nfrom collections import Counter\n\ndef select(item):\n    answers = [candidate.get("answer") for candidate in item["candidates"]]\n    counts = Counter(answer for answer in answers if answer is not None)\n    if not counts:\n        return 0\n    most = max(counts.values())\n    return next(index for index, answer in enumerate(answers) if answer is not None and counts[answer] == most)\n''',encoding='utf-8')
env=dict(os.environ,CODIFY_TASK_DIR=str(pack),CODIFY_WORK_DIR=str(home/'codify/diffullama/codification/pipeline/work'),HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='',GIT_AUTHOR_NAME='Aryan Bethmangalkar',GIT_AUTHOR_EMAIL='aryanbethman@gmail.com',GIT_COMMITTER_NAME='Aryan Bethmangalkar',GIT_COMMITTER_EMAIL='aryanbethman@gmail.com')
for split in ('dev','test'):
    subprocess.run(['python3',str(code/'pipeline/assemble.py'),'--split',split,'--models','dream-base','llada-base'],env=env,check=True)
    rows=[json.loads(s) for s in (pack/'data'/split/'items.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    assert collections.Counter(r['model'] for r in rows)=={'dream-base':200,'llada-base':200}
subprocess.run(['bash',str(code/'make_arbor_repo.sh'),str(dest)],env=env,check=True)
# README/config updates happen in this new task only; no protected evaluator changes.
(dest/'README.md').write_text('''# Arbor iteration 2 bootstrap\n\nStatus: NOT READY FOR SEARCH. Majority seed, dev200/gate200, Dream + LLaDA.\n\nBefore launch: complete cut AR and DLM self-score features; replace seed with N4 frozen tuned majority; integrate T1/T3/T4/N4 memory/plugin; confirm the evaluator change on issue 4.\n\nOnly solution.py is editable during search. Final300 remains outside this repository.\n''',encoding='utf-8')
manifest={'stage':'majority-bootstrap-not-ready','models':['dream-base','llada-base'],'seed':'ordinary majority, awaiting N4','source_ref':'iter2/aaditya','sha256':{name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in ('task.py','eval.py','eval.sh','baselines.py','solution.py')}}
(dest/'A3_BOOTSTRAP.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
for split in ('dev','test'):
    r=subprocess.run(['bash','eval.sh',split],cwd=dest,env=env,text=True,capture_output=True,check=True)
    print(r.stdout)
    assert 'correctness: PASS' in r.stdout
    (base/f'bootstrap_{split}.txt').write_text(r.stdout,encoding='utf-8')
subprocess.run(['git','add','README.md','A3_BOOTSTRAP.json'],cwd=dest,check=True)
subprocess.run(['git','-c','user.name=Aryan Bethmangalkar','-c','user.email=aryanbethman@gmail.com','commit','-qm','docs(arbor): record iteration-2 bootstrap limits'],cwd=dest,env=env,check=True)
print('Prepared',dest,'without opening final split or starting a GPU process')
