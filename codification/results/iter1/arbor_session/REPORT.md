# Research Report: Goal: maximize `score` from `bash eval.sh dev`. Merges are gated on `bash eval.sh test` (a held-o...

_Generated 2026-11-04 02:51_   _Exit: `ok`_

## Instruction

> Goal: maximize `score` from `bash eval.sh dev`. Merges are gated on `bash eval.sh test` (a held-out gate split). Only edit solution.py. Never touch task.py, eval.py, eval.sh, baselines.py, research_config.yaml or anything under data/.
> 
> What the first trials showed (iteration 1, GSM8K, N = 10 cached candidates per question):
> 1. The idea as specified fails where it matters. On Dream (dev: first sample 37.5%, oracle 84.5%) picking the candidate with the lowest AR-scorer NLL is WORSE than the first sample: 23.5-36.0% depending on scorer, with summed NLL the worst because it prefers short answers. Majority vote over the extracted answers reaches 70.0%.
> 2. The correct answer is usually among the candidates (oracle 85-92% on Dream and LLaDA), so the job is choosing among answers, not finding new ones.
> 3. DiffuGPT-S/M and DiffuLLaMA are at the floor (first sample under 2%, oracle 7-13%). Gains there are a few questions; do not tune for them. The macro score is driven by Dream and LLaDA.
> 4. Features per candidate: ar.<scorer>.nll_sum and ar.<scorer>.n_tokens for gpt2, qwen0.5b, tinyllama, qwen7b (Qwen2.5-7B, Dream's own AR base) and llama2-7b (Llama-2-7B, DiffuLLaMA's own AR base). The 7B scorers are the strongest; the small ones are weak.
> 
> Direction: start from majority vote, which already beats every AR selector. Use AR likelihood only where it can add information: to break ties between answers with equal votes, or to weight votes (for example, score each distinct answer by the sum over its candidates of exp(-per-token NLL)). Prefer per-token NLL to summed NLL. Keep select() deterministic and simple; a rule you can explain beats a tuned one. Report dev and gate scores per model, not only the macro average.

**Model**: `Qwen/Qwen3.8-27B-FP8`
**Duration**: 1h08m01s

## Event Summary

| Metric | Value |
|--------|-------|
| Cycles completed | 3 |
| Ideas proposed | 7 |
| Ideas completed | 2 |
| Ideas pruned | 2 |
| Ideas merged | 1 |
| Sub-agent runs | 3 |
| LLM errors | 0 |
| Eval failures | 0 |
| Wall clock | 1h 8m 1s |

## Run Stats

_Token scope_: `research_agent_process_llm_only; excludes LLM calls made by external Bash/RunTraining/eval subprocesses`

| Metric | Value |
|--------|-------|
| Total LLM calls | 132 |
| Total agents spawned | 4 |
| Total input tokens | 5,284,750 |
| Total uncached input tokens | 5,284,750 |
| Total output tokens | 142,780 |
| Coordinator turns | 69 |

## Results

**Baseline → Final**: `0.1480` → `0.2960` (+100.00%)

**Test set**: baseline=`0.157` final=`0.296`

## Exploration

_8 nodes total, 4 scored, 1 merged_

### Merged Ideas

- **4.1** (`0.296`): Mechanism: verification-weighted voting — for each candidate (on its trimmed text: everything before a line matching /\n\s*(?:Question|Q)\s*:/), parse explicit calculations with the regex "<<\s*(num)\s*([+\-*/])\s*(num)\s*=\s*(num)\s*>>" and "(num)\s*([+\-*/x×])\s*(num)\s*=\s*(num)" (num = -?[\d,]+(\.\d+)?; 'x'/'×' treated as *; divisions by 0 skipped, unparsable pairs skipped); re-evaluate each exactly and flag candidates having >=1 checked calculation with at least one mismatch (or none at all) as unverified. Weight = 1.0 if all checked calculations pass (and >=1 was checked), else 0.5. Answer score = sum of weights of its candidates; select the highest; ties -> earliest candidate index; all answers None -> 0. Uniform across all 5 models; deterministic, no state.

### Top Ideas by Score

- **4.1** `0.2960` _merged_: Mechanism: verification-weighted voting — for each candidate (on its trimmed text: everything before a line matching /\n\s*(?:Question|Q)\s*:/), parse explicit calculations with the regex "<<\s*(num)\s*([+\-*/])\s*(num)\s*=\s*(num)\s*>>" and "(num)\s*([+\-*/x×])\s*(num)\s*=\s*(num)" (num = -?[\d,]+(\.\d+)?; 'x'/'×' treated as *; divisions by 0 skipped, unparsable pairs skipped); re-evaluate each exactly and flag candidates having >=1 checked calculation with at least one mismatch (or none at all) as unverified. Weight = 1.0 if all checked calculations pass (and >=1 was checked), else 0.5. Answer score = sum of weights of its candidates; select the highest; ties -> earliest candidate index; all answers None -> 0. Uniform across all 5 models; deterministic, no state.
- **1** `0.2940` _done_: Direction: answer-level self-consistency — decide by consensus over the 10 extracted candidate answers (vote counting) instead of ranking candidates by AR likelihood. The brief's iteration-1 evidence: majority already beats every AR selector; the job is choosing among answers.
- **2.1** `0.2900` _pruned_: Mechanism: majority vote with qwen7b per-token NLL tie-break — count non-None answers; among answers TIED for the top count, pick the answer whose best candidate has the lowest per-token NLL (min of ar.qwen7b.nll_sum/ar.qwen7b.n_tokens over that answer's candidates); remaining ties/missing features -> earliest candidate index; all-None -> 0.
- **3.1** `0.2820` _pruned_: Mechanism: likelihood-weighted voting (falsification) — score each distinct extracted answer by sum of exp(-per-token NLL) over its candidates; per-token NLL = ar.<scorer>.nll_sum/ar.<scorer>.n_tokens; scorer: dream-base -> qwen7b, diffullama -> llama2-7b (its own AR base), all other models -> qwen7b; pick highest-scoring answer; ties -> earliest index; all-None -> 0.

## Artifacts

- Coordinator final report: `/home/antpc/arbor-rerank/.arbor/sessions/iter1b/COORDINATOR_FINAL_REPORT.txt`
- Interactive Q&A transcript: `/home/antpc/arbor-rerank/.arbor/sessions/iter1b/conversation.md`
- Idea tree (JSON): `/home/antpc/arbor-rerank/.arbor/sessions/iter1b/.coordinator/idea_tree.json`
- Idea tree (Markdown): `/home/antpc/arbor-rerank/.arbor/sessions/iter1b/.coordinator/idea_tree.md`
- Run stats: `/home/antpc/arbor-rerank/.arbor/sessions/iter1b/run_stats.json`
- Event log: `/home/antpc/arbor-rerank/.arbor/sessions/iter1b/events.jsonl`
