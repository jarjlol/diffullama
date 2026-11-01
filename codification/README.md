# Codification — idea 1.1 with Arbor

**Idea.** Idea 1.1 (P1-M1) is chosen in `IDEA_SELECTION.md`. A diffusion LM (DLM) samples
N = 10 answers to each GSM8K question, and a small autoregressive (AR) scorer picks one.
The idea claims that the accuracy this recovers is predicted by how much of its AR base's
representation the DLM preserves (layer-wise linear CKA).

**System.** Arbor is the only reference system with a complete implementation
(`REFERENCE_SYSTEMS.md`). Arbor's LLM, a local vLLM server, edits `rerank_task/solution.py`
(the selection rule) and iterates on dev. It keeps a change only when held-out test improves.

## Layout

```
rerank_task/          the Arbor task pack (becomes its own git repo; see make_arbor_repo.sh)
  task.py             PROTECTED  answer extraction + grading + data loading
  eval.py, eval.sh    PROTECTED  prints `score: <macro accuracy>`; correctness + leakage gates
  baselines.py        PROTECTED  first / random / majority (self-consistency) / AR sum / AR mean / oracle
  solution.py         EDITABLE   seed = lowest summed GPT-2 NLL (idea 1.1 as specified)
  data/{dev,test}/    gold.jsonl (committed), items.jsonl (made by the pipeline, git-ignored)
  research_config.yaml  Arbor config: local vLLM endpoint, budget, merge threshold
pipeline/             GPU side: makes items.jsonl and the CKA scores
  config.json         models, decoding settings, scorers, CKA corpus
  prepare_data.py     1  GSM8K -> dev (500 from train) / test (1319) / 4 shots     [done]
  generate_candidates.py  2  N = 10 samples per question, one DLM per process
  score_candidates.py 3  AR NLL features per candidate, one scorer per process
  assemble.py         4  merge into rerank_task/data/<split>/items.jsonl
  measure_preservation.py 5  per-layer CKA DLM vs AR base (calls extract_hidden.py twice)
  analyze.py          7  CKA vs reranking headroom, Spearman + exact permutation p
  selftest.py         CPU-only end-to-end wiring test with mock models
make_arbor_repo.sh    6  export the filled pack to a clean git repo and check the seed runs
```

`python3 pipeline/selftest.py` passes on a laptop with no GPU. It runs every step above,
apart from the real models, against temp copies.

## What is to be done

### A. Before the workstation (anyone)

1. **Request Llama-2 access** for `meta-llama/Llama-2-7b-hf` on Hugging Face. It is gated
   and approved by hand, and it is needed only as DiffuLLaMA's AR base for CKA. Without it,
   DiffuLLaMA has no CKA score and the correlation drops to three models.
2. **Novelty check (issue 6).** Start with the anchor authors' later work, then the two
   litreview papers that already rerank DLM output with an AR model. The claim to defend is
   the preservation → gain link, not reranking itself.
3. **Team confirmation of idea 1.1** (decision D-2026-11-01-b).

### B. On the workstation (teammate with the RTX PRO 6000)

The three model families pin incompatible versions: the anchor code needs transformers
4.44.2, Dream 4.46.2, and LLaDA's remote code 4.38.2. Use one venv per family, plus one
for the scorers:

| venv | models | notes |
|---|---|---|
| `anchor` | diffugpt-s/m, diffullama | repo `model.py` + `attention_patch.py`; flash-attn optional |
| `dream` | dream-base | `trust_remote_code` |
| `llada` | llada-base | fetches LLaDA `generate.py` at the pinned commit into `pipeline/third_party/` |
| `scorer` | AR scorers, AR bases for CKA | any recent transformers; must not import `attention_patch` |

Blackwell (sm_120) needs a CUDA 12.8+ build of torch. The old pins (torch 2.1.1) will not
run on it, so install a current torch in each venv and keep only the transformers pin.
**This combination is untested.** Try `--limit 5` on each backend first.

```bash
cd codification/pipeline
export HF_HOME=/path/to/hf-cache   CODIFY_WORK_DIR=/fast/disk/codify-work   # optional
python3 prepare_data.py            # step 1 (gold files are already committed; this rewrites them identically)

# step 2: one model per process, in its venv; resumable. Smoke-test with --limit 5 first.
for m in diffugpt-s diffugpt-m diffullama; do ~/venvs/anchor/bin/python generate_candidates.py --model $m --split dev; done
~/venvs/dream/bin/python generate_candidates.py --model dream-base --split dev
~/venvs/llada/bin/python generate_candidates.py --model llada-base --split dev
#   ...then the same with --split test

# step 3: every (model, split, scorer)
for m in diffugpt-s diffugpt-m diffullama dream-base llada-base; do for s in gpt2 qwen0.5b tinyllama; do for sp in dev test; do
  ~/venvs/scorer/bin/python score_candidates.py --model $m --split $sp --scorer $s; done; done; done

python3 assemble.py --split dev && python3 assemble.py --split test      # step 4
python3 ../rerank_task/baselines.py --split test                        # baseline table
```

Step 5 (CKA): `measure_preservation.py --python X` runs both roles with interpreter X. The
DLM role needs its family venv, while the AR-base role can use any venv. Simplest is to
run `extract_hidden.py` by hand:

```bash
~/venvs/anchor/bin/python extract_hidden.py --model diffullama --role dlm
~/venvs/scorer/bin/python extract_hidden.py --model diffullama --role base
python3 measure_preservation.py --skip-extract
```

Repeat for each model with a base. LLaDA is recorded as the control automatically.

GPU sharing: never kill other users' jobs. Check `nvidia-smi` first. The candidate
generation and Arbor's vLLM server can share the 96 GB card if vLLM's
`--gpu-memory-utilization` is lowered, for example to 0.5. Generation is finished before
Arbor starts anyway, since Arbor's loop is CPU-only.

### C. Run Arbor (the codification loop)

```bash
bash codification/make_arbor_repo.sh ~/arbor-rerank     # copies pack, runs seed eval, git init + commit
# start vLLM as in ideation/local/serve_models.sh (generator model only is enough)
pip install arbor-agent
cd ~/arbor-rerank && arbor config show && arbor        # intake reads README.md; confirm, then "go"
```

Or run it headless:

```bash
arbor run "maximize score from 'bash eval.sh dev'; gate merges on 'bash eval.sh test'; only edit solution.py; never touch task.py, eval.py, eval.sh, baselines.py or data/" --yes --yes-cwd .
```

`research_config.yaml` points Arbor at `http://127.0.0.1:8000/v1` with `provider:
openai-chat`, the type Arbor documents for self-hosted vLLM. Change `model:` to whatever
vLLM is serving.

**Spec gaps and refinement (the brief's "give feedback to the model").** Use
`ui.interaction_mode: review` or `collaborative` and record what you told Arbor. Seed the
first direction with the issues in `IDEA_SELECTION.md`: summed-NLL length bias,
agreement/majority signal, combining scorers. Keep Arbor's session folder
(`.arbor/sessions/<run>/`) as evidence of the refinement loop.

### D. Results and the SOTA / baseline comparison

1. **Per model on GSM8K test** (`baselines.py --json`, after Arbor with the final
   `solution.py`):
   - accuracy of first sample, majority vote, AR-sum, AR-mean, Arbor's selector, and oracle;
   - headroom, i.e. the share of the first-to-oracle gap closed.

   **Beating majority vote** is the baseline bar: it is self-consistency, the anchor's own
   Table 2 baseline.
2. **Against published numbers.** Compare first-sample accuracy with each model's reported
   GSM8K score, to check our generation is faithful. Compare the selector with published
   test-time methods for diffusion LMs. Take every published number from the primary PDF,
   with its shot count (`docs/verification-standards.md`); none are filled in here yet.
3. **The idea's own claim:** `python3 pipeline/analyze.py --split test`. This gives
   Spearman(CKA, headroom) over the ≤ 4 models with an AR base, with an exact permutation p
   (minimum 1/24 at n = 4). Report it as descriptive only (issue 3).

## Not implemented / known limits

- `dlm.selfscore`, the DLM's own-likelihood reranker from issue 1, is not computed. The
  harness accepts it if added to `score_candidates.py`.
- HumanEval and WinoGrande, also named in the idea, are out of scope. GSM8K only.
- Decoding settings differ between models (each follows its authors' recommendation). This
  is a stated confound.
- Nothing has run on a GPU yet. Every GPU path is written against the model code but has
  not been executed.
