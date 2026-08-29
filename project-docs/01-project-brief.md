# Project brief — what this is and what is required

**Course:** BITS F471, *Introduction to Large Language Models*, BITS Pilani Goa
**Instructor:** Tanmay Tulsidas Verlekar (TTV) — research background is computer vision, biometrics, gait analysis
**Industry collaborator:** Manasi Patwardhan, Principal Scientist, TCS Research (AI Agents, GenAI, NLP, Program Synthesis)
**Weighting:** project 60% · comprehensive exam 30% (03/12/26) · class discussion 10%

## Team — Group 3

| Name | ID |
|---|---|
| Neel Naik (leader) | 2023A7PS0429G |
| Aryan Bethmangalkar | 2023A7PS0433G |
| Aaditya H Vernenker | 2023A7PS1027G |
| Aalhad M Sawane | 2023A7PS0476G |
| Arjun P Thakur | 2024A7PS0006G |
| Khushi Pandey | 2023B5A70951G |
| Diya Srivastava | 2023B2A40737G |

## The project format — "AI-augmented research cycle"

Six stages, each pairing a classic research step with an LLM-assisted technique:

1. **Idea extraction & validation** — pick an anchor paper, extract its core idea, validate it
2. **Reproduction** — reproduce the paper's reported results
3. **Idea enhancement** — LLM-driven brainstorming / literature search for a genuine improvement
4. **Implementation** — implement the improvement, try to beat the paper's baseline
5. **Paper writing** — LLM-assisted manuscript
6. **Review & refinement** — LLM-assisted critique and iteration

## Anchor paper

**Scaling Diffusion Language Models via Adaptation from Autoregressive Models** (DiffuGPT / DiffuLLaMA)
Gong, Agarwal, Zhang, Ye, Zheng, Li, An, Zhao, Bi, Han, Peng, Kong — **ICLR 2025**
[arXiv:2410.17891](https://arxiv.org/abs/2410.17891) · upstream code `HKUNLP/DiffuLLaMA` · our fork `jarjlol/diffullama`

Converts pretrained autoregressive LMs into masked diffusion LMs via continual pre-training, using
attention mask annealing and a shift operation. Produces DiffuGPT (124M/355M) and DiffuLLaMA (6.74B).

**Topic constraint from TTV:** anchor paper must be NLP or CV, deep-learning/GenAI emphasis, published
within roughly the last 2 years.

## Deliverables and status

| Deliverable | Requirement | Status |
|---|---|---|
| Assignment Part 1 | Choose + understand an anchor paper (post-Jan-2024, top venue) | ✅ done — DiffuLLaMA |
| Assignment Part 2 | Describe a *unique* automatable research-lifecycle task | ✅ **submitted** — "Continuous Gap Liveness Verification (Automated Novelty Invalidation)" |
| SOTA assignment | Implement a 2024+ literature-review-generation method, generate a review for the anchor's domain, compare overlap with the anchor's related-work section | ✅ implemented (`litreview/`), ❌ **not submitted — deadline not yet announced** |
| Main project | Stages 2–6 above | ⏳ **blocked on direction decision** |

### SOTA assignment detail

TTV provided **two** papers and allowed either, or an alternative 2024+ top-tier NLP paper:
- [2508.17647](https://arxiv.org/abs/2508.17647) *SurveyGen: Quality-Aware Scientific Survey Generation* → **QUAL-SG implemented**
- [2412.13612](https://arxiv.org/abs/2412.13612) *LLMs for Automated Literature Review* (an **evaluation** framework) → **not implemented; available as an addition**

## Target venue — status uncertain

The project brief names the **Stanford Agents4Science** workshop.

- Its AI-authorship rule is confirmed and unusual: *"AI authorship is not only allowed but required"* —
  AI as both primary author and reviewer. TTV has independently confirmed everything must be AI.
- **But:** the 2025 edition closed submissions 5 Sept 2025 and held its event 22 Oct 2025. **No 2026
  edition has been found.** See `04-open-questions.md` Q-1.

## Compute available

### Shared compute

| Resource | Spec | Status |
|---|---|---|
| Workstation | **2× RTX 6000 Pro Blackwell, 96 GB each (192 GB total)** | ✅ confirmed — **this is the real compute; plan around it** |
| Sharanga cluster (BITS Hyderabad) | see below | ⚠️ **access not approved; availability far narrower than raw specs suggest** |

### Per-member machines

Seven people, seven different machines. **"The development laptop" is not one thing** — what runs locally
depends on whose machine, so record them individually.

| Member | GPU | VRAM | Verified? | Realistically runs |
|---|---|---|---|---|
| Aaditya | RTX 3050 Laptop | 4 GB | ✅ `nvidia-smi` 2026-08-29 | tokenizer work, DiffuGPT-S (124M) |
| Neel | RTX 3060 | 6 GB | ⚠️ per design doc §5.5, not independently checked | DiffuGPT-S/M |
| Aryan | ? | ? | ❌ unknown | — |
| Aalhad | ? | ? | ❌ unknown | — |
| Arjun | ? | ? | ❌ unknown | — |
| Khushi | ? | ? | ❌ unknown | — |
| Diya | ? | ? | ❌ unknown | — |

**To fill in your row:** run `nvidia-smi --query-gpu=name,memory.total --format=csv` and paste the result.
If you have no NVIDIA GPU, say so — CPU-only is fine for the tokenizer and AST work, which is a real chunk
of the remaining tasks.

**Also needs recording:** who can reach the 2× Blackwell workstation, and how (physical access? SSH? shared
account?). That determines who can actually run experiments once a direction is chosen.

> **Note on §5.5 of the design doc:** it plans local development against "a 6 GB RTX 3060." That is
> probably accurate for Neel's machine but is **not** a team-wide baseline — Aaditya's is 4 GB. Any
> "runs locally" claim should name whose machine it was tested on.

### Sharanga — inventory is not availability

The cluster's [configuration page](https://sharanga.hpc.bits-hyderabad.ac.in/docs/misc_docs/configuration/)
lists: gpu1–3 (V100 32GB), gpu4 (8× A100 80GB), gpu5–6 (4× H100 80GB each), gpu7 (8× H200 NVL 141GB),
gpu8 (6× RTX PRO 6000 Blackwell 96GB).

**That is the hardware inventory, not what a student job can obtain.** Per the team:

- It is **SLURM-scheduled** — jobs queue; you do not get interactive access to a whole node on demand.
- **You cannot request all GPUs on a node at once** in practice.
- **The Blackwell node (gpu8) is reserved for admins** and is not available to us.

The configuration page documents **no** access policy, which should be read as *unknown*, not as
*unrestricted*. Partition names, per-user GPU caps, and wall-time limits are still undocumented — ask when
requesting access.

**Planning consequence:** treat Sharanga as *possible batch capacity pending approval*, never as assumed
capacity. **The local 2× Blackwell workstation is the compute the project should be planned around.** It is
sufficient for the recommended direction.

## Software environment

`torch` and `transformers` are **not installed** on the development laptop. What each unblocked task
actually needs:

| Task | Needs | Size | Runs where |
|---|---|---|---|
| Real-tokenizer offset test (Q-6) | `transformers` only — `AutoTokenizer` needs no torch and no GPU | ~50 MB | laptop, today |
| DiffuGPT-S (124M) mechanics testing | `torch` + `transformers` | ~2.5 GB | laptop (4 GB fits a 124M model) |
| Any real experiment | full CUDA stack | — | Blackwell workstation |
| Batch jobs | conda/venv in home dir + SLURM scripts | — | Sharanga, if approved |
