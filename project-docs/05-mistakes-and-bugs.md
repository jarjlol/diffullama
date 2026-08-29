# Mistakes, bugs, and near-misses

Every error made on this project, with what caused it and how to avoid repeating it.

**This file exists because the same *classes* of mistake keep recurring.** Three of the entries below are
the same bug pattern appearing three separate times. Read this before starting new work.

---

## The three recurring patterns

| Pattern | How many times | Cost if unfixed |
|---|---|---|
| **Trusting a summary instead of the primary source** | 5× | Wrong citations, wrong design decisions |
| **Off-by-N in hand-built index mappings** | 3× | Silent, invisible corruption |
| **Documentation claiming behaviour the code does not have** | 2× | Building on a false premise |

---

## A. Trusting summaries over primary sources

### M-1 — "InspAIred" misattributed as NAACL 2025 Best Paper
**What happened:** a search-engine summary named it the Best Paper. It is a co-located *workshop* paper;
the real winner is "The BiGGen Bench."
**Caught by:** fetching the official NAACL awards page directly.
**Rule:** verify award and venue claims against the conference's own page. Never a search snippet.

### M-2 — Depth Anything V2's "transparent surfaces" gap did not exist
**What happened:** an automated pass flagged transparent surfaces as an open failure mode, based on
abstract-level retrieval. The paper itself reports 83.6% zero-shot / 91.2% fine-tuned accuracy there —
it had already solved it using synthetic training data.
**Caught by:** reading the full paper text. The real remaining gap was narrower (mirrors specifically,
a physically distinct failure mode).
**Rule:** read full text before committing to a claimed gap. Abstracts hide solved problems.

### M-3 — The design doc's sampler claim was wrong in both directions
**What happened:** the doc stated "LLaDA/Dream unmask lowest-confidence-first." In reality LLaDA does
`torch.topk(confidence)` — **highest**-first; the config name `remasking='low_confidence'` refers to which
tokens *stay* masked. And Dream ships `"alg": "origin"` — **random** order.
**Root cause:** the config *name* was read instead of the code it controls.
**Cost if unfixed:** the entire motivation for §4.1's sampler-normalization work, plus a silent-failure
mode where any Dream baseline runs random-order without anyone noticing.
**Rule:** read the code a config name controls, not the name.

### M-4 — CDC Fig. 8(b) framed as "tighter is better"
**What happened:** the audit initially reported "a tight AST neighbourhood beats the broader use-def
slice," which is true but incomplete. Parent+Leaf (34.3) beats the looser slice (26.9) **and** the tighter
token-window (24.1) — it is a **middle optimum**. Also, the metric is on a *security* benchmark, so
transfer to functional bugs is unestablished.
**Caught by:** reading the figure personally instead of relying on the automated pass that found it.
**Rule:** if a single number is driving a recommendation, read it yourself.

### M-5 — Recommending "abandon backward slicing" from an unread figure
**What happened:** the recommendation to move off backward slicing rested for two days on a figure known
only through an automated summary.
**Rule:** load-bearing claims get personally verified before they are acted on, not after.

---

## B. Off-by-N in hand-built index mappings

### M-6 — 222 relevance scores for 218 papers
**What happened:** an LLM-relevance score array was hand-built with 4 extra entries. If used, every score
after the drift point would have silently attached to the wrong paper.
**Caught by:** asserting `len(scores) == len(pool)` before use.
**Fixed by:** rebuilding as an explicit `{index: score}` map, which fails safe — one wrong entry instead
of a cascading shift.
**Rule:** never hand-build a positional array. Use an explicit key→value map, and assert length before use.

### M-7 — The original relevance scorer's off-by-N (pre-existing)
Documented in `litreview/REPORT.md` §4: an earlier version scored candidates by eyeballing row indices and
had a row-counting bug. Discarded in favour of content-matching.
**Same class as M-6, in the same project, months apart.**

### M-8 — The shift off-by-one (latent, would have been silent)
**What happened:** DiffuLLaMA's inherited AR shift means `canvas_position = returned_index + 1`. A naive
mapping masks the wrong token, runs end-to-end without error, and quietly underperforms.
**Status:** caught before any code depended on it. `audit/test_shift_offset.py` proves the invariant and
demonstrates the failure concretely (targeting `returned[2]` masks `'b'` when `'c'` was intended).
**Rule:** the design doc was right to rank this Critical. Write invariant tests before selectors.

---

## C. Documentation that did not match the code

### M-9 — `merge_finalize.py` claimed a stratification it did not implement
**What happened:** `REPORT.md` described the script as stratifying foundational vs recent papers before
ranking. The committed code was a flat top-K cut. Running it as-committed reproduced the exact
recency-dilution failure the report already diagnosed once — 0-citation 2025-26 preprints (including a
paper on *tire architecture design*) out-ranking Mamba and DiffusionBERT.
**Rule:** when a document describes code behaviour, run the code and check.

### M-10 — "DiffuLLaMA already computes the needed scores, so this is a small change"
**What happened:** the design doc's §4.1 claimed confidence scores were readily available. In fact
`x0_scores` is the *sampled-token* log-prob (not max probability), and it is **never returned** —
`model.py:158` returns only `x0`, and no per-step history is recorded at all.
**Cost if unfixed:** a confidence baseline built on it would be unfairly weak, **biasing results in favour
of the proposed method** — exactly what a reviewer looks for.

---

## D. Plain bugs

### M-11 — Windows `cp1252` decode crashes (twice)
`json.load(open(path))` and `open(path).read()` without `encoding='utf-8'` crash on any non-ASCII
character in a paper title or abstract. Hit once in `common.py`, once in `evaluate.py`.
**Rule:** always pass `encoding='utf-8'` explicitly. The platform default is not UTF-8 on Windows.

### M-12 — Retrieval queries missed papers by phrasing
**What happened:** six ground-truth papers (D3PM, SEDD, Argmax Flows, and others) were **never retrieved
at all** — not mis-ranked, never fetched — because query phrasing did not match their titles
("discrete state-spaces", "estimating the ratios of the data distribution").
**Rule:** a recall failure looks identical to a relevance failure in the output. Check the raw pool before
concluding your ranker is at fault.

### M-13 — Keyword relevance scorer admitted off-topic domains
**What happened:** the regex scorer's off-topic penalty is skipped whenever a paper also matches a strong
term like "discrete diffusion" — which protein, RNA, graph, and even tire-design papers do incidentally.
**Fixed by:** real semantic relevance judgment (F1 0.222 → 0.272, zero off-topic papers).

---

## E. Process errors

### M-14 — Hardware objection asserted before hardware was confirmed
**What happened:** bet 1 was rejected partly on "you likely cannot run a multi-GPU experiment at all."
The configuration turned out to be 2× RTX 6000 Pro Blackwell plus cluster access. The objection was
**partially retracted** — the verdict survived on novelty grounds, but the reasoning had to change.
**Rule:** confirm hardware before building an argument on it. The retraction is recorded in the audit
rather than quietly edited away.

### M-17 — Hardware *inventory* reported as available *capacity*
**What happened:** the Sharanga cluster's configuration page was read off as available compute — "gpu8 has
6× Blackwell, gpu7 has 8× H200" — and used to argue that a third-scale training run would be comfortable.

**What was actually true:** Sharanga is **SLURM-scheduled**; jobs queue rather than granting interactive
node access, all GPUs on a node cannot practically be requested at once, and **the Blackwell node is
reserved for admins.** None of this appears on the configuration page.

**Root cause:** the page documents *no* access policy. That was treated as "unrestricted" when it should
have been treated as **"unknown."** Absence of a stated restriction is not absence of a restriction.

**Rule:** on any shared cluster, inventory ≠ allocation. Confirm partitions, per-user GPU caps, and
wall-time limits with an administrator before planning around a node. Plan around hardware you control.

*Related:* M-14 is the same failure in the opposite direction — asserting a hardware *limit* before
confirming it. Both come from treating unverified hardware assumptions as settled.

### M-18 — Seven machines collapsed into one "development laptop"
**What happened:** the docs recorded a single "development laptop, RTX 3060 6 GB," taken from the design
doc's §5.5. Running `nvidia-smi` on the machine in use returned **RTX 3050 Laptop, 4 GB** — and this was
initially written up as *"the design doc's spec is wrong."*

**That framing was itself wrong.** There are **seven team members with seven different machines.** The
6 GB RTX 3060 is most likely accurate — for Neel's laptop. The 4 GB RTX 3050 is accurate for Aaditya's.
Neither is "the" development laptop, and the design doc was not necessarily mistaken.

**Root cause:** a singular noun ("the development laptop") in a document written by one person, read as a
team-wide fact by another. The correction then compounded the error by treating one measurement as
authoritative for everyone.

**Consequence:** any "this runs locally" claim is ambiguous until it names a machine. §5.5's local-dev plan
is sized for 6 GB and does not transfer to a 4 GB machine.

**Rules:**
- Record hardware **per person**, never as a team-wide singular.
- Measure rather than copy — `nvidia-smi` takes two seconds.
- When correcting someone else's spec, check whether they were describing a *different machine* before
  calling it an error.

### M-15 — Background research agents died on an API spend limit, twice
Four agents were killed mid-run, losing the novelty searches for four of Aalhad's six bets on the first
attempt and the Option E search on the second.
**Rule:** for long searches, scope agents tightly and expect them to fail. Record explicitly which findings
are *unverified because a search did not complete* rather than presenting partial coverage as complete.
See `audit/OPUS_Audit.md` §7 for how this was handled.

### M-16 — A verification pass reached a conclusion that was wrong
**What happened:** an automated check claimed STaRR "falsifies" the taxonomy of remasking signals. It does
not — STaRR's temporal-variance and spatial-deviance metrics are computed from token confidence, i.e.
internal model signals. It is an *instance* of the probabilistic class, not a counterexample.
**Rule:** agents can be well-sourced and still wrong in their reasoning. The disagreement is recorded in
the audit rather than silently accepted or silently overridden.

---

## Checklist before starting new work

- [ ] Is every citation resolved to a working link, personally?
- [ ] Have I read the full text of any paper I claim has (or lacks) a gap?
- [ ] Does the code actually do what the docs say it does?
- [ ] Have I asserted lengths / bounds on any index mapping I built by hand?
- [ ] Is every file open call passing `encoding='utf-8'`?
- [ ] For a config-driven behaviour, have I read the code the config controls?
- [ ] Am I labelling unverified claims as unverified?
