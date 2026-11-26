<!-- SUPERSEDED (pre-#106, negative-result framing): the submitted paper is the LaTeX build at ../paper.pdf. Numbers and claims in this file are not those of the paper; see README.md in this directory. -->

# Methodology

<!-- Owner: aaditya (issue #11). Notation: symbol table in the status comment of #10 / final_submission/paper/sections/problem_formulation.md (PR #19). Sources: ideation/output/research_problems_and_ideas.md (branch assignment/research-ideation), codification/IDEA_SELECTION.md, codification/ITERATION_1.md, codification/rerank_task/README.md (branch assignment/codification). Citation keys in [brackets] are in final_submission/paper/refs_methodology.bib (7 entries; gong2025diffullama is in refs_related_work.bib, #9). Results are out of scope (Experiments, #12). -->

We study whether a small autoregressive (AR) model can pick the best answer from the candidates of a diffusion language model (DLM), without any training, and whether the DLM's similarity to its AR base predicts how well this works. The idea comes from the ideation stage, where an LLM research-idea generator [baek2024researchagent] proposed and ranked ideas for the anchor paper; we codified the highest-ranked P1 idea (1.1, tied at 4.9/5).

## 1. Data and models

**Data.** We use GSM8K [cobbe2021gsm8k], math word problems with a numeric gold answer, so correctness is checkable by numeric comparison of the final number. Questions are split into disjoint development and held-out test sets; selectors may be developed on the former, and the latter decides which selector is accepted.

**Models.** The set $\mathcal{M}$ holds five released DLMs: DiffuGPT-S, DiffuGPT-M and DiffuLLaMA [gong2025diffullama], which were adapted from AR models GPT-2 and LLaMA-2, Dream 7B [ye2025dream7b], adapted from Qwen2.5-7B, and LLaDA [nie2025llada]. Four have an AR base $b(m)$. LLaDA was trained from scratch ($b(m)=\varnothing$) and serves as a control: it shows what AR scoring does for a model with no AR representations to preserve.

## 2. Approach

**Candidate generation.** For each question $q_j$, each model $m$ is prompted with $k$ worked examples ($x_j$) and samples $N$ candidate solutions $y_{j,1},\dots,y_{j,N}$. Candidates are generated once and cached, so every selector is compared on identical samples.

**Answer pickers.** A selector $s$ chooses one candidate per question using only that question's own candidates. We compare (i) the *first* sample, which is one run of the model; (ii) *majority vote* over the extracted final numbers, i.e. self-consistency [wang2023selfconsistency]; (iii) *AR reranking* $s_\psi$, which picks the candidate with the lowest negative log-likelihood under a frozen AR scorer $\psi$, either summed or per token, over the solution up to its final answer line (trailing generated text distorted the first round of scores and is excluded); and (iv) the *oracle*, the best any selector could do given the candidates. Scorers are small AR models, plus the 7B AR bases of DiffuLLaMA (LLaMA-2-7B) and Dream (Qwen2.5-7B) as the strongest matched scorers. How much of the first-to-oracle gap a selector closes is its headroom $H_m(s)$.

**Similarity measure.** Preservation of AR representations is measured per model as $\rho_m$, the layer-averaged linear centred kernel alignment (CKA) [kornblith2019cka] between the token-mean-pooled block outputs of $m$ and of $b(m)$ on the same held-out text. The hypothesis is that a larger $\rho_m$ goes with a larger AR-recoverable headroom, checked across the models that have an AR base.

**Codifying the selection rule with an agent.** The specification of idea 1.1 had gaps found on reading it (no majority-vote baseline, a length bias in summed likelihood, too few models for a significance test). We fixed the first two in the evaluation harness, reported the third as a descriptive limitation, and then handed the selection rule to Arbor [jin2026arbor], an LLM research agent that proposes hypotheses in a tree, implements each as a change to a single selector function, and keeps a change only if the macro-averaged accuracy $\overline{\mathrm{Acc}}(s)$ improves on held-out data. We reviewed every hypothesis before it ran and gave feedback when it was unfair or wrong (for example, rejecting selectors that read other models' answers, which would change the task). The target is a selector with $\overline{\mathrm{Acc}}(s) > \overline{\mathrm{Acc}}(s_{\text{maj}})$.
