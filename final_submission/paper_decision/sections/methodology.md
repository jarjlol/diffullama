# Methodology

We compare training-free selectors on fixed candidates from two DLMs. The comparison was pre-registered (`codification/DECISION_RERANK_PREREG.md`, commit `5ecd013`) before the results commit (`03d3dbe`).

## Data and models

**Data.** GSM8K [cobbe2021gsm8k] has math word problems with a numeric gold answer, so a candidate is checked by comparing its final number. HumanEval [chen2021humaneval] has 164 Python functions specified by a signature and docstring, each graded by hidden unit tests.

**Models.** Dream 7B [ye2025dream7b] is adapted from Qwen2.5-7B, and LLaDA 8B [nie2025llada] is trained from scratch. For GSM8K, each model has $N = 10$ cached candidates per question at 64 denoising steps and temperature 0.7, with a 4-shot prompt. For HumanEval, Dream generates $N = 10$ candidates zero-shot at 64 steps, temperature 0.7 and at most 512 new tokens. LLaDA was registered for HumanEval but dropped (Section V).

## Selectors

**References.** (i) The *first* sample. (ii) The *expected accuracy of a uniform pick*. (iii) *Majority vote* over extracted final numbers [wang2023selfconsistency], ties to the earliest candidate; GSM8K only. (iv) *AR likelihood*: the candidate with the lowest summed or per-token negative log-likelihood under Qwen2.5-7B. For GSM8K the text is cut after the final-answer line; for code it is the generated function, capped at 512 tokens. (v) The *example filter* (HumanEval only): the first candidate that passes the examples in its own docstring, else the first candidate. (vi) The *oracle*.

**Decision model.** `pplx-decider-flash-v1.3` was served by OpenRouter's decisions endpoint, which reported the snapshot `pplx-decider-flash-v1.3-20261112`. The run record describes it as a 4B, Apache-2.0 model; we did not check its model card or training data. Each call sends a state (the problem and candidates) and one typed question, and returns probabilities without chain of thought.

- **GSM8K listwise:** "Which candidate solution reaches the correct final answer to the math problem?", answered as a choice over A–J.

- **GSM8K pointwise:** "Does this candidate's final answer agree with the correct answer to the math problem?", answered as yes/no.

- **HumanEval listwise:** asks which candidate "correctly implements the function described by the docstring and will pass the hidden tests".

- **HumanEval pointwise:** asks the same of one candidate.

The prompts contain no tests and no execution results. The candidate text is the same cut text that the AR scorer sees. The longest GSM8K listwise prompt has 2,973 tokens, and no item was truncated.

**Prompted AR model (exploratory).** To separate the effect of the prompt from the effect of the model, Qwen2.5-7B receives the same listwise context. Its choice is read from the logits of the letters A–J. This arm was not pre-registered.

## Execution and statistics

Code runs in an isolated Python subprocess with a 5 s CPU limit, a 2 GB address space, a temporary working directory and a scrubbed environment. This is not a security boundary. Every comparison is paired on the same items. We report an exact two-sided McNemar test, with Holm correction over H1 and H2, and a paired bootstrap 95% CI (10,000 resamples, seed 0). GSM8K pairs are stacked over the two models (1,000 pairs).

## Controls

- **Candidate-order shuffle:** the candidates are permuted once (seed 1) on 200 GSM8K dev questions and on the first 50 HumanEval test problems.

- **Closed-book probe:** on 40 GSM8K test questions the decision model sees the question and two numbers, the gold answer and a wrong answer from another question, in random order, and picks one.

- **HumanEval memorisation:** a closed-book probe on 20 dev problems was registered; it is not reported (Section V).
