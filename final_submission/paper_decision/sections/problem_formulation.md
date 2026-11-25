# Problem Formulation

**Setting.** Let $\mathcal{M}$ be a set of masked diffusion language models (DLMs) and $\mathcal{D} = \{(q_j, a_j^\ast)\}_{j=1}^{n}$ a set of problems with gold answers. For each problem, a model $m \in \mathcal{M}$ draws $N$ candidates $y_{j,1}, \dots, y_{j,N}$ from a fixed prompt $x_j$. Candidates are generated once and cached, so every selector sees the same samples. A candidate is correct, $c_{j,i} = 1$, when its extracted final number equals $a_j^\ast$ (GSM8K) or when it passes the hidden tests (HumanEval); otherwise $c_{j,i} = 0$.

**Selectors.** A selector $s$ maps an item $(q_j, y_{j,1:N})$ to an index in $\{1, \dots, N\}$. It may use the problem and that item's candidates, never $a_j^\ast$, the hidden tests or any execution result on them. Its accuracy on $m$ is
$$\mathrm{Acc}_m(s) = \frac{1}{n} \sum_{j=1}^{n} c_{j,\, s(q_j, y_{j,1:N})}.$$
No selector can exceed the *oracle*, $\mathrm{Acc}_m^{\ast} = \frac{1}{n}\sum_j \max_i c_{j,i}$ (pass@$N$ for code). The share of the reachable gap that $s$ closes is its *headroom*
$$H_m(s) = \frac{\mathrm{Acc}_m(s) - \mathrm{Acc}_m(s_{\text{first}})}{\mathrm{Acc}_m^{\ast} - \mathrm{Acc}_m(s_{\text{first}})},$$
where $s_{\text{first}} = 1$ is a single run of the model.

**Decision-model selectors.** A decision model $d$ reads a state and a typed question and returns probabilities in one forward pass, without generated reasoning. The *listwise* selector shows all $N$ candidates, labelled A to J, and picks
$$s_{\text{list}} = \arg\max_i \; P_d(\text{choice} = i \mid q_j, y_{j,1:N}).$$
The *pointwise* selector asks one yes/no question per candidate and picks $\arg\max_i P_d(\text{yes} \mid q_j, y_{j,i})$. Ties go to the earliest index.

**Objective and hypotheses.** On GSM8K the primary hypothesis (H1) is that the macro accuracy $\overline{\mathrm{Acc}}(s_{\text{list}}) = \frac{1}{|\mathcal{M}|}\sum_m \mathrm{Acc}_m(s_{\text{list}})$ exceeds that of majority vote. The secondary hypothesis (H2) is that it exceeds the best AR-likelihood selector. On HumanEval the primary hypothesis (H4) is that $s_{\text{list}}$ exceeds a filter that picks the first candidate passing the docstring examples. Majority vote is undefined for code.

**Scope.** Decoding settings are fixed per model. The GSM8K test split (500 questions) is held out; a separate 500-question dev split is used only to check the pipeline and the order-shuffle control. HumanEval is split into 20 dev and 144 test problems (seed 0). We test one decision-model snapshot with one prompt wording per task.
