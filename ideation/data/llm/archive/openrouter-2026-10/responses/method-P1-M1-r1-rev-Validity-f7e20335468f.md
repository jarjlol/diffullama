**Review:**

The proposed method is well-structured and directly operationalizes the research problem into concrete, executable steps. It maintains strict adherence to the inference-only constraint, uses established representation-similarity metrics (CKA, linear probing), and introduces a thoughtful dual-scorer design to break scorer-model family confounding. The isolation of reranking gain from denoising-step gain via the gain-ratio ηᵢ and compute-normalized ratio εᵢ cleanly addresses the efficiency question. Comprehensive ablation plans (scorer size, alternative preservation metrics, probing-corpus domain, random-scorer control) demonstrate awareness of potential validity threats. Pre-registration, bootstrap CIs, and a Bayesian sensitivity check reflect rigor in statistical inference.

However, several limitations prevent a higher validity rating:

1. **PCA-derived UQS weights on a small validation set (~60 prompts):** With only four benchmarks and ~60 validation points, the first principal component may capture noise rather than shared signal, making the primary outcome metric unstable and potentially overfitted to the specific sample. This undermines the validity of the central dependent variable. A simpler equal-weight or literature-informed weighting scheme would be more defensible.

2. **Limited model count (5 DLMs) for key interaction tests:** The preservation × architecture interaction (β₈) and preservation × step-budget interaction (β₄) are tested with only 2–3 models per architecture family. With the preservation score varying only at the model level (5 distinct values), these tests are underpowered, and architecture effects are confounded with preservation effects. The fixed-effects approach appropriately avoids random-effects confounding but does not resolve the fundamental identifiability problem.

3. **Min-max normalization dependency:** Normalizing across all experimental conditions creates a coupling between the metric and the results; outlier conditions could inflate apparent gains. A fixed reference normalization pool would be more robust.

4. **Hidden-state extraction point:** The choice of "post-attention, pre-FF" is somewhat arbitrary and may yield different preservation rankings than post-FF or averaged-layer representations. While ablated, this should be motivated more explicitly.

5. **Additivity assumption:** The method assumes reranking gain and step-budget gain are independent levers, but their interaction is not explicitly modeled or tested.

**Feedback:**

- Replace PCA-derived UQS weights with equal-weighting or a fixed set of benchmark weights justified by domain literature; report PCA weights only as an exploratory sensitivity analysis.
- Acknowledge the limited power for interaction tests as a constraint rather than overstating conclusions about architecture × preservation; consider Bayesian hierarchical models with strong regularization if interaction estimates are reported.
- Use a fixed reference set for min-max normalization to avoid result-dependent metric construction.
- Justify the hidden-state extraction point with reference to prior probing studies (e.g., Mahabadi et al., 2021) and report extraction-point sensitivity.
- Test the additivity assumption by including a reranking × step-budget interaction term in the model.

**Rating (1-5): 3**

The method adequately addresses the research problem with a sound overall design and strong alignment with existing literature, but meaningful limitations in the primary outcome metric construction (PCA on small validation set) and statistical power for key interactions prevent a higher validity rating.