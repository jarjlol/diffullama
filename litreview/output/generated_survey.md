# Diffusion Language Models: Adaptation from Autoregressive Models, Continual Pre-training, and Non-Autoregressive Text Generation

*Generated survey — QUAL-SG (Bao et al., 2025) RAG-based Survey Generation task, implemented for this project.
References [1]-[48] were retrieved and ranked by the QUAL-SG retrieval pipeline
(`scripts/retrieve.py`, `scripts/arxiv_fetch.py`, `scripts/rerank.py`, `scripts/merge_finalize.py`); full
metadata, abstracts, and per-criterion ranking scores for each are in `data/candidates_topK.json`.*

## 1. Introduction

Large language models built on the autoregressive (AR), left-to-right decoding paradigm have driven most
recent progress in natural language generation, but this paradigm imposes a strict generation order that
limits parallelism, bidirectional planning, and easy revision of already-generated content. Diffusion models,
which generate data by learning to reverse a gradual noising process, offer an alternative: they can attend to
context in both directions, revise tokens after they are first produced, and generate many positions in
parallel. Adapting this idea to text has produced two related but distinct lines of work that this survey
brings together. The first is **text diffusion modeling** itself — building continuous or discrete diffusion
processes over token sequences [1, 42, 29, 30, 31, 40]. The second is the broader family of
**non-autoregressive (NAR) text generation** methods, which relax the strict left-to-right constraint using a
variety of mechanisms beyond diffusion, from fully parallel decoding [39] to semi-autoregressive block
generation [24, 31] and insertion-based generation [38]. A third thread, **continual pre-training and model
adaptation** [32, 36, 20, 27], has emerged as a practical way to reuse the enormous investment already made in
pretrained AR language models rather than training new architectures — including diffusion language models —
from scratch. Recent large-scale diffusion language models increasingly combine all three threads: they take a
pretrained AR checkpoint and continually adapt it into a diffusion model [14], echoing exactly the strategy this
survey's anchor topic is concerned with. This survey organizes the literature along these three axes, tracing
the technical lineage from early NAR machine translation and continuous text diffusion through masked/discrete
diffusion language models to the current generation of AR-to-diffusion adaptation efforts, and closes with
a review of very recent (2025-2026) work that has appeared since the most prominent AR-to-diffusion adaptation
papers were published.

## 2. Background: From Diffusion Models to Non-Autoregressive Text Generation

Diffusion probabilistic models learn a forward process that gradually corrupts data with noise and a reverse
process that denoises it step by step, and they have been highly successful for continuous data such as
images. Applying the same idea to text is complicated by the discrete nature of language: tokens cannot be
perturbed with continuous Gaussian noise in the same way pixels can. Two broad strategies have been used to
bridge this gap. The first embeds discrete tokens into a continuous space and applies standard Gaussian
diffusion there, as in Diffusion-LM [1] and DiffuSeq [42]. The second defines the forward noising process
directly over the discrete token space — e.g., by randomly replacing tokens with a special `[MASK]` symbol or
resampling them from a categorical distribution — giving rise to *discrete* or *masked* diffusion models
[40, 31, 30]. Both strategies inherit from the wider non-autoregressive generation literature, which predates
modern text diffusion by several years and originally targeted machine translation: Gu et al. [39] introduced
fully parallel NAR translation to cut inference latency at some cost to quality, and subsequent work explored
semi-autoregressive relaxations [24], iterative refinement [23], and explicit position modeling [33] to close
the resulting quality gap. Diffusion models can be viewed as a particularly structured and theoretically
grounded instance of this broader NAR family, since the sequence of denoising steps provides a principled,
learnable iterative-refinement schedule rather than a hand-designed one.

## 3. Text Diffusion Models

### 3.1 Continuous Diffusion for Text

Diffusion-LM [1] is among the first models to adapt continuous denoising diffusion to text generation,
embedding tokens into a continuous space and using gradient-based methods on the diffusion process to enable
fine-grained, plug-and-play controllable generation (e.g., over syntactic structure) that autoregressive LMs
struggle to support without retraining. DiffuSeq [42] extends continuous diffusion to conditional
sequence-to-sequence generation, showing that a single diffusion model can support diverse Seq2Seq tasks
(paraphrasing, generation, simplification) without a separate cross-attention encoder-decoder architecture.
GENIE [29] proposes a large-scale pretraining framework for continuous diffusion LMs, introducing a continuous
paragraph denoising objective so the diffusion decoder learns to reconstruct coherent multi-sentence text from
noise, rather than being trained only on short spans. TGM-DLM [17] applies a similar continuous diffusion LM
approach outside natural language, generating SMILES strings for text-guided molecule design, illustrating
that continuous text diffusion transfers to other symbolic sequence domains once framed as sequence generation
from noise.

### 3.2 Discrete and Masked Diffusion for Text

Because text is inherently discrete, a second line of work defines the diffusion process directly over the
token vocabulary instead of a continuous embedding space. Reparameterized discrete diffusion models [12]
derive an equivalent, more flexible formulation of sampling from discrete diffusion processes, improving both
training stability and decoding quality. A cheaper and better diffusion LM with soft-masked noise [48]
identifies that naively applying continuous-style Gaussian corruption is a poor match for discrete text and
proposes a soft-masking corruption scheme designed specifically for language. Diffusion-NAT [22] connects
discrete diffusion explicitly back to the NAR generation literature, using a self-prompting discrete diffusion
process built on top of pretrained language models for non-autoregressive text-to-text generation, and
explicitly frames discrete diffusion as a way to resolve the incompatibility between earlier continuous
diffusion approaches and standard pretrained LM tooling. Simple and effective masked diffusion language models
(MDLM) [30] show that a carefully engineered, simplified masked-diffusion training recipe with a
Rao-Blackwellized objective substantially narrows the previously reported performance gap between diffusion
and autoregressive language models — a key result that later scaling efforts (§3.4, §6) build on directly.

### 3.3 Initializing Diffusion Language Models from Pretrained (Masked) LMs

A recurring idea is to avoid training a diffusion LM from scratch by initializing it from an existing
pretrained masked or autoregressive LM. DiffusionBERT [40] combines a discrete diffusion process with a BERT
initialization, arguing that BERT's own masked-denoising pretraining objective is already closely related to
the diffusion training objective, so BERT's weights provide a strong starting point for a generative discrete
diffusion model. SSD-LM [31] instead builds a semi-autoregressive diffusion LM: it iteratively generates
blocks of text with a simplex-based diffusion process, combining the flexible-length generation and modular,
plug-and-play controllability of diffusion with block-wise autoregressive-style decoding for efficiency. This
initialize-from-an-existing-LM strategy is the direct conceptual predecessor of the more recent trend (§3.4,
§4.3) of adapting *large-scale autoregressive* LLMs — rather than smaller masked LMs like BERT — into diffusion
models.

### 3.4 Scaling Masked Diffusion Language Models (2024-2026)

Building on the MDLM recipe [30], recent work has pushed masked/discrete diffusion LMs toward much larger
scale and toward matching autoregressive LLMs on general instruction-following and reasoning tasks rather than
only on language modeling perplexity. TESS 2 [14] is a particularly direct precedent for AR-to-diffusion
adaptation: it builds a general instruction-following diffusion LM by first *continuing pretraining* a strong
existing AR model with a diffusion loss and only then performing instruction tuning, and the authors report
that both this adaptation step and the choice of base AR model are crucial to final quality — closely
paralleling the anchor paper's own adaptation recipe. Dream-Coder 7B [9] applies the same
adapt-from-pretrained-AR-model philosophy specifically to code generation, and shows that a diffusion LM can
learn to *adaptively choose* its own decoding order (sketch-first for complex algorithms, left-to-right for
simple completions), an emergent any-order capability that autoregressive decoders cannot express by
construction. Beyond capability scaling, a growing body of 2025-2026 work analyzes and improves the diffusion
LM training and inference process itself: Svete and Sabharwal [4] give a theoretical characterization of what
reasoning problems masked diffusion models can and cannot efficiently solve, connecting them to chain-of-thought
and padded looped-transformer computational models; Ringel et al. [16] identify that parallel unmasking degrades
quality when the tokens chosen for simultaneous decoding are statistically dependent, and propose a lightweight
dependency predictor to guide which positions are safe to decode in parallel; and DiSPO [3] introduces a
plug-in reinforcement-learning credit-assignment layer that optimizes intermediate masked-filling decisions
rather than only the final output, addressing the coarse credit assignment that comes from training a
multi-step iterative refinement process with only a terminal reward.

## 4. Adapting and Continually Pre-training Language Models

### 4.1 Continual / Domain-Adaptive Pre-training of Autoregressive LMs

Training a large language model from scratch is extremely resource-intensive, which has motivated a large body
of work on *continuing* to pretrain an already-trained model instead. Ke et al. [32] study continual
domain-adaptive pre-training explicitly, proposing a method to continually adapt an LM across a sequence of
unlabeled domain corpora while mitigating forgetting of earlier domains. Gupta et al. [36] study a practical
ingredient of this process — how to "re-warm" the learning rate schedule when resuming pretraining on new data
— showing that naive continuation can cause large, avoidable performance regressions on previously learned
distributions. Xie et al. [27] and Fujii et al. [20] apply continual pre-training to build domain- and
language-specialized models (a financial-domain LLM and a Japanese-capable LLM built by extending Llama 2's
vocabulary and continuing pretraining on Japanese web text, respectively), demonstrating that continual
pre-training is now a standard, practical recipe for specializing an existing open AR LLM rather than training
a new one.

### 4.2 Architecture Transfer: From Transformers to Alternative Architectures

A related strand of adaptation work targets not a new *domain* but a new *architecture*. Mamba [28] proposes a
selective state-space model as a subquadratic-time alternative to the Transformer's attention mechanism, and
subsequent work (outside this survey's core retrieved set, but widely cited alongside it) has shown that such
alternative architectures can themselves be obtained by transferring the weights of an existing pretrained
Transformer rather than training from random initialization — the same "reuse an existing pretrained model to
bootstrap a differently-structured model" logic that underlies AR-to-diffusion adaptation. This makes
architecture-transfer methods a close methodological cousin of AR-to-diffusion adaptation: both treat a
pretrained AR Transformer as a reusable source of learned linguistic knowledge that can be redirected into a
model with a fundamentally different computational structure (a different attention/state mechanism in one
case, a bidirectional non-causal diffusion denoiser in the other).

### 4.3 Adapting Autoregressive Models into Diffusion Models

Combining §3.4 and §4.1-4.2, the most direct precedents for adapting large pretrained AR LLMs into diffusion
LMs are TESS 2 [14] and Dream-Coder 7B [9], both of which explicitly start from a pretrained autoregressive
checkpoint and continue training it with a diffusion objective before further tuning. Planner and Executor [21]
takes an alternative, non-weight-sharing approach to combining the two paradigms: rather than converting an AR
model into a diffusion model, it studies hybrid architectures in which a discrete diffusion model and an
autoregressive model collaborate at inference time (e.g., one plans while the other executes), reporting
complementary benefits from the two decoding styles. Together, these results indicate that the field has
converged on continual pretraining with a diffusion objective as the standard recipe for building large
diffusion LMs efficiently, while collaborative AR+diffusion inference remains a less-explored complementary
direction.

## 5. Non-Autoregressive Text Generation

### 5.1 Foundational Non-Autoregressive Machine Translation

Gu et al. [39] introduced non-autoregressive neural machine translation, removing the sequential dependency
between output tokens to obtain an order-of-magnitude reduction in decoding latency, at a cost in translation
quality that a large subsequent literature has worked to close. Glancing Transformer [23] narrows this gap with
a curriculum-style "glancing" sampling strategy during training, and Gu and Kong [43] systematically catalog
and combine several such tricks to produce fully non-autoregressive models competitive with iterative or
semi-autoregressive alternatives. PNAT [33] argues that explicitly modeling the *positions* of generated words
as a latent variable — rather than assuming a fixed, monotonic alignment — is essential to closing the quality
gap, and Ran et al. [47] similarly inject explicit reordering information into the NAR decoding process to
mitigate the "multimodality problem" in which multiple valid translations correspond to the same source
sentence.

### 5.2 Semi-Autoregressive and Position-Aware Models

Between fully parallel and fully sequential generation lies a spectrum of semi-autoregressive methods that
keep some local sequential structure while parallelizing across blocks. Wang et al. [24] propose the
semi-autoregressive Transformer, which preserves autoregressive dependencies globally across blocks but
generates multiple tokens per block in parallel; SmBoP [25] applies an analogous bottom-up, semi-autoregressive
strategy to semantic parsing, decoding top-k sub-trees at each height of the target abstract syntax tree rather
than a single top-down path; and Deng and Rush [15] propose cascaded decoding with bounded-context Markov
transformers, achieving sub-linear-time parallel generation while retaining most of the modeling benefits of
autoregressive conditioning. SSD-LM [31] (§3.3) can be read as a diffusion-based instantiation of this same
semi-autoregressive principle. Yang et al. [2] instead constrain the parallel decoding process directly with
part-of-speech information, showing that structural linguistic constraints can guide which positions are safe
to decode in parallel — a similar idea to the recent dependency-aware parallel decoding for diffusion LMs
discussed in §3.4 [16].

### 5.3 Non-Autoregressive Generation Beyond Translation

Non-autoregressive decoding has been applied well beyond machine translation. Liu et al. [34] apply NAR
decoding to unsupervised sentence summarization by first performing an edit-based search for a heuristic
pseudo-ground-truth and then training an encoder-only NAR model on the resulting search output. Su et al. [44]
show that a pretrained BERT encoder can serve as an effective backbone for NAR generation more broadly,
addressing the classic NAR weaknesses of fixed output length and conditional-independence-induced repetition.
The same parallel-decoding idea has also been adapted to non-text-generation-adjacent modalities and tasks
sharing the same sequential-output structure: video captioning [37], scene-text recognition [46],
non-autoregressive automatic speech recognition (Paraformer [45] and the comparative study of Higuchi et al.
[41]), and even non-autoregressive diffusion-based time-series forecasting [7], underscoring that "generate a
structured sequence in parallel rather than strictly left-to-right" is a general modeling pattern rather than a
translation-specific trick. Huang et al. [35] provide a theoretical account of *why* NAR training is
challenging in the first place, showing that the multimodality problem underlying many of these applied results
follows from simplifying assumptions common to standard NAR training objectives.

### 5.4 Constrained and Insertion-Based Generation

A further branch of NAR-adjacent work targets *lexically constrained* generation, where the output must
contain a given set of keywords. POINTER [38] generates text by progressively inserting new tokens between
existing ones, expanding a template that already contains the required constraint words, rather than decoding
purely left-to-right or fully in parallel. AutoTemplate [26] offers a simpler recipe for the same
lexically-constrained setting by casting it as a template-filling task, trading some of POINTER's architectural
complexity for implementation simplicity. Mansimov et al. [8] generalize the connection between undirected
sequence models (such as BERT) and text generation more broadly, providing a unifying framework that
foreshadows why masked/discrete diffusion LMs — which are themselves undirected, denoising models — are a
natural fit for flexible-order and infilling-style generation (see also DDOT [13] in §3.4/§6, which targets
flexible-length infilling specifically for discrete diffusion models).

## 6. Recent Developments Since DiffuLLaMA (2025-2026): Scaling, Reasoning, and Applications

Because the anchor paper for this survey (Gong et al., *Scaling Diffusion Language Models via Adaptation from
Autoregressive Models*, ICLR 2025) was written in 2024, our recency-oriented retrieval surfaces a substantial
body of directly relevant work published after it, which the anchor paper's own related-work section could not
have cited. This growth reflects rapid consolidation around the "adapt a pretrained AR model into a diffusion
LM" recipe: TESS 2 [14] and Dream-Coder 7B [9] (§3.4) both scale this recipe to general instruction-following
and code generation respectively. Several 2025-2026 papers dig into *why* and *how well* the resulting models
work: Svete and Sabharwal [4] formally analyze the reasoning capability of masked diffusion LMs; Berrayana et
al. [21] explore hybrid diffusion-plus-autoregressive collaboration as an alternative to full conversion; and
Ringel et al. [16] and DiSPO [3] target specific weaknesses of the parallel-decoding process (dependency
violations and coarse credit assignment, respectively) that only become apparent once diffusion LMs are trained
at a scale where their generation quality is competitive enough to analyze closely. Zhang et al. [13] extend
the infilling capability that motivated much of the original interest in diffusion LMs (§2, §5.4) to flexible,
variable-length spans, and application-oriented papers such as text-attributed graph learning [19] and
retrieval-augmented generation with diffusion LMs [5] show the paradigm being adopted as a component within
larger systems rather than studied only in isolation. Taken together, this recent cluster indicates the field
has moved from asking "can an AR model be adapted into a competitive diffusion LM at all" (the anchor paper's
question) to "how do we scale, specialize, and analyze the diffusion LMs this adaptation recipe now reliably
produces."

## 7. Open Challenges and Future Directions

Several open problems recur across the surveyed literature. First, the quality gap between non-autoregressive
and autoregressive generation, while substantially narrowed by masked diffusion approaches [30], is not fully
closed, and the theoretical work characterizing NAR/diffusion learning difficulty [35, 4] suggests some of the
gap may be structural rather than purely a matter of scale or training recipe. Second, parallel decoding
still faces a fundamental tension between speed and the conditional-independence assumption it relies on;
dependency-aware decoding [16, 2] is a promising but still early direction for resolving this without
sacrificing the parallelism that motivates NAR/diffusion approaches in the first place. Third, while AR-to-
diffusion adaptation [14, 9] has proven more efficient than training diffusion LMs from scratch, it inherits
open questions from the continual pre-training literature more broadly — how to avoid forgetting the base
model's capabilities [36, 32] and how much adaptation data and compute are truly required — that have not yet
been studied as thoroughly in the diffusion-adaptation setting as in the standard AR continual-pretraining
setting. Finally, most of the retrieved diffusion-LM literature remains centered on English text and general
- domain benchmarks; extending the adaptation recipe to multilingual and domain-specialized settings (in the
way continual pre-training has already been extended, e.g. [20, 27]) remains comparatively unexplored for
diffusion LMs specifically.

## 8. Conclusion

This survey traced three intertwined literatures — text diffusion models, continual pre-training/architecture
adaptation, and non-autoregressive text generation — that converge on a now-dominant recipe for building
large diffusion language models: start from a pretrained autoregressive LLM and continually adapt it with a
diffusion training objective, rather than training a diffusion LM from scratch or initializing only from a
small masked LM as earlier work did [40, 31]. Early non-autoregressive [39] and continuous/discrete text
diffusion [1, 42, 12, 30] work established the modeling foundations; continual pre-training research [32, 36]
established that reusing a pretrained model is both cheaper and often better than retraining; and a cluster of
2025-2026 papers [14, 9, 4, 3, 16] shows the adaptation recipe now being scaled, analyzed, and specialized
across code generation, reasoning, and retrieval-augmented settings. The remaining open challenges — closing
the residual AR/NAR quality gap, decoding safely under token dependence, and avoiding catastrophic forgetting
during adaptation — define the near-term research agenda for this area.

## References

[1] Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori Hashimoto (2022). Diffusion-LM Improves Controllable Text Generation. *arXiv (Cornell University)*. [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)

[2] Kexin Yang, Wenqiang Lei, Dayiheng Liu, Weizhen Qi, Jiancheng Lv (2021). POS-Constrained Parallel Decoding for Non-autoregressive Generation. *arXiv preprint*. [https://doi.org/10.18653/v1/2021.acl-long.467](https://doi.org/10.18653/v1/2021.acl-long.467)

[3] Daisuke Oba, Hiroki Furuta, Naoaki Okazaki (2026). Diffusion-State Policy Optimization for Masked Diffusion Language Models. *arXiv (Cornell University)*. [arXiv:2602.06462](https://arxiv.org/abs/2602.06462)

[4] Anej Svete, Ashish Sabharwal (2025). On the Reasoning Abilities of Masked Diffusion Language Models. *arXiv (Cornell University)*. [arXiv:2510.13117](https://arxiv.org/abs/2510.13117)

[5] Paul Jünger, Justin Lovelace, Linxi Zhao, Dongyoung Go, Kilian Q. Weinberger (2026). Self-Augmenting Retrieval for Diffusion Language Models. *arXiv (Cornell University)*. [arXiv:2606.06474](https://arxiv.org/abs/2606.06474)

[6] Hanchen Xia, Baoyou Chen, Yutang Ge, Guojiang Zhao, Siyu Zhu (2026). T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforcement Learning. *arXiv (Cornell University)*. [arXiv:2601.11214](https://arxiv.org/abs/2601.11214)

[7] Lifeng Shen, James Kwok (2023). Non-autoregressive Conditional Diffusion Models for Time Series Prediction. *arXiv (Cornell University)*. [arXiv:2306.05043](https://arxiv.org/abs/2306.05043)

[8] Elman Mansimov, Alex Wang, Sean Welleck, Kyunghyun Cho (2019). A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models. *arXiv (Cornell University)*. [arXiv:1905.12790](https://arxiv.org/abs/1905.12790)

[9] Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, et al. (2025). Dream-Coder 7B: An Open Diffusion Language Model for Code. *arXiv (Cornell University)*. [arXiv:2509.01142](https://arxiv.org/abs/2509.01142)

[10] Despoina Kosmopoulou, Efthymios Georgiou, Vaggelis Dorovatas, Georgios Paraskevopoulos, Alexandros Potamianos (2025). Masked Diffusion Language Models with Frequency-Informed Training. *arXiv (Cornell University)*. [arXiv:2509.05056](https://arxiv.org/abs/2509.05056)

[11] Yichen Yang, Hong Li, Haodong Zhu, Linin Yang, Guojun Lei, Sheng Xu, et al. (2025). PartDiffuser: Part-wise 3D Mesh Generation via Discrete Diffusion. *arXiv (Cornell University)*. [arXiv:2511.18801](https://arxiv.org/abs/2511.18801)

[12] Lin Zheng, Jianbo Yuan, Lei Yu, Lingpeng Kong (2023). A Reparameterized Discrete Diffusion Model for Text Generation. *arXiv (Cornell University)*. [arXiv:2302.05737](https://arxiv.org/abs/2302.05737)

[13] Andrew Zhang, Anushka Sivakumar, Chiawei Tang, Chris Thomas (2025). Flexible-length Text Infilling for Discrete Diffusion Models. *arXiv (Cornell University)*. [arXiv:2506.13579](https://arxiv.org/abs/2506.13579)

[14] Jaesung Tae, Hamish Ivison, Sachin Kumar, Arman Cohan (2025). TESS 2: A Large-Scale Generalist Diffusion Language Model. *arXiv (Cornell University)*. [arXiv:2502.13917](https://arxiv.org/abs/2502.13917)

[15] Yuntian Deng, Alexander M. Rush (2020). Cascaded Text Generation with Markov Transformers. *arXiv (Cornell University)*. [arXiv:2006.01112](https://arxiv.org/abs/2006.01112)

[16] Liran Ringel, Ameen Ali, Yaniv Romano (2026). Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models. *arXiv (Cornell University)*. [arXiv:2604.02560](https://arxiv.org/abs/2604.02560)

[17] Haisong Gong, Qiang Liu, Shu Wu, Liang Wang (2024). Text-Guided Molecule Generation with Diffusion Language Model. *Proceedings of the AAAI Conference on Artificial Intelligence*. [https://doi.org/10.1609/aaai.v38i1.27761](https://doi.org/10.1609/aaai.v38i1.27761)

[18] Hui‐Wen Chang, Han Zhang, Jarred Barber, AJ Maschinot, José Lezama, Lu Jiang, et al. (2023). Muse: Text-To-Image Generation via Masked Generative Transformers. *arXiv (Cornell University)*. [arXiv:2301.00704](https://arxiv.org/abs/2301.00704)

[19] Lingjie Chen, Yuanchen Bei, Haobo Xu, Yanjun Zhao, Yuzhong Chen, Hanghang Tong (2026). TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning. *arXiv (Cornell University)*. [arXiv:2606.31166](https://arxiv.org/abs/2606.31166)

[20] Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, et al. (2024). Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities. *arXiv (Cornell University)*. [arXiv:2404.17790](https://arxiv.org/abs/2404.17790)

[21] Lina Berrayana, Ahmed Heakl, Muhammad Abdullah Sohail, Thomas Hofmann, Salman Khan, Wei Chen (2025). Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning. *arXiv (Cornell University)*. [arXiv:2510.15244](https://arxiv.org/abs/2510.15244)

[22] Kun Zhou, Yifan Li, Wayne Xin Zhao, Ji-Rong Wen (2023). Diffusion-NAT: Self-Prompting Discrete Diffusion for Non-Autoregressive Text Generation. *arXiv (Cornell University)*. [arXiv:2305.04044](https://arxiv.org/abs/2305.04044)

[23] Lihua Qian, Hao Zhou, Yu Bao, Mingxuan Wang, Lin Qiu, Weinan Zhang, et al. (2021). Glancing Transformer for Non-Autoregressive Neural Machine Translation. *arXiv preprint*. [https://doi.org/10.18653/v1/2021.acl-long.155](https://doi.org/10.18653/v1/2021.acl-long.155)

[24] Chunqi Wang, Ji Zhang, Haiqing Chen (2018). Semi-Autoregressive Neural Machine Translation. *arXiv (Cornell University)*. [arXiv:1808.08583](https://arxiv.org/abs/1808.08583)

[25] Ohad Rubin, Jonathan Berant (2021). SmBoP: Semi-autoregressive Bottom-up Semantic Parsing. *arXiv preprint*. [https://doi.org/10.18653/v1/2021.naacl-main.29](https://doi.org/10.18653/v1/2021.naacl-main.29)

[26] Hayate Iso (2022). AutoTemplate: A Simple Recipe for Lexically Constrained Text Generation. *arXiv (Cornell University)*. [arXiv:2211.08387](https://arxiv.org/abs/2211.08387)

[27] Yong Xie, Karan Aggarwal, Aitzaz Ahmad (2023). Efficient Continual Pre-training for Building Domain Specific Large Language Models. *arXiv (Cornell University)*. [arXiv:2311.08545](https://arxiv.org/abs/2311.08545)

[28] Albert Gu, Tri Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv (Cornell University)*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

[29] Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, et al. (2022). Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise. *arXiv (Cornell University)*. [arXiv:2212.11685](https://arxiv.org/abs/2212.11685)

[30] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, et al. (2024). Simple and Effective Masked Diffusion Language Models. *arXiv (Cornell University)*. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)

[31] Xiaochuang Han, Sachin Kumar, Yulia Tsvetkov (2023). SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control. *arXiv preprint*. [https://doi.org/10.18653/v1/2023.acl-long.647](https://doi.org/10.18653/v1/2023.acl-long.647)

[32] Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, Bing Liu (2023). Continual Pre-training of Language Models. *arXiv (Cornell University)*. [arXiv:2302.03241](https://arxiv.org/abs/2302.03241)

[33] Yu Bao, Hao Zhou, Jiangtao Feng, Mingxuan Wang, Shujian Huang, Jiajun Chen, et al. (2019). Non-autoregressive Transformer by Position Learning. *arXiv (Cornell University)*. [arXiv:1911.10677](https://arxiv.org/abs/1911.10677)

[34] Puyuan Liu, Chenyang Huang, Lili Mou (2022). Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. [https://doi.org/10.18653/v1/2022.acl-long.545](https://doi.org/10.18653/v1/2022.acl-long.545)

[35] Fei Huang, Tianhua Tao, Hao Zhou, Lei Li, Minlie Huang (2022). On the Learning of Non-Autoregressive Transformers. *arXiv (Cornell University)*. [arXiv:2206.05975](https://arxiv.org/abs/2206.05975)

[36] Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L. Richter, Quentin Anthony, Eugene Belilovsky, et al. (2023). Continual Pre-Training of Large Language Models: How to (re)warm your model?. *arXiv (Cornell University)*. [arXiv:2308.04014](https://arxiv.org/abs/2308.04014)

[37] Bang Yang, Yuexian Zou, Fenglin Liu, Can Zhang (2021). Non-Autoregressive Coarse-to-Fine Video Captioning. *Proceedings of the AAAI Conference on Artificial Intelligence*. [https://doi.org/10.1609/aaai.v35i4.16421](https://doi.org/10.1609/aaai.v35i4.16421)

[38] Yizhe Zhang, Guoyin Wang, Chunyuan Li, Zhe Gan, Chris Brockett, Bill Dolan (2020). POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training. *arXiv (Cornell University)*. [arXiv:2005.00558](https://arxiv.org/abs/2005.00558)

[39] Jiatao Gu, James Bradbury, Caiming Xiong, Victor O. K. Li, Richard Socher (2017). Non-Autoregressive Neural Machine Translation. *arXiv (Cornell University)*. [arXiv:1711.02281](https://arxiv.org/abs/1711.02281)

[40] Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, Xipeng Qiu (2023). DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models. *arXiv preprint*. [https://doi.org/10.18653/v1/2023.acl-long.248](https://doi.org/10.18653/v1/2023.acl-long.248)

[41] Yosuke Higuchi, Nanxin Chen, Yuya Fujita, Hirofumi Inaguma, Tatsuya Komatsu, Jaesong Lee, et al. (2021). A Comparative Study on Non-Autoregressive Modelings for Speech-to-Text Generation. *arXiv (Cornell University)*. [arXiv:2110.05249](https://arxiv.org/abs/2110.05249)

[42] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, LingPeng Kong (2022). DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models. *arXiv (Cornell University)*. [arXiv:2210.08933](https://arxiv.org/abs/2210.08933)

[43] Jiatao Gu, Xiang Kong (2021). Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade. *arXiv preprint*. [https://doi.org/10.18653/v1/2021.findings-acl.11](https://doi.org/10.18653/v1/2021.findings-acl.11)

[44] Yixuan Su, Deng Cai, Yan Wang, David Vandyke, Simon Baker, Piji Li, et al. (2021). Non-Autoregressive Text Generation with Pre-trained Language Models. *arXiv (Cornell University)*. [arXiv:2102.08220](https://arxiv.org/abs/2102.08220)

[45] Zhifu Gao, Shiliang Zhang, Ian McLoughlin, Zhijie Yan (2022). Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition. *Interspeech 2022*. [https://doi.org/10.21437/interspeech.2022-9996](https://doi.org/10.21437/interspeech.2022-9996)

[46] Zhi Qiao, Yu Zhou, Wei Jin, Wei Wang, Yuan Zhang, Ning Jiang, et al. (2021). PIMNet: A Parallel, Iterative and Mimicking Network for Scene Text Recognition. *arXiv preprint*. [https://doi.org/10.1145/3474085.3475238](https://doi.org/10.1145/3474085.3475238)

[47] Qiu Ran, Yankai Lin, Peng Li, Jie Zhou (2021). Guiding Non-Autoregressive Neural Machine Translation Decoding with Reordering Information. *Proceedings of the AAAI Conference on Artificial Intelligence*. [https://doi.org/10.1609/aaai.v35i15.17618](https://doi.org/10.1609/aaai.v35i15.17618)

[48] Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, Diyi Yang (2023). A Cheaper and Better Diffusion Language Model with Soft-Masked Noise. *arXiv preprint*. [https://doi.org/10.18653/v1/2023.emnlp-main.289](https://doi.org/10.18653/v1/2023.emnlp-main.289)
