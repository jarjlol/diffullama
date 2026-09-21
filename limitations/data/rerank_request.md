You are the LLM re-ranker of the RAG stage described in arXiv:2601.11578 section 3.2.

TASK. For each candidate chunk below, score 0-10 how useful that chunk is for
identifying a LIMITATION of the input paper. A chunk is useful when it reports a
result, method, or critique that exposes a weakness, unexamined assumption,
untested regime, or superseded choice in the input paper.

Score 0-3  : unrelated topic, or relevant topic with nothing limitation-bearing.
Score 4-7  : same area, but only generic overlap; no specific weakness exposed.
Score 8-10 : directly exposes or implies a concrete weakness of the input paper.

Only chunks scoring >= 8 are retained.

INPUT PAPER
-----------
Published as a conference paper at ICLR 2025




S CALING D IFFUSION L ANGUAGE M ODELS
VIA A DAPTATION FROM AUTOREGRESSIVE M ODELS
                        ∗
 Shansan Gong∗1 , Shivam Agarwal∗2 , Yizhe Zhang3 , Jiacheng Ye1 , Lin Zheng1
 Mukai Li1 , Chenxin An1 , Peilin Zhao4 , Wei Bi4 , Hao Peng2 , Jiawei Han2 , Lingpeng Kong1
 1
   The University of Hong Kong 2 University of Illinois at Urbana-Champaign
 3
   Apple 4 Tencent AI Lab
 sansa933@connect.hku.hk,shivama2@illinois.edu



                                              A BSTRACT

             Diffusion Language Models (DLMs) have emerged as a promising new paradigm
             for text generative modeling, potentially addressing limitations of autoregressive
             (AR) models. However, current DLMs have been studied at a smaller scale com-
             pared to their AR counterparts and lack fair comparison on language modeling
             benchmarks. Additionally, training diffusion models from scratch at scale remains
             challenging. Given the prevalence of open-source AR language models, we pro-
             pose adapting these models to build text diffusion models. We demonstrate con-
             nections between AR and diffusion modeling objectives and introduce a simple
             continual pre-training approach for training diffusion models. Through system-
             atic evaluation on language modeling, reasoning, and commonsense benchmarks,
             we show that we can convert AR models ranging from 127M to 7B parameters
             (GPT2 and LLaMA) into diffusion models DiffuGPT and DiffuLLaMA, using
             less than 200B tokens for training. Our experimental results reveal that these
             models outperform earlier DLMs and are competitive with their AR counterparts.
             We release a suite of DLMs (127M-355M-7B) capable of generating fluent text,
             performing in-context learning, filling in the middle without prompt re-ordering,
             and following instructions. https://github.com/HKUNLP/DiffuLLaMA


1       I NTRODUCTION

Large language models (LLMs) have ushered in a new era of artificial intelligence, demonstrating
remarkable capabilities in generating high-quality text, in-context learning, and following complex
instructions (OpenAI, 2023; Touvron et al., 2023a). These advancements are primarily rooted in
the scaling up of autoregressive (AR) language models. During both training and inference, these
models leverage vast datasets and billions of parameters, employing a strict left-to-right sequential
process for memorization and generation. This approach has resulted in the emergence of intelli-
gence capable of tackling diverse tasks (Wei et al., 2022a; Hoffmann et al., 2024). However, the
ultimate upper limit of intelligence achievable through this paradigm remains an open question.
While AR mechanisms form the foundation of current LLMs, they are not without limitations (Lin
et al., 2021). Notable challenges include difficulties in future planning (Bachmann & Nagarajan,
2024; Hu* et al., 2024; Xie et al., 2024) and self-correction (Huang et al., 2024). These constraints
have spurred researchers to explore alternative architectures for next-generation LLMs.
A compelling direction in current research focuses on the development of text diffusion models (Li
et al., 2023b). Building upon the rapid evolution of diffusion models in various domains (Ho et al.,
2020; Nichol & Dhariwal, 2021; Ramesh et al., 2021), innovative text diffusion models (Li et al.,
2022; Lou et al., 2024) have opened up new possibilities for text generation. A unifying insight
across these models is the potential of diffusion language models (DLMs) for controllable (Venka-
traman et al., 2024), any-order, and parallel text generation (Gong et al., 2023a). Notably, DLMs
exhibit promising capabilities in intermediate token correction (Ye et al., 2024b) and global plan-
ning (Zhang et al., 2023), thereby addressing key limitations inherent in the AR approach.
    ∗
        Equal contribution


                                                     1
Published as a conference paper at ICLR 2025




Despite the promising potential of text diffusion models, the relatively small model size limits the
competitiveness of DLMs compared to AR models. Existing state-of-the-art DLMs such as Plaid
1B (Gulrajani & Hashimoto, 2023) and SEDD (Lou et al., 2024) are relatively small in size (127M-
1B parameters) and under-trained, with less than 400B tokens of training data. This substantial gap
in scale prevents fair comparisons with larger AR language models on many advanced capabilities
and tasks, such as chain-of-thought reasoning abilities on complex mathematical benchmarks. Re-
cent approaches (Ye et al., 2023) attempt adapt LLaMA models to DLMs based on masked language
modeling (He et al., 2023). However, they find that the base model capabilities are lost during their
adaptation stage. Pre-training at such a scale is extremely resource-intensive, and the challenge is
even more pronounced for diffusion models. These models lack the computational optimizations
that have been developed for LLMs (Samragh et al., 2024) and require significantly more resources
than their AR counterparts, as noted by Gulrajani & Hashimoto (2023).
Given these scaling challenges, pre-trained LLMs emerge as an invaluable resource that we can
leverage, considering the extensive computational efforts already invested in their development. This
strategy aligns with recent trends where new models are scaled up or adapted to new architectures
using existing LLMs (Wang et al., 2024; Zhang et al., 2024c). However, building DLMs through
adaptation from AR models is non-trivial due to fundamental differences in their language modeling
objectives. Two key distinctions present significant hurdles. First, AR models employ causal mask-
ing to prevent future information leakage, whereas diffusion models utilize bi-directional att

CANDIDATE CHUNKS
----------------
[in::7] (cited_in, 2019) A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models
A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models. Undirected neural sequence models such as BERT (Devlin et al., 2019) have received renewed interest due to their success on discriminative natural language understanding tasks such as question-answering and natural language inference. The problem of generating sequences directly from these models has received relatively little attention, in part because generating from undirected models departs significantly from conventional monotonic generation in directed sequence models. We investigate this problem by proposing a generalized model of sequence generation that unifies decoding in directed and undirected models. The proposed framework models the process of generation rather than the resulting sequence, and under this framework, we derive various neural sequence models as special cases, such as

[by::3] (cited_by_substitute, 2025) Enabling Autoregressive Models to Fill In Masked Tokens
Enabling Autoregressive Models to Fill In Masked Tokens. Historically, LLMs have been trained using either autoregressive (AR) or masked language modeling (MLM) objectives, with AR models gaining dominance in recent years. However, AR models are inherently incapable of masked infilling, which is the ability to predict masked tokens between past and future context. In contrast, MLM models suffer from intrinsic computational inefficiencies during both training and inference that hinder their scalability. This work introduces MARIA (Masked and Autoregressive Infilling Architecture), a novel approach that leverages the strengths of both paradigms to achieve state-of-the-art masked infilling performance. MARIA combines a pre-trained MLM and AR model by training a linear decoder that takes their concatenated hidden states as input. This minimal modification enables the AR model to perform infi

[by::37] (cited_by_substitute, 2025) Dream 7B: Diffusion Large Language Models
Dream 7B: Diffusion Large Language Models. We introduce Dream 7B, the most powerful open diffusion large language model to date. Unlike autoregressive (AR) models that generate tokens sequentially, Dream 7B employs discrete diffusion modeling to refine sequences in parallel through iterative denoising. Our model consistently outperforms existing diffusion language models on general, mathematical, and coding tasks. Dream 7B demonstrates superior planning abilities and inference flexibility, including arbitrary-order generation, infilling capabilities, and tunable quality-speed trade-offs. These results are achieved through simple yet effective training techniques, including AR-based LLM initialization and context-adaptive token-level noise rescheduling. We release both Dream-Base and Dream-Instruct to facilitate further research in diffusion-based language modeling.

[by::40] (cited_by_substitute, 2026) Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment
Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment. Diffusion language models (DLMs) have recently demonstrated capabilities that complement standard autoregressive (AR) models, particularly in non-sequential generation and bidirectional editing. Although recent work has shown that pretrained autoregressive checkpoints can be converted into diffusion language models, existing recipes primarily transfer parameters through continued denoising training with objective- and attention-level modifications. We instead ask whether the internal representation geometry learned by next-token prediction can be explicitly preserved during AR-to-DLM conversion. We hypothesize that much of the semantic structure learned by AR pretraining can transfer across generation orders, and thus DLM training should be viewed as relearning the decoding path rather than r

[in::29] (cited_in, 2024) Simple and Effective Masked Diffusion Language Models
Simple and Effective Masked Diffusion Language Models. While diffusion models excel at generating high-quality images, prior work reports a significant performance gap between diffusion and autoregressive (AR) methods in language modeling. In this work, we show that simple masked discrete diffusion is more performant than previously thought. We apply an effective training recipe that improves the performance of masked diffusion models and derive a simplified, Rao-Blackwellized objective that results in additional improvements. Our objective has a simple form -- it is a mixture of classical masked language modeling losses -- and can be used to train encoder-only language models that admit efficient samplers, including ones that can generate arbitrary lengths of text semi-autoregressively like a traditional language model. On language modeling benchmarks, a range of masked diffusion models

[in::35] (cited_in, 2023) Continual Pre-Training of Large Language Models: How to (re)warm your model?
Continual Pre-Training of Large Language Models: How to (re)warm your model?. Large language models (LLMs) are routinely pre-trained on billions of tokens, only to restart the process over again once new data becomes available. A much cheaper and more efficient solution would be to enable the continual pre-training of these models, i.e. updating pre-trained models with new data instead of re-training them from scratch. However, the distribution shift induced by novel data typically results in degraded performance on past data. Taking a step towards efficient continual pre-training, in this work, we examine the effect of different warm-up strategies. Our hypothesis is that the learning rate must be re-increased to improve compute efficiency when training on a new dataset. We study the warmup phase of models pre-trained on the Pile (upstream data, 300B tokens) as we continue to pre-train o

[by::10] (cited_by_substitute, 2026) dLLM: Simple Diffusion Language Modeling
dLLM: Simple Diffusion Language Modeling. Although diffusion language models (DLMs) are evolving quickly, many recent models converge on a set of shared components. These components, however, are distributed across ad-hoc research codebases or lack transparent implementations, making them difficult to reproduce or extend. As the field accelerates, there is a clear need for a unified framework that standardizes these common components while remaining flexible enough to support new methods and architectures. To address this gap, we introduce dLLM, an open-source framework that unifies the core components of diffusion language modeling -- training, inference, and evaluation -- and makes them easy to customize for new designs. With dLLM, users can reproduce, finetune, deploy, and evaluate open-source large DLMs such as LLaDA and Dream through a standardized pipeline. The framework also provi

[by::43] (cited_by_substitute, 2026) UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective
UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective. Existing methods mainly adapt pretrained autoregressive (AR) language models to masked diffusion, whereas we directly adapt them to uniform-noise diffusion, where every token remains editable during sampling. However, adapting AR checkpoints across corruption kernels remains challenging because existing DLMs use different objectives and prediction parameterizations. We establish connections among SEDD, MDLM/GIDD, M2S, and Neural CTMC by expressing their conditional losses as a single generalized Kullback--Leibler objective over model reverse rates. We further derive conversions from clean-token predictions to concrete-score, posterior-mean, and exit-rate/jump parameterizations, yielding a shared \(x_0\) interface that supports switching between mask and uniform kernels. Buil

[by::39] (cited_by_substitute, 2026) Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching
Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching. Flow matching retains the generation quality of diffusion models while enabling substantially faster inference, making it a compelling paradigm for generative modeling. However, when applied to language modeling, it exhibits fundamental limitations in representing complex latent distributions with irregular geometries, such as anisotropy and multimodality. To address these challenges, we propose a mixture-of-experts flow matching (MoE-FM) framework, which captures complex global transport geometries in latent space by decomposing them into locally specialized vector fields. Building on MoE-FM, we develop a non-autoregressive (NAR) language modeling approach, named YAN, instantiated with both Transformer and Mamba architectures. Across multiple downstream tasks, YAN achieves generation quality on par with both

[by::6] (cited_by_substitute, 2026) From Table to Cell: Attention for Better Reasoning with TABALIGN
From Table to Cell: Attention for Better Reasoning with TABALIGN. Multi-step LLM reasoning over structured tables fails because planning and execution share no explicit cell-grounding contract. Existing methods constrain the planner to a left-to-right factorization at odds with table permutation invariance, and score intermediate states by generated content alone, overlooking cell grounding. We conduct a pilot study showing that diffusion language models (DLMs) produce more human-aligned and permutation-stable cell attention on tables than autoregressive models, with a 40.2% median reduction in attention-AUROC variability under row reordering. Motivated by this, we propose TABALIGN, a planned table reasoning framework that operationalizes the contract. TABALIGN pairs a masked DLM planner, whose bidirectional denoising emits plan steps as binary cell masks, with TABATTN, a lightweight ver

[by::9] (cited_by_substitute, 2026) PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention
PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention. Discrete masked diffusion language models support bidirectional generation and infilling, but adapting pretrained autoregressive (AR) transformers requires reconciling causal pretraining with bidirectional denoising. We study this problem at the level of attention rather than claiming AR-weight reuse itself as novel. PreDiff-LM preserves causal attention within the observed prompt while allowing full bidirectional attention within the masked target. Under a matched GPT-2 Medium, WikiText-103, 90K-step setup, this hybrid mask improves unconditional perplexity from 34.1 to 28.7 and MAUVE from 0.71 to 0.78 over uniform bidirectional attention with the same AR initialization. Attention adaptation also composes with a DiffuGPT-style objective adaptation, reaching 26.9 perplexity. Pretrained initializatio

[in::13] (cited_in, 2025) TESS 2: A Large-Scale Generalist Diffusion Language Model
TESS 2: A Large-Scale Generalist Diffusion Language Model. We introduce TESS 2, a general instruction-following diffusion language model that outperforms contemporary instruction-tuned diffusion models, as well as matches and sometimes exceeds strong autoregressive (AR) models. We train TESS 2 by first adapting a strong AR model via continued pretraining with the usual cross-entropy as diffusion loss, and then performing further instruction tuning. We find that adaptation training as well as the choice of the base model is crucial for training good instruction-following diffusion models. We further propose reward guidance, a novel and modular inference-time guidance procedure to align model outputs without needing to train the underlying model. Finally, we show that TESS 2 further improves with increased inference-time compute, highlighting the utility of diffusion LMs in having fine-gra

[by::0] (cited_by_substitute, 2025) TESS 2: A Large-Scale Generalist Diffusion Language Model
TESS 2: A Large-Scale Generalist Diffusion Language Model. We introduce TESS 2, a general instructionfollowing diffusion language model that outperforms contemporary instruction-tuned diffusion models, as well as matches and sometimes exceeds strong autoregressive (AR) models.We train TESS 2 by first adapting an AR model via continued pretraining with the usual cross-entropy as diffusion loss, and then performing further instruction tuning.We find that adaptation training as well as the choice of the base model is crucial for training good instruction-following diffusion models.Furthermore, we propose reward guidance, a novel and modular inference-time guidance procedure to align model outputs without needing to train the underlying model.Finally, we show that TESS 2 further improves with increased inference-time compute, highlighting the utility of diffusion LMs in having fine-grained c

[by::5] (cited_by_substitute, 2025) Fast and Accurate Causal Parallel Decoding using Jacobi Forcing
Fast and Accurate Causal Parallel Decoding using Jacobi Forcing. Multi-token generation has emerged as a promising paradigm for accelerating transformer-based large model inference. Recent efforts primarily explore diffusion Large Language Models (dLLMs) for parallel decoding to reduce inference latency. To achieve AR-level generation quality, many techniques adapt AR models into dLLMs to enable parallel decoding. However, they suffer from limited speedup compared to AR models due to a pretrain-to-posttrain mismatch. Specifically, the masked data distribution in post-training deviates significantly from the real-world data distribution seen during pretraining, and dLLMs rely on bidirectional attention, which conflicts with the causal prior learned during pretraining and hinders the integration of exact KV cache reuse. To address this, we introduce Jacobi Forcing, a progressive distillati

[in::8] (cited_in, 2025) Dream-Coder 7B: An Open Diffusion Language Model for Code
Dream-Coder 7B: An Open Diffusion Language Model for Code. We present Dream-Coder 7B, an open-source discrete diffusion language model for code generation that exhibits emergent any-order generation capabilities. Unlike traditional autoregressive (AR) models that decode strictly left-to-right, Dream-Coder 7B adaptively determines its decoding strategy based on the coding task: sketch-first generation for complex algorithms, left-to-right generation for straightforward completions, and interleaved reasoning generation for code understanding tasks. We adapt a pretrained AR checkpoint to a discrete diffusion frameworks with a continuous-time weighted cross-entropy objective. Our post-training recipe comprises (i) supervised fine-tuning, where we mitigate padding pathologies via random truncation and a padding penalty to improve sample efficiency and stabilize generation; and (ii) reinforcem

[by::25] (cited_by_substitute, 2025) Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model
Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model. Diffusion language models (DLMs) are emerging as a compelling alternative to the dominant autoregressive paradigm, offering inherent advantages in parallel generation and bidirectional context modeling. However, for the tasks with strict structural constraints such as code generation, DLMs face a critical trade-off between inference speed and output quality, where accelerating generation by reducing sampling steps often leads to catastrophic performance collapse. We find that the fundamental reasons are: 1) the generation difficulty is non-uniform in the structured sequence decoding steps, making DLM's static acceleration strategy suboptimal; 2) the context of tokens generated by DLM evolves continuously, causing early high-confidence predictions to turn into irrevers

[in::47] (cited_in, 2023) A Cheaper and Better Diffusion Language Model with Soft-Masked Noise
A Cheaper and Better Diffusion Language Model with Soft-Masked Noise. Diffusion models that are based on iterative denoising have been recently proposed and leveraged in various generation tasks like image generation. Whereas, as a way inherently built for continuous data, existing diffusion models still have some limitations in modeling discrete data, e.g., languages. For example, the generally used Gaussian noise can not handle the discrete corruption well, and the objectives in continuous spaces fail to be stable for textual data in the diffusion process especially when the dimension is high. To alleviate these issues, we introduce a novel diffusion model for language modeling, Masked-Diffusion LM, with lower training cost and better performances, inspired by linguistic features in languages. Specifically, we design a linguistic-informed forward process which adds corruptions to the t

[by::7] (cited_by_substitute, 2026) Diffuse Thinking: Exploring Diffusion Language Models as Efficient Thought Proposers for Reasoning
Diffuse Thinking: Exploring Diffusion Language Models as Efficient Thought Proposers for Reasoning. Large language models (LLMs) have demonstrated strong capabilities in complex reasoning tasks, yet the multi-step reasoning processes often lead to expensive computation cost.The recent advance of diffusion language models (DLMs) adopts a parallel, non-autoregressive generation mechanism, which enables the efficient production of multiple outputs.In this paper, we explore a collaborative reasoning framework that combines diffusion-based generation with autoregressive evaluation.Specifically, we leverage DLMs to efficiently propose stepwise reasoning thoughts, and employ LLMs as evaluators to assess and select candidates based on their plausibility and correctness.By decoupling proposal generation from evaluation, our framework exploits the strengths of both models: efficient exploration fr

[in::12] (cited_in, 2025) Flexible-length Text Infilling for Discrete Diffusion Models
Flexible-length Text Infilling for Discrete Diffusion Models. Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability 

[in::17] (cited_in, 2023) Muse: Text-To-Image Generation via Masked Generative Transformers
Muse: Text-To-Image Generation via Masked Generative Transformers. We present Muse, a text-to-image Transformer model that achieves state-of-the-art image generation performance while being significantly more efficient than diffusion or autoregressive models. Muse is trained on a masked modeling task in discrete token space: given the text embedding extracted from a pre-trained large language model (LLM), Muse is trained to predict randomly masked image tokens. Compared to pixel-space diffusion models, such as Imagen and DALL-E 2, Muse is significantly more efficient due to the use of discrete tokens and requiring fewer sampling iterations; compared to autoregressive models, such as Parti, Muse is more efficient due to the use of parallel decoding. The use of a pre-trained LLM enables fine-grained language understanding, translating to high-fidelity image generation and the understanding

Return JSON: {"scores": {"<chunk_id>": <int 0-10>, ...}}
