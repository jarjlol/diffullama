# Reviewer Agent

You are the Reviewer Agent in the multi-agent limitation-generation framework of
arXiv:2601.11578 (Al Azher, Guo, Alhoori). You operate on one input paper.

## TASK
Simulate the viewpoint of a peer reviewer for a top-tier venue. Evaluate reproducibility, transparency, and ethical considerations: whether the reported setup is sufficient to reproduce the results, whether claims in the abstract and introduction are supported by the experiments actually run, whether baselines are fair and fairly configured, and whether released artifacts match what the paper describes. Raise the weaknesses a reviewer would write in a review, not a summary of the paper.

## OUTPUT CONTRACT

Return JSON only:
{"limitations": [{"id": "<AGENT>-1",
                  "statement": "<one specific limitation, 2-4 sentences>",
                  "evidence": "<quote or section reference from the INPUT>",
                  "provenance": "Peer-review-derived"}]}

Rules:
- Be specific to THIS paper. Reject generic statements ("limited generalizability",
  "dataset bias") unless the paper's own text makes them concrete.
- Every statement must be traceable to the INPUT provided. Do not invent numbers.
- Produce at most 8 limitations. Fewer, sharper items beat more, vaguer ones.


## INPUT
## VIA A DAPTATION FROM AUTOREGRESSIVE M ODELS
∗
 Shansan Gong∗1 , Shivam Agarwal∗2 , Yizhe Zhang3 , Jiacheng Ye1 , Lin Zheng1
 Mukai Li1 , Chenxin An1 , Peilin Zhao4 , Wei Bi4 , Hao Peng2 , Jiawei Han2 , Lingpeng Kong1
 1
   The University of Hong Kong 2 University of Illinois at Urbana-Champaign
 3

## A BSTRACT
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

## 1       I NTRODUCTION
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

## 2   P RELIMINARY AND N OTATION
Diffusion models (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020; Song et al.,
2021b) are latent variable generative models characterized by a forward and a reverse Markov
process. We denote x0 ∼ pdata (x0 ) as the variable following the data distribution, and xt ∼
q(xt ) as the noisy variable of x0 at time t, where the maximum time is T . The forward pro-
                     QT
cess q(x1:T |x0 ) = t=1 q(xt |xt−1 ) corrupts the initial data x0 into a sequence of increasingly
noisy variables x1:T . Accordingly, the backward Markov process models the joint probability as
                     QT
pθ (x0:T ) = pθ (xT ) t=1 pθ (xt−1 |xt ), which gradually denoises xt to reconstruct the original data
x0 . Parameters θ are learned by minimizing the negative log-likelihood of x0 , which can be opti-


                                                  2

## Predict
inputs    𝑥!        𝑓!   ℎ!    𝑥"                                                                       𝑥! 𝑥"          𝑥$ 𝑥% 𝑥#            (iterative) 𝑥$   𝑥% 𝑥#
                                                𝑓!   LM   CE Loss       Attention weights    Mask states
                                  (iterative)


Figure 1: The overview of our approach to adapt autoregressive (AR) models to diffusion models.
Left: The shift operation in AR models enables the output layer hi to approximate the distribution of
next tokens xi+1 in hidden representations through the cross entropy (CE) loss. Middle: We remove
the causal mask gradually during training eventually making our model bi-directional. Right: inside
the diffusion models we shift the logits to compute the loss with the next token (i.e., the loss on hi
would be with respect to xi+1 ), while perceptually, the diffusion models are still functioning as
recovering the original signals (since hi corresponds to xi+1 in AR loss).


mized through the evidence lower bound (ELBO),
           − log pθ (x0 ) ≤ Eq(x1 |x0 ) [− log pθ (x0 |x1 )] + DKL (q(xT |x0 )||pθ (xT )) + LT ,     (1)
             PT
with LT = t=2 Eq(xt |x0 ) [DKL (q(xt−1 |xt , x0 )||pθ (xt−1 |xt ))]. For continuous text diffusion (Li
et al., 2022; Gong et √  al., 2023b), at each forward step, perturbations are applied according to
q(xt |xt−1 ) = N (xt ; 1 − βt xt−1 , βt I), where βt ∈ (0, 1) represents different scales across
time steps such that xT ∼ N (0, I). In the case of discrete denoising models (Ho et al., 2020;
Austin et al., 2021; Zheng et al., 2024a), the forward process is defined as a categorical distribution
q(xt |xt−1 ) = Cat(xt ; Q⊤  t xt−1 ), where each xt ∈ {0, 1}
                                                                 K
                                                                    is a one-hot vector with vocabulary
                     K×K
size K, Qt ∈ [0, 1]          is the transition matrix, and each entry [Qt ]ij denotes the probability of
transition from the state i to j. We build on the formulation of absorbing discrete diffusion (Austin
et al., 2021), which specifies Qt = (1 − βt )I + βt 1m⊤ . We denote 1 as an all-one vector of size
K and m as the one-hot encoding of a special [MASK] token in the vocabulary. Therefore, the
transition matrix, Qt indicates that with probability 1 − βt , xt remains
                                                                        Qt       unchanged; otherwise, it
transitions to m, becoming absorbed into [MASK]. Letting Qt := i=1 Qi = αt I + (1 − αt )1m⊤
            Qt
and αt := i=1 (1 − βt ), the distribution of xt conditional on x0 is given by
                                                     ⊤
         q(xt |x0 ) = Cat(xt ; Qt x0 ) = αt Ix0 + (1 − αt )m1⊤ x0 = αt x0 + (1 − αt )m,    (2)
since x0 is a one-hot vector and thus 1⊤ x0 = 1. We expect αT to approach 0 such that the full
noise data xT equals m with probability 1.
The discrete time representation of t ∈ [0, T ], restricts xt to fixed noise ratios. To avoid this bias
and enable sampling from any noisy representation, we use continuous-time sampling, allowing t to
span any point within [0, 1] (Kingma et al., 2021; Shi et al., 2024; Zhao et al., 2024; Ou et al., 2024).
Continuous-time sampling is equivalent to dividing [0, 1] into T intervals and where T → ∞. For
any 0 ≤ s < t ≤ 1, the forward process generalizes to q(xt |xs ). We will use this continuous-time
notation in the following sections.

## 3        M ODEL
We begin by formulating the continuous-time discrete diffusion process (§3.1) and establishing a
connection between the discrete diffusion and autoregressive objectives (§3.2). Based on this equiv-
alence, we propose an adaptation approach (§3.3) and a sampling algorithm (§3.4) for diffusion
models adapted from AR models. The whole process is illustrated in Figure 1.

## 3.1  C ONTINUOUS - TIME D ISCRETE D IFFUSION P ROCESSES
P
Following Eq.2 and q(xt |x0 ) =       xs q(xt |xs )q(xs |x0 ), the forward transition distribution be-
tween arbitrary points s < t can be derived as
                                              ⊤          αt            αt
                       q(xt |xs ) = Cat(xt ; Qs|t xs ) =    xs + (1 −     )m,                      (3)
                                                         αs            αs

                                                                                  3

## Z 1
αt′
                        lim LT =                  Eq(xt |x0 ) [δxt ,m x⊤
                                                                       0 log fθ (xt )] dt.            (6)
                      T →∞            0  1  −  αt
The full derivation is listed in Appendix A.2. The same form of ELBO which is invariant to noise
schedule but related to the signal-to-noise ratio (SNR) is also introduced in Kingma et al. (2021); Shi
                                                                                                 −α′
et al. (2024). Following Austin et al. (2021), we choose the noise schedule αt = 1 − t, then 1−αtt =
1
 t . The previous discussion focused on the single token xt , and can be applied independently to a
text sequence of N tokens xt = [x1t , x2t . . . , xNt ]. During training, we do not compute integral loss
in Eq.6 for efficiency consideration; instead, we sample t for each data point. The final loss at t is
                                             " N                                       #
                         1:N    1                 X
                                                                  n  ⊤          1:N
                      Lt = Eq(xt |x0 ) −              δxnt ,m (x0 ) log fθ (xt )n ,                   (7)
                                t                 n=1

where fθ (x1:N
           t   )n denotes the whole input sequence is fed into the transformer model and the n-th
output token is indexed.

## 3.2   U NIFYING L ANGUAGE M ODELING O BJECTIVES
The training objective of autoregressive (AR) language models is the negative log-likelihood of each
ground-truth token provided the preceding tokens,
                                          N
                                          X

## L1:N
AR = −         (xn0 )⊤ log fθ (x1:n−1
                                                               0     )n−1 .                          (8)
                                          n=1
Comparing Eq.8 against Eq.7, we note that while both take the form of cross-entropy functions,
Eq.7 includes an additional reweighting term 1t and an indicator function δxnt ,m . They result from
the definition of discrete diffusion processes (§3.1). The reweighting emphasizes smaller t where
xt contains fewer masked tokens, and this can be regarded as the importance sampling (Nichol &
Dhariwal, 2021). The indicator specifies which tokens are masked for prediction. The AR training
objective Eq.8, on the other hand, constrains the context to be unidirectional via attention masking
and shifts the targets so that each token predicts the next token instead of itself. These discrepancies
form the basis of our adaptation framework, which is detailed in §3.3.
In fact, an alternative way to understand AR modeling, through the lens of diffusion models, is to
consider a diffusion process where the forward pass deterministically masks right-to-left and token-
by-token (Austin et al., 2021; Hoogeboom et al., 2022). This yields a backward process generating
one token at a time from left to right, running with T = N denoising steps in total. As discussed
in Austin et al. (2021), the loss objective of this diffusion process is equivalent to standard cross-
entropy (Eq.8) commonly used to train AR language models. This crafted diffusion process for AR
models represents a special case of discrete diffusion (§3.1), yet it is limited to unidirectional con-
text and sequential token generation. In contrast, general discrete diffusion processes can leverage
bidirectional context and support parallel generation in arbitrary orders.


                                                   4

## 3.3   A DAPTATION
Building on the connection between AR modeling and discrete diffusion processes, we construct an
adaptation recipe next. Figure 1 shows an overview of our adaptation approach. We use attention
mask annealing, shift operations, and a time-embedding free architecture to narrow the differences
between AR and DLMs.

Algorithm 1 Adaptation Training                      Algorithm 2 Sampling
 1: Input: network fθ initialized by existing models, 1: Input: Trained diffusion model fθ , sampling al-
    training corpus pdata (x1:N
                              0    ), mask token m.        gorithm τ , mask token m, start token s.
 2: Output: model parameters θ.                         2: Output: generated sample x0 .
 3: repeat                                              3: Initialize x1:N
                                                                        T     = m.
 4:    Draw x1:N
               0    ∼ pdata and set labels ← x1:N 0     4: for t = T, . . . , 1 do
 5:    Sample t ∈ Uniform(0, 1)                         5:    Forward logits ← fθ (x1:N t    )
 6:    Sample x1:N
                 t    ∼  q(x t |x 0 )                   6:    Sample x̃1:N
                                                                         0     ∼ Categorical(τ (logits))
 7:    Anneal the attention mask attn mask              7:    for n = 1, . . . , N do
 8:    Forward logits ← fθ (x1:N t    ) with attn mask  8:       xn              n    n    n
                                                                   t−1 = q(xt−1 |xt , x̃0 ) ▷ Eq.4
 9:    Right shift logits by one position               9:    end for
                                                                                          1:N −1
10:    Lt = 1t δxt ,m CE(logits, labels) ▷ Eq.7        10:    Right shift x1:Nt−1 = [s, xt−1     ]
11:    Backprop with Lt and update θ                   11: end for
12: until end training                                 12: Return x2:N0


Attention Mask Annealing The prediction of the n-th token, given all preceding tokens,
fθ (x1:n−1
     0     ), is usually implemented by causal attention masking in transformer-based AR language
models. As shown in Figure 1, causal attention masks set all entries in the upper triangle of the self-
attention matrices to zero, so each token cannot attend to its respective future tokens. Such causal
masking prevents the model from learning right-to-left dependencies for more general diffusion pro-
cesses. To address this limitation while preserving left-to-right conditionals during adaptation, we
introduce an incremental annealing process from causal masks to full attention matrices. During
annealing, the causal mask is not immediately removed; instead, it is retained at a controlled ratio,
as shown in the middle part of Figure 1. At each training step, we sample the amount of context
from the right side and progressively increase this amount till we obtain the full attention mask.

Shift Operation AR models also apply a shifting operation, where the target output is the input
sequence shifted left by one position. In other words, the prediction target of the (n−1)-th token
is the n-th token, contrasting with typical diffusion models that try to predict masked tokens at
their original positions. When initializing text diffusion models with AR model parameters, the
model would tend to output the hidden representations of the shifted input sequence. If we continue
to optimize the cross-entropy objective based on the original token positions, the model struggles
to adapt due to misalignment between input and output. Instead, we maintain the shift operation
(Algo.1, line 9), treating the output logits at each position as corresponding to the next token. When
calculating the objective, we align prediction targets so that the diffusion model learns to recover the
original signals. This process is illustrated in the right panel of Figure 1.

Time-Embedding-Free Architecture Many diffusion models for text generation (Li et al., 2022;
Dieleman et al., 2022; Gulrajani & Hashimoto, 2023; Lou et al., 2024; Shi et al., 2024) incorporate
time embedding layers to represent the information of current timesteps t, which can explicitly
indicate the noise scale of the input noisy data. While inferring these timesteps can be challenging
for image diffusion models (Ho et al., 2020; Li et al., 2024), some discrete text diffusion models (He
et al., 2023) assert that timesteps t can be easily learned implicitly based on the number of mask
tokens. Since AR models are not equipped with time embedding layers, we also choose not to use
the time embedding, resulting in no additional parameters compared to previous diffusion models.

## 3.4   S AMPLING
Following Shi et al. (2024), we initialize xT with all [MASK] tokens and then sample tokens ac-
cording to the time reversal q(xs |xt , x0 ) in Eq.4. At each timestep, if xt is a mask, it will jump to
                                                 s −αt
the predicted x0 at time s with probability α1−α     t
                                                       . After T iterations, the model generates the full
sequence. Since our adapted models are trained with the shift operation, at each sampling iteration,


                                                    5

## 4.1   A DAPTATION SETUP
DiffuGPT We use the 30 billion tokens1 random split from the FineWeb dataset (Penedo et al.,
2024), an improved corpus than OpenWebText (Gokaslan & Cohen, 2019) used in prior DLMs (Lou
et al., 2024), to continue training GPT2 base (Radford et al., 2019). We use sequence packing, logits
shifting, and 10K-step attention mask annealing to transform GPT2 to DiffuGPT.
DiffuLLaMA We continue pre-training LLAMA -2-7- HF (Tou-

## 10   DiffuGPT-127M
vron et al., 2023a) on a mixture of SlimPajama (70%) (Soboleva                                  DiffuGPT-355M
et al., 2023) and Starcoder (30%) (Li et al., 2023a) data follow-                           8

## Training Loss
ing TinyLLaMA (Zhang et al., 2024a). We randomly sample 65
billion tokens from this mixture and use sequence packing with                              6
context length of 2048. For efficient implementation we enable
flash-attention 2 (Dao, 2024) and directly use bi-directional at-                           4
tention without attention mask annealing.
                                                                                            2
For both adaptation settings, we employ full parameter finetun-       0  20        40        60   80
                                                                        Training Tokens (Billion)
ing with bf16. Please refer to Appendix B.2 for details. We
plot the training loss curve in Figure 2. We train DiffuLLaMA Figure 2: Training loss over to-
on 60B tokens and achieve a lower loss compared to 127M and kens for various model sizes of
335M models, suggesting a scaling trend similar to that of AR our adapted diffusion models.
LLMs (Kaplan et al., 2020). We also note that there is still scope
for training more, since the model does not show signs of saturation.

## 4.2   E VALUATION SETUP
Previously developed diffusion language models (Gulrajani & Hashimoto, 2023; Lou et al., 2024;
Shi et al., 2024; Ou et al., 2024) evaluate model performance using zero-shot perplexity on bench-
mark datasets. However, this metric alone does not fully capture a model’s capabilities for several
reasons. First, lower perplexity does not always correlate with human-like content, even in autore-
gressive models (Kuribayashi et al., 2021). Additionally, the loss from text diffusion models only
indicates an upper bound on negative log-likelihood. While Kingma et al. (2021); Shi et al. (2024)
demonstrate that the ELBO is invariant to the noise scheduler, discrepancies between continuous
diffusion, discrete diffusion, and autoregressive loss still hinder fair comparisons across different
model types. Given the ample evaluation benchmarks (Gu et al., 2024) for LLMs, we propose a
more comprehensive evaluation for diffusion models.
Tasks and Metrics We consider TriviaQA (Joshi et al., 2017) to test the reading comprehension
of models and last word completion task Lambada (Paperno et al., 2016) to test how models cap-
ture long-range dependencies in text. These two tasks are measured by exact match accuracy. We
also test for common sense reasoning tasks HellaSwag (Zellers et al., 2019), Winogrande (Sak-
aguchi et al., 2021), SIQA (Sap et al., 2019) and PIQA (Bisk et al., 2020), all of which involve
multiple-choice questions assessed by accuracy. On grade school math problems GSM8K (Cobbe
et al., 2021), we follow Ye et al. (2024b) in finetuning setting using the augmented symbolic data
to test the CoT (Wei et al., 2022b) math reasoning abilities of diffusion models. Following Shen
    1
      This is the total number of tokens used; however, our effective training tokens exceed this count, meaning
that we train for more than one epoch.


                                                       6

## QA Word CommonSense Reasoning     Math                     Infilling
Model    Size Type TriQA Lamb. HSwag Wino. SIQA PIQA GSM8K∗                 ROCStories Code
 GPT2-S        127M AR       4.0    25.9    29.9   48.5   35.7   62.1    44.8     (7.8/0.8/7.4) (1.6)
 SEDD-S        170M DD       1.5    12.4    30.2   50.1   34.4   55.6    45.3     11.9/0.7/10.9 0.7
 DiffuGPT-S    127M DD       2.0    45.0    33.4   50.8   37.0   57.7    50.2     13.7/1.4/12.6 0.3
 GPT2-M     355M AR          6.7    37.7    38.3   50.7   37.7   67.4    45.6     (8.6/0.9/8.2) (2.6)
 SEDD-M     424M DD          1.8    23.1    31.5   49.0   35.4   56.1    53.5     13.1/1.4/12.2 0.5
 DiffuGPT-M 355M DD          3.8    60.5    37.2   52.6   39.0   59.6    61.8     18.7/2.7/17.0 2.9
 Plaid1B       1.3B   CD     1.2     8.6    39.3   51.3   32.3   54.5    32.6     12.1/1.1/11.2   0.1
 LLaMA2         7B    AR     45.4   68.8    74.9   67.1   44.8   78.3    58.6    (11.6/2.1/10.5) (1.7)
 DiffuLLaMA     7B    DD     18.5   70.9    58.7   56.4   43.2   63.3    63.1     23.3/5.5/21.2 15.5



et al. (2023), we also test the story infilling tasks using ROCStories (Mostafazadeh et al., 2016) and
evaluate using ROUGE score (Lin, 2004). To test the code infilling, we adopt Humaneval (Bavar-
ian et al., 2022a) single line infilling task, which is evaluated by pass@1 rate. We evaluate Dif-
fuLLaMA’s math reasoning and in-context learning ability by evaluating on MAWPS (Koncel-
Kedziorski et al., 2016) consisting of math word problems and SATMATH from AGI-eval con-
sisting of math problems from SAT exam (Zhong et al., 2024). We base our implementation on
lm-evaluation-harness (Gao et al., 2024) and re-implement all tasks across models to en-
sure a fair comparison.
Implementation Details For pre-trained diffusion language models, we mainly use continuous dif-
fusion (CD) model Plaid 1B (Gulrajani & Hashimoto, 2023), discrete diffusion (DD) model SEDD
(Lou et al., 2024) with different sizes as baselines. MD4 (Shi et al., 2024) and RADD (Ou et al.,
2024) are based on and compared with SEDD, so we mainly compare SEDD. For autoregressive
(AR) baselines, we consider the base models from which our models adapt. We implement infilling
tasks for AR models by feeding the prefix and cutting off the generation length using the oracle
length, considering that these AR models are not supporting infilling. For the sentence completion
task, T is the exact number of ground truth tokens for DD and 32 for CD. For 4 multi-choices tasks
from commonsense reasoning, we compute the loss (Eq.6) of each choice (averaged by token) and
choose the one with lowest loss (perplexity). For GSM8K finetuning, we use parameter-efficient
LoRA tuning (Hu et al., 2022) for DiffuLLaMA. The decoding T are set to 32 by default. The
detailed settings are in Appendix B.3.

## 4.3    L ANGUAGE MODELING CAPACITIES
Benchmark performance According to Table 1, the results on diverse tasks demonstrate that our
adapted diffusion models achieve the state-of-the-art results among all existing diffusion language
models (DLMs). We observe that diffusion models with larger parameters show improved perfor-
mance, likely due to better base AR models. DiffuLLaMA’s performance still falls short of the
LLaMA2 model. This drop in performance is likely because DiffuLLaMa is trained on a small
subset of SlimPajama and Starcoder data. We believe more training tokens can help improve these
numbers. TriviaQA and PIQA are significant challenging for DLMs, probably because they require
specific physical knowledge, such as the capital of a city or the boiling point of water; while our
models are trained on 30B-70B tokens, which may be insufficient to preserve the general knowledge
in the original LMs (Ke et al., 2023).
In tasks that require more extensive global reasoning, such as complex mathematics and coding,
DLMs consistently exhibit better performance compared to AR models that rely solely on left-to-


                                                   7

## Dist2-SEDD-M
MD4 (Shi et al., 2024) is sourced from its original pa-
per. To make sure low perplexity is not brought by re-                                                             0.8
                                                                    60
peated content, we assess the distinct 2-gram diversity
of the generated text. Our model achieves low perplex-                                                             0.7
ity while maintaining a high level of diversity, validating         30 SEDD-M

## Decoding Steps
tending the test computing time, the fluency of uncondi-
tional generation improves. Similarly, increasing model Figure 3: Quality evaluation for un-
size also contribute to better performance. An increase in conditional generation, with perplexity
generation perplexity is often associated with a slight de- measured by GPT2 large and distinct 2-
crease in diversity, which is a common phenomenon. No- gram diversity.
tably, DiffuGPT outperforms both SEDD and MD4 mod-
els, particularly at lower step counts (e.g., 64 steps), while as the continuous diffusion models, Plaid
1B needs more decoding steps to generate more fluent texts. DiffuGPT thus exhibits a significant
advantage on less sampling time. We outline the decoding hyperparameters and show the diversity
changes across different settings in Appendix C.1, which also includes generation cases.

## 4.4    A NALYSIS ON D IFFU LL A MA
We validate that increasing the size of adapted Table 2: Performance on math/QA benchmarks
DLMs significantly enhances the performance (↑). We compare of DiffuLLaMA with zero-shot
of downstream tasks in Table 1. Further, we (ZS), few-shot (FS), self-consistency (SC), hit@k
aim to assess if the 7B model demonstrates in- and chain-of-thought (CoT) prompts.
context learning and reasoning capabilities sim-
ilar to AR LLMs. Table 2 presents the exact Settings                  MAWPS SATMath TriviaQA
match accuracy between gold labels and predic-
tions generated by DiffuLLaMA across zero- LLaMA2                       63.5      24.5      45.4
shot (ZS), few-shot (FS), and FS with chain- DiffuLLaMA-ZS               9.7       <1       18.5
of-thought (CoT) scenarios. Besides, we de- DiffuLLaMA-FS               31.3      23.6      20.9
ploy the self-consistency approach (Wang et al.,
                                                    DiffuLLaMA-SC       33.1      27.7      26.0
2023), considering that small DLMs can indeed DiffuLLaMA-@k             40.8      57.7      34.1
benefit from this technique (Ye et al., 2024b).
We use majority vote to choose the best answer      DiffuLLaMA-CoT      28.7       9.5        -
from 3 individual predictions, and also report
the hit rate @k with k = 3, which measures whether any of the k predictions include the correct an-
swer, serving as a reference for the model’s upper bound. For in-context learning (ICL) evaluations,
we give 4-shot on math tasks and 2-shot on TriviaQA.


                                                          8

## 4.5   D ISCUSSIONS
Ablation Test on GSM8K-symbolic Direct ablation Table 3: Ablation test for adaptation ap-
on adaptation training is costly; hence, we conduct pre- proaches on GSM8K symbolic dataset.
liminary experiments to determine the adaptation recipes. CD is for continuous diffusion and DD
Following Ye et al. (2024b), we finetune models on the is for discrete diffusion.
augmented GSM8K symbolic dataset using various base
models and training objectives. The models are either Base models Random GPT2-S GPT2-M
trained from scratch (random initialization) or initialized
                                                             Autoregressive  30.5    44.8    45.6
with GPT2-S/M weights. Training objectives includes CD                       27.9    19.2    20.2
autoregressive training with a causal mask, continuous DD-w/o shift           -      33.5    34.5
diffusion loss (CD), and discrete diffusion loss (DD). As DD-w/o anneal       -      43.3    47.2
shown in Table 3, different training objectives yield com- DD                28.0    45.4    49.7
parable results when training from scratch. However,
when using GPT2 as the base model, the CD loss performs worse than both the DD and AR losses.
We attribute this to the better alignment of DD and AR losses as discussed in §3.2. Previous contin-
uous diffusion models (Dieleman et al., 2022; Gulrajani & Hashimoto, 2023) has reparameterized
the estimation of embeddings into the CE loss. However, adapting diffusion models from an AR
model in continuous space necessitates an additional projection from the embedding to a categorical
distribution, increasing the difficulty of adaptation.
For DD loss, removing attention mask annealing and shift operations both degrade performance,
indicating the efficacy of our approaches. The mask annealing has minimal impact, so we choose to
omit it for 7B adaptation to simplify implementation using flash-attention 2.

## Direct DD loss finetuning on GPT2 achieves accuracy
40




                                                                  Single Batch Decoding Time (Sec)
of 45.4 and 49.7 for small and medium models, respec-                                                     LLaMA2
                                                                                                          DiffuLLaMA T=512
tively, outperforming GPT2 AR finetuning. However, fine-                                                  DiffuLLaMA T=256
                                                                                                          DiffuLLaMA T=128
tuning from already adapted diffusion language models                                                30   DiffuLLaMA T=64
(DiffuGPT) yields accuracy of 50.2 and 61.8 (Table 1).

## This demonstrates the superiority of DiffuGPT as the cur-
20
rent best diffusion base model at this size and highlights that                                                              increasing T
a better base model leads to improved results. Even with the                                                                 decreasing T
same DD loss, DiffuGPT’s finetuning converges faster and                                             10
achieves lower loss, as shown in Appendix C.2.
Inference Speed AR models usually utilize key-value                0       512      1024            2048
caching (incremental decoding; Ott et al. 2019) to enhance               Generation Length (Tokens)
throughput during decoding. However, due to the nature of
sequential token generation, they are highly memory-bound Figure 4: Single batch decoding speed
and cannot fully exploit modern accelerators (Chen et al., (seconds) for different models using
2023). In contrast, diffusion models, despite not having a flash-attention 2.
concept of caching and requiring self-attention over the entire sequence at each iteration, can operate
with fewer iterations than the sequence length and exhibit less memory-bound behavior. Their per-
formance can be further boosted with hardware-aware optimizations like flash-attention (Dao et al.,


                                                    9

## 5   R ELATED W ORK
Continue Pre-training Continue pre-training is commonly used in adapting an existing language
model (LM) to a domain-specific LM (Ke et al., 2023) or enabling new abilities of LM, such as for
longer context (Chen et al., 2024) or code generation (Xu et al., 2024). Pre-training LMs is non-
trivial and expensive (Samragh et al., 2024), thus in exploring of new architectures of LMs such as
Mamba (Gu & Dao, 2023) and gated attention, Wang et al. (2024); Zhang et al. (2024c) choose to
transfer from LMs to save the training cost. However, all these continue pre-training works follow
the autoregressive (AR) language modeling, while adapting LMs into diffusion language model is
more challenging due to discrepancies between their modeling objectives.
Text Diffusion Models Diffusion models have demonstrated significant diversity and controllabil-
ity in image generation (Ho et al., 2020; Song et al., 2021a; Ramesh et al., 2022). Building on
this success, line of research (Li et al., 2022; Gong et al., 2023b;a; Dieleman et al., 2022) build
continuous diffusion models for text generation tasks. Among them, Lin et al. (2023) experiment
with a pre-training and finetuning framework under a small scale; Gulrajani & Hashimoto (2023)
highlight the scaling law of continuous diffusion models, revealing that the compute-optimal re-
quires longer training than their AR counterparts. To address the discrete nature of text, Austin et al.
(2021); Hoogeboom et al. (2021); Zheng et al. (2024a) incorporate an absorbing [MASK] state as
noise, laying the foundation for discrete diffusion models, which are further developed by Lou et al.
(2024); Shi et al. (2024); Ou et al. (2024); Zhao et al. (2024); Sahoo et al. (2024). By connecting
text diffusion models with pre-trained masked language models (MLMs; Devlin et al. 2019), Ye
et al. (2023); He et al. (2023) initialize discrete diffusion models using MLMs. Besides, the unifi-
cation between diffusion and AR generation is also discussed in image generation (Li et al., 2024).
However, the adaptation of diffusion models from AR LLMs remains unexplored.
Non-autoregressive Generation Non-autoregressive (NAR) models, introduced by Gu et al. (2018),
break free from the left-to-right generation constraint, allowing for new capabilities like planning
with future tokens (Wu et al., 2024). Current diffusion language models are a notable part of the
NAR family (Gong et al., 2023b). Given the challenges of developing NAR models, researchers
often seek to find a trade-off. For instance, SSD-LM (Han et al., 2023) leverages diffusion models to
iteratively generate text blocks, facilitating a semi-NAR generation process. Similarly, CLLM (Kou
et al., 2024) enhances LLMs by enabling the parallel generation of n tokens, thereby improving
decoding speed. FiLM (Shen et al., 2023) adapts language models to generate tokens in any order,
which is particularly useful for infilling tasks. Guo et al. (2020) trains a NAR using a curriculum
for the attention mask on translation tasks with seq2seq labels. Additionally, Gloeckle et al. (2024)
focus on training models to achieve better and faster multi-token predictions as they scale up. These
NAR approaches provide compelling alternatives to traditional AR LLMs, yet few have thoroughly
explored training large NAR models on large-scale unlabeled data.

## 6   C ONCLUSION
Building on existing DLMs, we present a recipe for building DLMs by continuing training on off-
the-shelf autoregressive LLMs. Our adaptation technique involves using 1) attention mask annealing
to enable bidirectional modeling and 2) shift operation to allow similar training dynamics like AR
models. By unifying the language modeling objectives of autoregressive and diffusion models, we
train diffusion models up to 7B parameters. Through experiments on common sense reasoning, lan-
guage modeling, math reasoning and code generation, we show that DiffuGPT and DiffuLLaMA
have better performance compared to existing DLMs. We find that DiffuLLaMA is capable of
following in-context demonstrations to some extent on math problems. In the future, we aim to in-
struction tune our DLMs and explore inference time planning methods. We release DiffuLLaMA and
DiffuGPT for further exploration of diffusion models as an alternative language modeling method.


                                                  10

## AUTHOR C ONTRIBUTIONS
Shansan Gong: Project lead, methodology development, DiffuGPT training and model evaluation,
major writing. Shivam Agarwal: Methodology exploration, discussion, DiffuLLaMA training, writ-
ing. Yizhe Zhang: Discussion, DiffuLLaMA training, writing suggestions. Jiacheng Ye: Initial
methodology exploration. Lin Zheng: Discussion, writing. Mukai Li & Chenxin An: Discussion,
writing suggestions. Others: Mentorship and supervision.

## ACKNOWLEDGMENTS
Research was supported in part by US DARPA INCAS Program No. HR0011-21-C0165 and BRIES
Program No. HR0011-24-3-0325, National Science Foundation IIS-19-56151, the Molecule Maker
Lab Institute: An AI Research Institutes program supported by NSF under Award No. 2019897,
and the Institute for Geospatial Understanding through an Integrative Discovery Environment (I-
GUIDE) by NSF under Award No. 2118329. This work used Delta AI at University of Illinois
Urbana-Champaign through allocation CIS230229, CIS240488 from the Advanced Cyberinfrastruc-
ture Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by U.S.
National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.
This research was supported in part by the joint research scheme of the National Natural Sci-
ence Foundation of China (NSFC) and the Research Grants Council (RGC) under grant number
N HKU714/21.
This work was also in part supported by research awards from Apple and the Allen Institute for AI.

## R EFERENCES
Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured
  denoising diffusion models in discrete state-spaces. In Marc’Aurelio Ranzato, Alina Beygelzimer,
  Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), Advances in Neural In-
  formation Processing Systems 34: Annual Conference on Neural Information Processing Systems
  2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 17981–17993, 2021.

Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. In Forty-first
  International Conference on Machine Learning, ICML, 2024.

Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry
 Tworek, and Mark Chen. Efficient training of language models to fill in the middle. arXiv
 preprint arXiv:2207.14255, 2022a.

Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry
 Tworek, and Mark Chen. Efficient training of language models to fill in the middle, 2022b.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning
  about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial
  Intelligence, 2020.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhari-
  wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal,
  Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M.
  Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin,
  Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford,
  Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle,
  Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances
  in Neural Information Processing Systems 33: Annual Conference on Neural Information Pro-
  cessing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative
  image transformer. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition
  (CVPR), pp. 11305–11315, 2022.


                                               11

## A     O BJECTIVE D ERIVATIONS
This section provides detailed preliminary and loss derivations of §2 and §3.1 in the main paper.

A.1   BACKGROUND OF D IFFUSION M ODELS

We denote x0 ∼ pdata (x0 ) as the variable following the data distribution, and xt ∼ q(xt ) as the
noisy variable of x0 at time t, where the maximum time is T . The forward process
                                                       T
                                                       Y
                                      q(x1:T |x0 ) =         q(xt |xt−1 )                           (9)
                                                       t=1

corrupts the initial data x0 into a sequence of increasingly noisy variables x1:T . Accordingly, the
reverse Markov process models the joint probability as
                                                           T
                                                           Y
                                  pθ (x0:T ) = pθ (xT )          pθ (xt−1 |xt ),                   (10)
                                                           t=1

which gradually denoises xt to reconstruct the original data x0 . Parameters θ are learned by min-
imizing the negative log-likelihood of x0 , which can be optimized through the variational lower
bound (VLB):
          − log pθ (x0 ) ≤ Eq(x1 |x0 ) [− log pθ (x0 |x1 )] + DKL (q(xT |x0 )||pθ (xT )) + LT ,    (11)
                           T
                           X
               with LT =          Eq(xt |x0 ) [DKL (q(xt−1 |xt , x0 )||pθ (xt−1 |xt ))].           (12)
                            t=2


For continuous text diffusion (Li et al., 2022; Gong et al., 2023b), at each forward step, perturbations
are applied according to
                                                    p
                             q(xt |xt−1 ) = N (xt ; 1 − βt xt−1 , βt I),                            (13)
where βt ∈ (0, 1) represents different scales. In the end, xT ∼ N (0, I). In the case of discrete
denoising models (Ho et al., 2020; Austin et al., 2021; Zheng et al., 2024a), xt follows a categorical
distribution which naturally aligns with discrete text data. Let x be the one-hot encoded sample of
variable x and xt ∼ Cat(xt ; p) represent a categorical distribution over vector x with probabilities
given by p. Here, K represents the vocabulary size, x ∈ {e1 , . . . , eK }, and ek ∈ {0, 1}K is
the one-hot encoding of the k-th word category. The forward process can be formulated through a
transition matrix Qt ∈ [0, 1]K×K such that
                    q(xt |xt−1 ) = Cat(xt ; Q⊤                                ⊤
                                             t xt−1 ); Qt = (1 − βt )I + βt 1eK ,                  (14)
with 1 as an all-one vector of size K and we assume eK as the special [mask] state, also defined as
the absorbing state in discrete diffusion m. Each entry in [Qt ]ij denotes the probability of transition
from the state ei to ej , and thus the previously defined Qt means with probability 1 − βt , xt will
stay unchanged and otherwise it will jump to the mask state eK .
Starting from x0 , the t-step marginal distribution and the posterior at previous time t − 1 is respec-
tively
                                        ⊤                       q(xt |xt−1 , x0 )q(xt−1 |x0 )
         q(xt |x0 ) = Cat(xt ; p = Qt x0 ); q(xt−1 |xt , x0 ) =                               (15)
                                                                         q(xt |x0 )

## Qt                                              Qt
where cumulative products Qt = i=1 Qi = αt I + (1 − αt )1m⊤ , and αt = i=1 (1 − βt ). We
expect αT approaches 0 such that the full noise data xT is equal to eK with probability 1. In the
following sections, we primarily takes the discrete diffusion formulation.

A.2   L OSS D ERIVATION

Previous discrete time t ∈ [0, T ] restricts xt to fixed time points whereas the continuous-time sam-
pling allows for more flexibility covering any point in the range (Kingma et al., 2021; Shi et al.,


                                                      19

## Similarly, after simplification,
αt           αt
                                   q(xt |xs ) =       xs + (1 −    )m.                             (17)
                                                   αs           αs
Following Zheng et al. (2024a); Shi et al. (2024) and extend the formulation to continuous time, we
have the backward transition probability:
                                         ( 1·(1−α )                 αs −αt
                   q(xt |xs )q(xs |x0 )      1−α
                                                   s
                                                      = 1−α
                                                         1−αt = 1 − 1−αt
                                                              s
                                                                            if xt = xs = m,
 q(xs |xt , x0 ) =                      = (1−  αtt
                                                   )·αs       −α
                                                                                               (18)
                       q(xt |x0 )              α s
                                                        = α s   t
                                                                            if xt = m ̸= xs .
                                                   1−αt        1−αt

For xt ̸= m, the q(xs |xt , x0 ) will stick to the observed data. For xt = m, we get the simplified
                                                   αs − αt      1 − αs
                               q(xs |xt , x0 ) =           x0 +        m.                          (19)
                                                   1 − αt       1 − αt
In diffusion process, the generative model aims to approximate the reverse transitions using a de-
noising model pθ (xs |xt , fθ (xt )) ⇝ q(xs |xt , x0 ), where fθ (xt ) represents the probability vector
obtained from the softmax applied to the logits generated by the neural network, usually using trans-
former networks (Vaswani et al., 2017) in text domain. We can similarly have
                                                αs − αt            1 − αs
                             pθ (xs |xt ) =             fθ (xt ) +        m.                       (20)
                                                1 − αt             1 − αt
Given Eq.19 and Eq.20, the KL-divergence loss is optimized by
                                               (
                                                 αs −αt
                                                        DKL (x0 ||fθ (xt )), for xt = m;
         DKL (q(xs |xt , x0 )||pθ (xs ||xt )) = 1−αt                                               (21)
                                                 0,                          for xt ̸= m.

We can use the indicator function δxt ,m to unify the conditional cases. In addition, given x0 , we
have DKL (x0 ||fθ (xt )) = −x⊤  0 log fθ (xt ) which corresponds to the cross-entropy widely used in
the classification. Therefore, we have
                                                            αs − αt
                 DKL (q(xs |xt , x0 )||pθ (xs ||xt )) = −           δx ,m x⊤
                                                                           0 log fθ (xt ).         (22)
                                                            1 − αt t
Following Eq.12, if we set a small timestep ∆t = t − s = T1 ∈ (0, 1),
                               T
                               X              αs − αt
                        LT =         [−                   δx ,m x⊤
                                                                 0 log fθ (xt )∆t ].               (23)
                               t=2
                                          (t − s)(1 − αt ) t
                                               −αs
By taking the limit as T → ∞, we have αt′ = αtt−s    , and the sum is transformed into an integral:

## Z 1
αt′
                      lim LT =             Eq(xt |x0 ) [δxt ,m x⊤
                                                                0 log fθ (xt )] dt.             (24)
                     T →∞       0 1 − αt

Also, the first two terms in Eq.11 are → 0 and a constant, respectively. Thus we can formulate the
evidence lower bound (ELBO) of − log pθ (x0 ) as Eq.24.
The same form of ELBO which is invariant to noise schedule but related to the signal-to-noise ratio
(SNR) is also introduced in Kingma et al. (2021); Shi et al. (2024). Following Austin et al. (2021);
                                                                      −α′
Zheng et al. (2024a), we choose the noise schedule αt = 1 − t, then 1−αtt = 1t .


                                                      20

## L1:N
t   = Eq(xt |x0 ) −             δxnt ,m (xn0 )⊤ log fθ (x1:N
                                                                               t   )n ,           (25)
                             t                   n=1

where fθ (x1:N
             t  )n denotes the whole input sequence is fed into the transformer model and the n-th
output token is indexed. During training, we sample t for each data point to optimize the expectation
in L1:N
    t    instead of the integral LT , while for evaluation, we use integral LT .

## B       I MPLEMENTATION D ETAILS
B.1     T RAINING DATA

DiffuGPT Previous diffusion language models such as Plaid 1B (Gulrajani & Hashimoto, 2023),
SEDD (Lou et al., 2024) and MD4 (Shi et al., 2024) use OpenWebText (Gokaslan & Cohen, 2019) to
pre-train from scratch, referring to GPT2 (Radford et al., 2019). We choose the advanced FineWeb2
corpus (Penedo et al., 2024), which is also derived from Common Crawl. We randomly sample 30
billion tokens from subset sample-100BT.

DiffuLLaMA Following Zhang et al. (2024b)3 we construct the training data for DiffuLLaMA
by mixing SlimPajama (Soboleva et al., 2023) and Starcoder data (Li et al., 2023a). We randomly
sample 65 billion tokens in the ratio of 7:3 from SlimPajama and Starcoder, respectively. We use
sequence packing and pre-tokenize the dataset for efficient computing.

Data Selection Consideration Our dataset selection aims to align with their respective pre-
training objectives. Since we train on a relatively smaller number of tokens and do not intend to
introduce new capabilities, we prioritize maintaining continuity with the models’ pre-training dis-
tributions to minimize distributional shift. For GPT2-based DiffuGPT, we use the FineWeb dataset,
which closely resembles OpenWebText (the dataset used in GPT2’s pre-training). Considering that
SEDD (Lou et al., 2024) iterates OpenWebText for more than 1 epoch and the total training amount
is around 200B tokens, we also iteratively train DiffuGPT on 30B FineWeb data to more than 100B
training tokens. In contrast, LLaMA2 is pre-trained over 1T tokens within one epoch on web and
code data. To align with this, we follow TinyLLaMA (Zhang et al., 2024a) and use a mixture of
SlimPajama and Starcoder data, designed to reflect LLaMA2’s pre-training data.

B.2     M ODEL OPTIMIZATION AND HYPERPARAMETERS

DiffuGPT We implement DiffuGPT using LLaMA-Factory4 with DeepSpeed Zero-2 paralleliza-
tion (Rajbhandari et al., 2020). The hyperparameter setting compared with previous work is listed in
Table 4. The global batch size is calculated by multiplying the single GPU batch size, the number of
gradient accumulation steps, and the number of GPUs, where we use 8 A100 80G. We use learning
rate of 3e − 4 with cosine scheduler. The warm up steps are set to 2K and attention mask annealing
steps are 10K. As shown in Table 4, our effective training tokens are less or equal than SEDD and
MD4, while DiffuGPT exhibits better performance according to Table 1 and Figure 3.

DiffuLLaMA For a more efficient pre-training, we implement DiffuLLaMA using huggingface5 .
We use DeepSpeed Zero-3 parallelization with CPU offloading (Rajbhandari et al., 2020) to effi-
ciently scale DiffuLLaMA to multiple GPUs and nodes. Furthermore, we use flash-attention 2 and
fused cross-entropy loss for optimized GPU memory usage and compute time (Dao, 2024). With
these settings, we get set a batch size of 60 per GPU with context length 2048 on a GH200 96GB
GPU. We use AdamW (Loshchilov & Hutter, 2019) to optimize our models with a constant learning
rate of 2e − 5 and accumulate gradients every 4 steps. We train our model for 65 billion tokens on
16 4xGH200 nodes.
    2
      https://huggingface.co/datasets/HuggingFaceFW/fineweb
    3
      https://github.com/jzhang38/TinyLlama
    4
      https://github.com/hiyouga/LLaMA-Factory
    5
      https://github.com/huggingface/transformers


                                                 21

## C     A DDITIONAL R ESULTS
C.1   U NCONDITIONAL GENERATION

The generation quality is different for different hyperparameters, shown in Figure 5. Lowering the
temperature increases fluency but reduces diversity, leading to noticeable repetition in sentences.
                                                  200                                                                                              1.0

## T1024-ppl                     T1024-dist2
50                                                                                              0.4

                                                    0                                                                                              0.2
                                                          plin
                                                              g                  1.0              k0.9
                                                                                                      8                  1.0                 0.8
                                                        am               -t   opk              top               -t   opk            -t   opk
                             au                    lt s             p0.9               p  0.9-              p0.8                p0.9
                          Def                                    top                top                  top                top

Figure 5: The unconditional generation quality for different diffusion time steps T and sampling
algorithms. We annotate the temperature of top-k sampling and top-p sampling.

We randomly selected samples generated by our DiffuGPT-M models with 1024 tokens for various
T , as shown in Table 12, Table 13, and Table 14. Lower T values result in less fluent text.

C.2   A BLATION ON GSM8K- SYMBOLIC

Using the same discrete diffusion loss, if we direct finetune on GPT2 achieves accuracy of 45.4
and 49.7 for small and medium models, respectively. In contrast, Finetuning from DiffuGPT yields
accuracy of 50.2 and 61.8 (Table 1). Comparing with GPT2, DiffuGPT, as the base model, converges
faster and attains a lower loss, as shown in Figure 6. This indicates that a better base model leads
to improved results and also demonstrates the superiority of DiffuGPT as the current best diffusion
base model.
Training with the initial weightings of GPT2, we evaluate three loss functions: DD, DD (no shift),
and DD (no annealing) when finetuning on GSM8K-symbolic data. The corresponding loss and
accuracy are shown in Table 5. Additionally, results for DiffuGPT and DiffuLLaMa are presented.
All results highlight the negative correlation between the loss and accuracy.

      Table 5: The training loss (ELBO) and test accuracy on the GSM8K-symbolic dataset.

                        Models                                                                                   Loss (ELBO)                       Acc
                        GPT2-M + DD                                                                              0.015                             49.7
                        GPT2-M + DD (no shift)                                                                   0.028                             34.5
                        GPT2-M + DD (no anneal)                                                                  0.019                             47.2
                        DiffuGPT                                                                                 0.009                             61.8
                        DiffuLLaMA                                                                               0.003                             63.1


C.3   A DVANTAGES OF DLM S

To explore the self-correction advantages of DLMs noted by Ye et al. (2024b), we perform a quali-
tative analysis and find a similar self-correction capability in DiffuGPT. Our observation of the final
steps of sampling trajectories, as shown in Table 6, indicates that DLMs refine intermediate numbers
without adhering to a left-to-right constraint.
For global planning, we follow Ye et al. (2024a) to finetune DLMs on counting down (CD) datasets.
CD is a mathematical reasoning challenge and a generalized version of the game 24, which many
AR models struggle with (Gandhi et al., 2024). We compare DiffuGPT with other AR baselines
with different model sizes in Table 7, demonstrating the advantages of DLMs.


                                                                                                     23

## Training Steps            Training Steps
Figure 6: Finetune GSM8K data with discrete diffusion objectives, using a base model of either
GPT2-S/M or DiffuGPT-S/M. DiffuGPT converges faster and attains a lower loss.


Table 6: A Case study to show self-correction capacity of DiffuGPT. t/T refers to the current
decoding step over the total diffusion steps. The incorrect rationales are marked in red.

     Steps (t/T )                                                DoT rationales
          ...                                  ...
         9/32           <<3*15=45>> <<4*45=180>> <<180+300=00>> #### 00
         8/32       <<3*15=45>> <<4*45=180>> <<180+400=580000 #### #### 000
         7/32          <<3*15=45>> <<4*45=180>> <<180+400=400>> #### 480
         6/32          <<3*15=45>> <<4*45=180>> <<180+300=500>> #### 580
         5/32          <<3*15=45>> <<4*45=180>> <<180+300=580>> #### 580
         4/32          <<3*15=45>> <<4*45=180>> <<180+300=480>> #### 580
         3/32          <<3*15=45>> <<4*45=180>> <<180+300=480>> #### 580
         2/32          <<3*15=45>> <<4*45=180>> <<180+300=480>> #### 480
         1/32          <<3*15=45>> <<4*45=180>> <<180+300=480>> #### 480




For infilling tasks, we attempt to query the LLaMA model with the prompt given the
<prefix> and <suffix>, please answer the <middle> part, which includes
both prefix and suffix information. However, this approach is no better than simply completing
the prefix, likely because the LLaMA model needs tuning for filling in the middle (FIM; Bavarian
et al. 2022b). Additionally, Bavarian et al. (2022b) notes that using AR models for infilling presents
challenges, such as prompting difficulties and repetition. In contrast, DLMs are naturally suited for
this task, as they are trained to handle masked inputs, which is a key advantage.
Additionally, we conduct a controlled experiment by training both AR and DLMs on 100M tokens
from the Starcoder dataset, using CodeLLaMA as the base model and evaluating performance on
HumanEval infilling. We finetune CodeLLaMA autoregressively with FIM in both suffix-prefix-
middle (SPM) and prefix-suffix-middle (PSM) formats. Our results in Table 8 show that Diffu-
CodeLLaMA outperforms PSM, suggesting that prompt format affects AR models but not DLMs.
We believe that training on more than 100M tokens, which is relatively small, could enhance perfor-
mance.


                    Table 7: The finetuning results (accuracy) on the CD4 dataset.

## Models                          Size      CD4
GPT2-scratch                    85M        45.8
                                          LLaMA FT                        13B        51.1
                                          SoS (Gandhi et al., 2024)       250M       54.2
                                          DiffuGPT                        355M       87.5


                                                                24

## Models               HSwag     Wino     SIQA    PIQA [MODEL-GENERATED SAMPLE OUTPUT, NOT AUTHOR PROSE]
DiffuLLaMA T=32        58.7     56.4    43.2     63.3
                       DiffuLLaMA T=8         47.1     52.6    41.9     57.1




                   Table 12: Generation examples of DiffuGPT-M (T = 1024).


   If you’re considering applying to the JBCC school, I’m confident you will find something
   there!
   What exactly do the schools have required?
   I HAVE DREAMED (I know it’s not in my DNA). In fact, I have my life time.
   What do you need anyway?
   - MA or PhD degree in non-religious education and/or MA or PhD degree.
   - Our Schools ensure that you have a strong academic background in the non-religious field
   of your choice and a passion for teaching in literature, research, writing, reading, English as
   a second language, or teaching.
   - The concentration may help to provide a world-class emphasis or offer unique opportunities
   for teaching of new skills or in dynamic contexts.
   - Practical Program coursework enables students to study in specific fields and areas in their
   career, this may be an essential part of education if you work as a tutor or if your goal will
   include working in an office, as a high school social science teacher, or a classroom.
   What makes JBCC unique?
   The National International Baccala Practical Programs are located locally and offer JBCC’s
   scope and the preparation to complete an advanced master’s degree.
   How do students go to graduate from JBCC?
   Get in now!
   Congratulations for applying for the JBCC program. Become a part of this community.
   What does JBCC achieve?
   JBCC prepares leaders to be a catalyst for an educational and emotional enriching environ-
   ment for service. Leaders are within their capacity to call others to service. Want to know
   more?
   To find additional information, please use this form.
   You can reach us to find your application materials for JBCC here.

## Educator and Director [MODEL-GENERATED SAMPLE OUTPUT, NOT AUTHOR PROSE]
School of Christian Education Hello there everyone! Welcome to our tutorials section. We
   have everything you need to know what and how to make quality t-shirts apparel especially
   for those who are new to t-shirts. If you are not sure, they are a very popular item.
   As you might imagine, there are a wider audience of people than others.
   - T- t shirt – how to make yourself?
   - T-shirt to wear?
   - What can I use to make my t shirt?
   ...
   4. Do I need to reuse my tee shirt?
   There is no need to reuse your tee shirt. This means you can spend a few minutes putting it
   on another pair of clothing. Polyester is great for quick clean, allowing you to get out the
   stains on the front of your tee shirt, without damaging them.
   Since polyester is very breathable, which means it is able to be used on both your skin and
   anything else




                                                 26

## Google Digital Advertising Cookies [MODEL-GENERATED SAMPLE OUTPUT, NOT AUTHOR PROSE]
Advertising partners may use internet analytics with information on your websites visit, how
   far you come, through our different advertising networks in order to tailor them for you. This
   may provide your personal information to our advertising partners and may be shared with
   third-party advertisers.
   Google may also use tracking cookies that analyse visitor preferences, such as the type of
   device used, visits to pages the user has access to most frequently, or how frequently visitors
   visit particular pages, to analyse how people find what sort of information most interest to
   them. They can keep track of all such visits as they collect more information, using cookies
   and analytics to improve the content of our website.
   For more information on how to control Google’s cookies on our websites, see
   https://geo.com/sies to learn moreLogan has been very popular among travelers in the area
   since 1845. The early explorers had a place and that’s the best place to stay after a long jour-
   ney. Fast-forward the years, and the ones that stayed in Logan migrated before even crossing
   over into the mountains to the north. These changes have made it from a midwestern city to
   a postmodern city of sorts where everyone lives.
   This is what global locations mean. It’s easy enough to get lost after a day spent traveling,
   but every city offers different experiences and fun when it comes to being on the road.
   The most important parts of Logan experience are the historic districts of Logan Square.
   Discover the history of so many different places and gather with people who enjoy exploring
   something new, or even if with children or pets in tow. Discover the Broadway complex of
   Logan Square and the rest of the neighborhood, as it’s a charm sprinkled with charm.
   Sitting the North of Telegraph Hill one mile from Loomers you will find the Logan mountain
   trail. The 18-mile-long trail is one of the finest in the state and enjoy a day on your bike or
   biking, surrounded by all sorts of cafes, unique eateries and art galleries.
   Lawsony Park has the Home Park Beer Cellar and the Home Park Bike Shop, and a great
   gift shop for that. Explore the neighborhood on Memorial weekend and warm up with a
   delicious bite from Logan Friendly Brews.
   A week on Memorial Weekend is a new tradition for the area of Loomers. Here’s the excite-
   ment of heading on the road on one of the weekends of the year!
   The summer at Logan is one of the most popular times of year to explore the quirky neigh-
   borhoods. Our parks are vibrant and busy. We are downtown and our restaurant and bars
   stay open to have a drink and coffee. Stop in at the rooftop patio to enjoy a really nice
   evening with a quick bite. And every week on Memorial Weekend, you’ll find a really nice
   indoor pool!
   For kids with a weekend, Logan has a playground and good for two and on August 4 and 8
   is the Logan Fun Day. A few spots are available (and we’re always open to both men and
   women) and your group allows you to just catch up and season with friends while enjoying
   a pint of beer.
   John Moody Brews is about to bring their Labor Day fun to the streets and your backyard
   on Labor Day Weekend. Logan Friendly Brews has a brewery cruise, street vendors and live
   music on Thursday night and they’ll have giveaways on Sunday. It’s a fun




                                                 27

## There no factors that could be [MODEL-GENERATED SAMPLE OUTPUT, NOT AUTHOR PROSE]
MeanESK 2 is a class II agent with a teratogenic activity as determined by the MSL. We
   propose not treatment of ORIK on patients that exceed the 2.0 threshold.
   Both biological factors predispose risk of the.
   ORISA-WEVINB8: X at leastISMRC IX.
   Non-Mouse type 8a(d12).
   About both factors that promote the development of ORIK.
   To specify a reasonable regulation that would affect a p.ss.The United States Juvenile Court
   as an Authority for Administration of the Bureau of Corrections by Arthur Whyte, Jr. Chair:
   John F. Bronz Board: Buck Morgan Jr. Rep: John Little Reader: David Gervis.
   THURY Fisher, WOOD and her son.

## REV: July 3, 1949 [MODEL-GENERATED SAMPLE OUTPUT, NOT AUTHOR PROSE]
By resolution which is passed: Either a person uses nothing more than a boat within a
   warehouse, apartment or barn. Department of the Interior or Department of Labor or system
   of it existed as a whole under the laws of Michigan three. through December one of such it
   would not. Each $100 person shall be fined and the division shall collect all damages of each
   two hundred dollars and dollars’ the total of such amount and enter a. Rights reserved: also a
   trustee thereof may order the delinquent bonds. : If a person convicted if that has committed
   any act of attachment. Assisting duties shall deemed to have ceased within fifteen days, then
   the Commissioner to be appointed shall pay the district the sum of four percent of the value
   of the money recovered from these costs. Pursuant in this section shall be revoked without
   notice. Revocation: also the equivalent of instructions accompanied with. Fifteen days of
   the entry of such order. : The blight or place known to be in a person has an occupancy.k
   enables.
   He can throw in an about picture. How he loves it so much he is the greater these online webit
   himself right from scuba dive and wishes to be removed by the government of America,
   that’s a much sleazier story. From time social circles, the members of MoolahFast, a dating
   site focusing on the planet of different scuba divers,




                                                28
