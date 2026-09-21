# Citation Agent

You are the Citation Agent in the multi-agent limitation-generation framework of
arXiv:2601.11578 (Al Azher, Guo, Alhoori). You operate on one input paper.

## TASK
Identify and articulate limitations for the current paper, focusing on shortcomings that align with findings in the related works supplied below. These works are drawn from the paper's reference pool and from the surrounding literature. Use them to place the input paper in context: where does a related work test something this paper assumed, supersede a choice this paper made, or report a result that undercuts one of its claims? Every limitation must cite the specific retrieved work that grounds it.

## OUTPUT CONTRACT

Return JSON only:
{"limitations": [{"id": "<AGENT>-1",
                  "statement": "<one specific limitation, 2-4 sentences>",
                  "evidence": "<quote or section reference from the INPUT>",
                  "provenance": "Cited"}]}

Rules:
- Be specific to THIS paper. Reject generic statements ("limited generalizability",
  "dataset bias") unless the paper's own text makes them concrete.
- Every statement must be traceable to the INPUT provided. Do not invent numbers.
- Produce at most 8 limitations. Fewer, sharper items beat more, vaguer ones.


## INPUT
### INPUT PAPER
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
 Tworek, and Mark Chen. Efficient training of language models to fill in t

### RETRIEVED RELATED WORKS (RAG, re-ranked >= 8/10)
[by::40] (cited_by_substitute, 2026, rerank 10/10) Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment
Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment. Diffusion language models (DLMs) have recently demonstrated capabilities that complement standard autoregressive (AR) models, particularly in non-sequential generation and bidirectional editing. Although recent work has shown that pretrained autoregressive checkpoints can be converted into diffusion language models, existing recipes primarily transfer parameters through continued denoising training with objective- and attention-level modifications. We instead ask whether the internal representation geometry learned by next-token prediction can be explicitly preserved during AR-to-DLM conversion. We hypothesize that much of the semantic structure learned by AR pretraining can transfer across generation orders, and thus DLM training should be viewed as relearning the decoding path rather than relearning language representations. To investigate this, we introduce REPR-ALIGN, a representation alignment objective that adapts a bidirectional masked diffusion model to reuse representations from a pretrained AR model of identical architecture. Concretely, we align the hidden states of the DLM t

[by::9] (cited_by_substitute, 2026, rerank 10/10) PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention
PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention. Discrete masked diffusion language models support bidirectional generation and infilling, but adapting pretrained autoregressive (AR) transformers requires reconciling causal pretraining with bidirectional denoising. We study this problem at the level of attention rather than claiming AR-weight reuse itself as novel. PreDiff-LM preserves causal attention within the observed prompt while allowing full bidirectional attention within the masked target. Under a matched GPT-2 Medium, WikiText-103, 90K-step setup, this hybrid mask improves unconditional perplexity from 34.1 to 28.7 and MAUVE from 0.71 to 0.78 over uniform bidirectional attention with the same AR initialization. Attention adaptation also composes with a DiffuGPT-style objective adaptation, reaching 26.9 perplexity. Pretrained initialization reduces the steps required to reach perplexity below 50 from about 350K to 8K, although a compute-matched fine-tuned AR model remains stronger at equal scale (18.9 versus 28.7). Beyond perplexity, PreDiff-LM improves repetition, distributional quality, four zero-shot downstream tasks, and human pr

[by::43] (cited_by_substitute, 2026, rerank 9/10) UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective
UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective. Existing methods mainly adapt pretrained autoregressive (AR) language models to masked diffusion, whereas we directly adapt them to uniform-noise diffusion, where every token remains editable during sampling. However, adapting AR checkpoints across corruption kernels remains challenging because existing DLMs use different objectives and prediction parameterizations. We establish connections among SEDD, MDLM/GIDD, M2S, and Neural CTMC by expressing their conditional losses as a single generalized Kullback--Leibler objective over model reverse rates. We further derive conversions from clean-token predictions to concrete-score, posterior-mean, and exit-rate/jump parameterizations, yielding a shared \(x_0\) interface that supports switching between mask and uniform kernels. Building on these connections, we propose \ours{}, a simple continual pre-training approach for directly adapting pretrained GPT2 checkpoints to uniform-noise diffusion. Through systematic evaluation of 124M- and 355M-parameter models, we show that \ours{} steadily improves the trade-off between generat

[in::13] (cited_in, 2025, rerank 9/10) TESS 2: A Large-Scale Generalist Diffusion Language Model
TESS 2: A Large-Scale Generalist Diffusion Language Model. We introduce TESS 2, a general instruction-following diffusion language model that outperforms contemporary instruction-tuned diffusion models, as well as matches and sometimes exceeds strong autoregressive (AR) models. We train TESS 2 by first adapting a strong AR model via continued pretraining with the usual cross-entropy as diffusion loss, and then performing further instruction tuning. We find that adaptation training as well as the choice of the base model is crucial for training good instruction-following diffusion models. We further propose reward guidance, a novel and modular inference-time guidance procedure to align model outputs without needing to train the underlying model. Finally, we show that TESS 2 further improves with increased inference-time compute, highlighting the utility of diffusion LMs in having fine-grained controllability over the amount of compute used at inference time. Code and models are available at https://github.com/hamishivi/tess-2.

[by::0] (cited_by_substitute, 2025, rerank 9/10) TESS 2: A Large-Scale Generalist Diffusion Language Model
TESS 2: A Large-Scale Generalist Diffusion Language Model. We introduce TESS 2, a general instructionfollowing diffusion language model that outperforms contemporary instruction-tuned diffusion models, as well as matches and sometimes exceeds strong autoregressive (AR) models.We train TESS 2 by first adapting an AR model via continued pretraining with the usual cross-entropy as diffusion loss, and then performing further instruction tuning.We find that adaptation training as well as the choice of the base model is crucial for training good instruction-following diffusion models.Furthermore, we propose reward guidance, a novel and modular inference-time guidance procedure to align model outputs without needing to train the underlying model.Finally, we show that TESS 2 further improves with increased inference-time compute, highlighting the utility of diffusion LMs in having fine-grained controllability over the amount of compute used at inference time.Code and models are available at https://github.com/hamishivi/tess-2.

[by::5] (cited_by_substitute, 2025, rerank 9/10) Fast and Accurate Causal Parallel Decoding using Jacobi Forcing
Fast and Accurate Causal Parallel Decoding using Jacobi Forcing. Multi-token generation has emerged as a promising paradigm for accelerating transformer-based large model inference. Recent efforts primarily explore diffusion Large Language Models (dLLMs) for parallel decoding to reduce inference latency. To achieve AR-level generation quality, many techniques adapt AR models into dLLMs to enable parallel decoding. However, they suffer from limited speedup compared to AR models due to a pretrain-to-posttrain mismatch. Specifically, the masked data distribution in post-training deviates significantly from the real-world data distribution seen during pretraining, and dLLMs rely on bidirectional attention, which conflicts with the causal prior learned during pretraining and hinders the integration of exact KV cache reuse. To address this, we introduce Jacobi Forcing, a progressive distillation paradigm where models are trained on their own generated parallel decoding trajectories, smoothly shifting AR models into efficient parallel decoders while preserving their pretrained causal inference property. The models trained under this paradigm, Jacobi Forcing Model, achieves 3.8x wall-clock

[in::12] (cited_in, 2025, rerank 9/10) Flexible-length Text Infilling for Discrete Diffusion Models
Flexible-length Text Infilling for Discrete Diffusion Models. Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability previously missing in text diffusion. Our method is orthogonal to existing discrete text diffusion methods and is compatible with various pretrained text denoisers. Extensive experiments on text infilling benchmarks such as One-Billion-Word and Yelp demonstrate that DDOT outperforms naive diffusion 

[by::3] (cited_by_substitute, 2025, rerank 8/10) Enabling Autoregressive Models to Fill In Masked Tokens
Enabling Autoregressive Models to Fill In Masked Tokens. Historically, LLMs have been trained using either autoregressive (AR) or masked language modeling (MLM) objectives, with AR models gaining dominance in recent years. However, AR models are inherently incapable of masked infilling, which is the ability to predict masked tokens between past and future context. In contrast, MLM models suffer from intrinsic computational inefficiencies during both training and inference that hinder their scalability. This work introduces MARIA (Masked and Autoregressive Infilling Architecture), a novel approach that leverages the strengths of both paradigms to achieve state-of-the-art masked infilling performance. MARIA combines a pre-trained MLM and AR model by training a linear decoder that takes their concatenated hidden states as input. This minimal modification enables the AR model to perform infilling while retaining its inherent advantages in terms of faster inference with KV caching. Our results demonstrate that MARIA significantly outperforms existing methods, namely discrete diffusion models, on masked infilling tasks.

[by::37] (cited_by_substitute, 2025, rerank 8/10) Dream 7B: Diffusion Large Language Models
Dream 7B: Diffusion Large Language Models. We introduce Dream 7B, the most powerful open diffusion large language model to date. Unlike autoregressive (AR) models that generate tokens sequentially, Dream 7B employs discrete diffusion modeling to refine sequences in parallel through iterative denoising. Our model consistently outperforms existing diffusion language models on general, mathematical, and coding tasks. Dream 7B demonstrates superior planning abilities and inference flexibility, including arbitrary-order generation, infilling capabilities, and tunable quality-speed trade-offs. These results are achieved through simple yet effective training techniques, including AR-based LLM initialization and context-adaptive token-level noise rescheduling. We release both Dream-Base and Dream-Instruct to facilitate further research in diffusion-based language modeling.

[in::29] (cited_in, 2024, rerank 8/10) Simple and Effective Masked Diffusion Language Models
Simple and Effective Masked Diffusion Language Models. While diffusion models excel at generating high-quality images, prior work reports a significant performance gap between diffusion and autoregressive (AR) methods in language modeling. In this work, we show that simple masked discrete diffusion is more performant than previously thought. We apply an effective training recipe that improves the performance of masked diffusion models and derive a simplified, Rao-Blackwellized objective that results in additional improvements. Our objective has a simple form -- it is a mixture of classical masked language modeling losses -- and can be used to train encoder-only language models that admit efficient samplers, including ones that can generate arbitrary lengths of text semi-autoregressively like a traditional language model. On language modeling benchmarks, a range of masked diffusion models trained with modern engineering practices achieves a new state-of-the-art among diffusion models, and approaches AR perplexity. We provide the code, along with a blog post and video tutorial on the project page: https://s-sahoo.com/mdlm

[in::35] (cited_in, 2023, rerank 8/10) Continual Pre-Training of Large Language Models: How to (re)warm your model?
Continual Pre-Training of Large Language Models: How to (re)warm your model?. Large language models (LLMs) are routinely pre-trained on billions of tokens, only to restart the process over again once new data becomes available. A much cheaper and more efficient solution would be to enable the continual pre-training of these models, i.e. updating pre-trained models with new data instead of re-training them from scratch. However, the distribution shift induced by novel data typically results in degraded performance on past data. Taking a step towards efficient continual pre-training, in this work, we examine the effect of different warm-up strategies. Our hypothesis is that the learning rate must be re-increased to improve compute efficiency when training on a new dataset. We study the warmup phase of models pre-trained on the Pile (upstream data, 300B tokens) as we continue to pre-train on SlimPajama (downstream data, 297B tokens), following a linear warmup and cosine decay schedule. We conduct all experiments on the Pythia 410M language model architecture and evaluate performance through validation perplexity. We experiment with different pre-training checkpoints, various maximum l

[by::10] (cited_by_substitute, 2026, rerank 8/10) dLLM: Simple Diffusion Language Modeling
dLLM: Simple Diffusion Language Modeling. Although diffusion language models (DLMs) are evolving quickly, many recent models converge on a set of shared components. These components, however, are distributed across ad-hoc research codebases or lack transparent implementations, making them difficult to reproduce or extend. As the field accelerates, there is a clear need for a unified framework that standardizes these common components while remaining flexible enough to support new methods and architectures. To address this gap, we introduce dLLM, an open-source framework that unifies the core components of diffusion language modeling -- training, inference, and evaluation -- and makes them easy to customize for new designs. With dLLM, users can reproduce, finetune, deploy, and evaluate open-source large DLMs such as LLaDA and Dream through a standardized pipeline. The framework also provides minimal, reproducible recipes for building small DLMs from scratch with accessible compute, including converting any BERT-style encoder or autoregressive LM into a DLM. We also release the checkpoints of these small DLMs to make DLMs more accessible and accelerate future research.

[in::8] (cited_in, 2025, rerank 8/10) Dream-Coder 7B: An Open Diffusion Language Model for Code
Dream-Coder 7B: An Open Diffusion Language Model for Code. We present Dream-Coder 7B, an open-source discrete diffusion language model for code generation that exhibits emergent any-order generation capabilities. Unlike traditional autoregressive (AR) models that decode strictly left-to-right, Dream-Coder 7B adaptively determines its decoding strategy based on the coding task: sketch-first generation for complex algorithms, left-to-right generation for straightforward completions, and interleaved reasoning generation for code understanding tasks. We adapt a pretrained AR checkpoint to a discrete diffusion frameworks with a continuous-time weighted cross-entropy objective. Our post-training recipe comprises (i) supervised fine-tuning, where we mitigate padding pathologies via random truncation and a padding penalty to improve sample efficiency and stabilize generation; and (ii) reinforcement learning with verifiable rewards over a curated high-quality prompt set drawn from open-source datasets, using a tailored reinforcement learning recipe for diffusion language models. The resulting Dream-Coder 7B Instruct attains 21.4\% pass@1 on LiveCodeBench (2410--2505) and demonstrates compet
