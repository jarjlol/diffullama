# Related Work

<!-- Owner: aaditya (issue #9). Source: litreview/output/generated_survey.md (references [n] = litreview/data/reference_brief.txt), filtered by hand: final_submission/notes/related_work_filtering.md. Citation keys in [brackets] are in final_submission/paper/refs_related_work.bib (16 entries). Claims were re-checked against the abstracts in litreview/data/candidates_topK.json. -->

We organise prior work in three groups: text diffusion language models, adaptation of existing autoregressive (AR) models, and non-autoregressive generation, the family to which diffusion language models belong.

## 1. Text diffusion language models

**Continuous diffusion.** Diffusion-LM [li2022diffusionlm] denoises Gaussian vectors into word embeddings and uses gradients on the latent variables for fine-grained control. DiffuSeq [gong2022diffuseq] extends this to sequence-to-sequence generation, and GENIE [lin2022genie] pre-trains a continuous diffusion decoder with a paragraph-denoising objective, reaching results comparable to AR models on four generation benchmarks.

**Discrete and masked diffusion.** Reparameterized discrete diffusion [zheng2023reparam] gives a more flexible formulation of discrete sampling. Simple and Effective Masked Diffusion Language Models (MDLM) [sahoo2024mdlm] train with a Rao-Blackwellized masked-diffusion objective that approaches AR perplexity. Two models start from existing language models: DiffusionBERT [he2023diffusionbert] trains BERT to reverse a discrete diffusion process, and SSD-LM [han2023ssdlm] generates text block by block (semi-autoregressively) and matches or outperforms GPT-2 on its generation benchmarks. Svete and Sabharwal [svete2025reasoning] show that masked diffusion models are equivalent to padded looped transformers and can solve every problem that chain-of-thought transformers can, which bears on the reasoning tasks we evaluate on.

## 2. Adapting autoregressive models

**Continual pre-training.** Continuing to pre-train an existing model is cheaper than training from scratch. Ke et al. [ke2023continual] adapt a language model to a sequence of domain corpora while limiting forgetting, and Gupta et al. [gupta2023rewarm] study learning-rate warm-up strategies when pre-training resumes on new data.

**AR-to-diffusion adaptation.** DiffuLLaMA and DiffuGPT [gong2025diffullama] convert GPT-2 and LLaMA models (127M to 7B parameters) into diffusion models with continual pre-training; they are among the diffusion models we evaluate. TESS 2 [tae2025tess2] reports the same recipe at instruction-following scale: adapt a strong AR model with a diffusion loss, then instruction-tune, and the authors find both the adaptation step and the choice of base model to be crucial. Dream-Coder 7B [xie2025dreamcoder] adapts a pre-trained AR checkpoint for code generation.

**Combining AR and diffusion models at inference.** Planner and Executor [berrayana2025planner] couples a discrete diffusion model with an AR model, in text space (one plans, the other answers) and through a learned latent projector, and reports gains from the latter. Our method also uses both model families but in a different role: an AR model selects among the candidate answers of a diffusion model (see Methodology).

## 3. Non-autoregressive generation

Non-autoregressive translation [gu2017nat] drops the left-to-right dependency to decode in parallel, cutting latency by an order of magnitude at a cost in BLEU. Semi-autoregressive Transformers [wang2018sat] keep the AR order across blocks and emit several tokens per step. Diffusion language models can be read as a learned, iterative form of this parallel decoding.
