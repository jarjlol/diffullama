# Related Work: manual filtering of the literature-review references (issue #9)

Source: `litreview/data/reference_brief.txt` (48 references `[1]`-`[48]`, the numbering used by `litreview/output/generated_survey.md`). The submitted survey text and its numbering were **not** edited; the paper section re-cites by BibTeX key (`refs_related_work.bib`), so no positional marker can desync.

Criteria: keep papers that support the three groups of the section (text diffusion LMs, adaptation of AR models, non-autoregressive generation as background) or the reasoning/GSM8K setting. Remove other modalities and tasks, narrow technique papers, and overlaps. Two kinds of removal are labelled: **unrelated** (other modality or task) and **space** (related, but the paper has one reference page in total).

Result: 12 of 48 kept from the lit review, +1 added from outside it (DiffuLLaMA, the anchor paper, which a survey about the anchor does not list), +8 added for the two new groups (below). 36 removed.

## Kept (survey number -> BibTeX key)

| # | key | title |
|---|---|---|
| 1 | `li2022diffusionlm` | Diffusion-LM Improves Controllable Text Generation |
| 4 | `svete2025reasoning` | On the Reasoning Abilities of Masked Diffusion Language Models |
| 9 | `xie2025dreamcoder` | Dream-Coder 7B: An Open Diffusion Language Model for Code |
| 12 | `zheng2023reparam` | A Reparameterized Discrete Diffusion Model for Text Generation |
| 14 | `tae2025tess2` | TESS 2: A Large-Scale Generalist Diffusion Language Model |
| 21 | `berrayana2025planner` | Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning |
| 30 | `sahoo2024mdlm` | Simple and Effective Masked Diffusion Language Models |
| 31 | `han2023ssdlm` | SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control |
| 32 | `ke2023continual` | Continual Pre-training of Language Models |
| 39 | `gu2017nat` | Non-Autoregressive Neural Machine Translation |
| 40 | `he2023diffusionbert` | DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models |
| 42 | `gong2022diffuseq` | DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models |
| (new) | `gong2025diffullama` | Scaling Diffusion Language Models via Adaptation from Autoregressive Models (anchor paper, arXiv:2410.17891) |

## Removed (survey number, reason)

| # | title | reason |
|---|---|---|
| 2 | POS-Constrained Parallel Decoding for Non-autoregressive Generation | POS-constrained parallel decoding for NAR translation: narrow NAR decoding trick, not needed for the three-group story (space). |
| 3 | Diffusion-State Policy Optimization for Masked Diffusion Language Models | DiSPO: RL credit assignment for masked diffusion LMs; narrow training technique, 0 citations (space). |
| 5 | Self-Augmenting Retrieval for Diffusion Language Models | Self-augmenting retrieval for diffusion LMs: RAG application, not our setting. |
| 6 | T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajecto | T*: progressive block scaling via RL; narrow training curriculum, 0 citations. |
| 7 | Non-autoregressive Conditional Diffusion Models for Time Series Prediction | Non-autoregressive diffusion for time-series prediction: unrelated modality. |
| 8 | A Generalized Framework of Sequence Generation with Application to Undirected Sequence Mod | Generalised sequence generation for undirected models: tangential background, dropped for space. |
| 10 | Masked Diffusion Language Models with Frequency-Informed Training | Frequency-informed masked diffusion for the BabyLM challenge: data-efficiency niche, unrelated to our task. |
| 11 | PartDiffuser: Part-wise 3D Mesh Generation via Discrete Diffusion | PartDiffuser: 3D mesh generation: unrelated modality. |
| 13 | Flexible-length Text Infilling for Discrete Diffusion Models | Flexible-length infilling for discrete diffusion: infilling task, not our task. |
| 15 | Cascaded Text Generation with Markov Transformers | Cascaded generation with Markov transformers: NAR variant, dropped for space. |
| 16 | Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models | Dependency-guided parallel decoding for diffusion LMs: inference technique we do not use (space). |
| 17 | Text-Guided Molecule Generation with Diffusion Language Model | Text-guided molecule generation: unrelated domain (SMILES). |
| 18 | Muse: Text-To-Image Generation via Masked Generative Transformers | Muse: text-to-image masked generative transformers: unrelated modality. |
| 19 | TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning | TAG-DLM: text-attributed graph learning: unrelated task. |
| 20 | Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capab | Continual pre-training for Japanese: one language-specific application, dropped for space (the general continual pre-training papers are kept). |
| 22 | Diffusion-NAT: Self-Prompting Discrete Diffusion for Non-Autoregressive Text Generation | Diffusion-NAT: text-to-text discrete diffusion; overlaps the kept discrete-diffusion papers (space). |
| 23 | Glancing Transformer for Non-Autoregressive Neural Machine Translation | Glancing Transformer: NAR translation training trick. Its abstract is not in the retrieved data (citation string only), so no claim about it could be checked; dropped. |
| 25 | SmBoP: Semi-autoregressive Bottom-up Semantic Parsing | SmBoP: semantic parsing: unrelated task. |
| 26 | AutoTemplate: A Simple Recipe for Lexically Constrained Text Generation | AutoTemplate: lexically constrained generation: unrelated task. |
| 27 | Efficient Continual Pre-training for Building Domain Specific Large Language Models | Domain-specific (finance) continual pre-training: one domain application, dropped for space. |
| 28 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Mamba: new architecture, not diffusion or AR-to-diffusion adaptation; the survey's link to adaptation is not backed by a retrieved paper. |
| 33 | Non-autoregressive Transformer by Position Learning | PNAT: position modelling for NAR translation: dropped for space. |
| 34 | Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization | NAUS: NAR unsupervised summarization: unrelated task. |
| 35 | On the Learning of Non-Autoregressive Transformers | Learning of NAR Transformers (theory of multimodality): dropped for space. |
| 37 | Non-Autoregressive Coarse-to-Fine Video Captioning | NAR video captioning: unrelated modality. |
| 38 | POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-traini | POINTER: insertion-based constrained generation: unrelated task. |
| 41 | A Comparative Study on Non-Autoregressive Modelings for Speech-to-Text Generation | Comparative study of NAR speech-to-text: unrelated modality (speech). |
| 43 | Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade | Fully NAR NMT tricks: translation-only, dropped for space. |
| 44 | Non-Autoregressive Text Generation with Pre-trained Language Models | NAR generation with pre-trained LMs: dropped for space. |
| 45 | Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speec | Paraformer: NAR speech recognition: unrelated modality. |
| 46 | PIMNet: A Parallel, Iterative and Mimicking Network for Scene Text Recognition | PIMNet: scene-text recognition: unrelated modality (images). |
| 47 | Guiding Non-Autoregressive Neural Machine Translation Decoding with Reordering Information | Reordering information for NAR translation: translation-only, dropped for space. |
| 48 | A Cheaper and Better Diffusion Language Model with Soft-Masked Noise | Soft-masked diffusion LM: overlaps the kept discrete-diffusion papers (space). |
| 24 | Semi-Autoregressive Neural Machine Translation | Semi-autoregressive NMT: NAR background, cut on 2026-11-20 (space). |
| 29 | Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Pa | GENIE: continuous diffusion pre-training; overlaps Diffusion-LM and DiffuSeq (space, cut on 2026-11-20 to make room for the two new groups). |
| 36 | Continual Pre-Training of Large Language Models: How to (re)warm your model? | Re-warming the learning rate for continual pre-training: narrow, cut on 2026-11-20 (space). |

## Verification

- Title, author list and year of all 17 bib entries in `refs_related_work.bib` were fetched from the arXiv API on 2026-11-20 (script output in the issue #9 status comment); titles match the lit review's titles exactly. Venues: ICLR 2023 for `ke2023continual` is arXiv's `journal_ref`; ACL 2023 for DiffusionBERT and SSD-LM comes from their ACL Anthology DOIs in `litreview/data/candidates_topK.json`; ICLR 2025 for the anchor comes from the repo README. Other entries are cited as arXiv preprints, no venue claimed.
- Every statement about a paper in the section was checked against that paper's abstract in `litreview/data/candidates_topK.json`. Only abstracts were read, **not full texts**: the section describes what each paper proposes, not its detailed results.
- Not used as evidence: the survey's statements that the field 'converged' on a recipe, and its Mamba architecture-transfer paragraph (it names no retrieved paper).

## Added on 2026-11-20 (groups 3 and 4, requested by the team; the lit review has none)

| key | paper | where it came from |
|---|---|---|
| `lee2025hex` | HEX: Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts (arXiv:2510.05040) | in the lit review's candidate pool (`litreview/data/candidates_merged_reranked.json`) but not in the top-48 |
| `shao2025diffusethinking` | Diffuse Thinking (arXiv:2510.27469) | named in `codification/IDEA_SELECTION.md` as a near neighbour; no title match in the three pool files, found by arXiv title search |
| `wang2023selfconsistency` | Self-Consistency (arXiv:2203.11171) | the anchor's own baseline, per `codification/IDEA_SELECTION.md`; bib entry lives in `refs_methodology.bib` (#11) |
| `cobbe2021gsm8k` | Training Verifiers to Solve Math Word Problems / GSM8K (arXiv:2110.14168) | the dataset paper; entry in `refs_methodology.bib` (#11) |
| `lu2024aiscientist` | The AI Scientist (arXiv:2408.06292) | general background on research agents, new |
| `weng2025deepscientist` | DeepScientist (arXiv:2509.26603) | compared in `codification/REFERENCE_SYSTEMS.md` |
| `baek2024researchagent` | ResearchAgent (arXiv:2404.07738) | used for ideation (`ideation/output/research_problems_and_ideas.md`); entry in `refs_methodology.bib` |
| `jin2026arbor` | Arbor (arXiv:2606.11926) | used for codification; entry in `refs_methodology.bib` |

Left out of the paper for space: EvoScientist (arXiv:2603.08127), AutoSOTA (arXiv:2604.05550), both in `REFERENCE_SYSTEMS.md`; Snell et al. on test-time compute (arXiv:2408.03314).
Titles, authors and years of all added papers were fetched from the arXiv API on 2026-11-20. Claims were checked against abstracts only.

Reference count for the paper's one reference page: **21** for this section, 17 of them in this section's bib plus the four shared with #11. Cut list if needed: `han2023ssdlm`, `ke2023continual`, `gu2017nat`, `xie2025dreamcoder`.
