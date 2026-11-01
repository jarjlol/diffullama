# Model-Agnostic Structure-Guided Repair for Diffusion Language Models

## Background

Diffusion language models generate text by starting with `[MASK]` tokens and gradually filling them over several refinement steps. Unlike ordinary autoregressive models, which generate strictly from left to right, masked diffusion models can predict and revise multiple positions using context from both directions.

DiffuLLaMA showed that an existing autoregressive LLaMA model can be converted into a diffusion language model. It is our anchor paper, but our proposed method will **not** depend on DiffuLLaMA specifically.

**Anchor paper:** *Scaling Diffusion Language Models via Adaptation from Autoregressive Models*, ICLR 2025  
**Paper:** https://proceedings.iclr.cc/paper_files/paper/2025/file/0fa81c3f0d57f95b8776de3a248ef0ed-Paper-Conference.pdf  
**Code:** https://github.com/HKUNLP/DiffuLLaMA

Other open masked diffusion models include:

- **LLaDA-8B:** https://github.com/ML-GSAI/LLaDA
- **Dream-7B / Dream-Coder:** https://github.com/DreamLM/Dream

All these models expose the same basic capability needed by our method: accept partially masked text, predict the missing tokens, and repeat the denoising process.

## General problem

Masked diffusion models can revise generated tokens, but they still need a rule for deciding **which tokens should be masked and regenerated together**.

Existing approaches commonly select tokens using:

- low confidence,
- changes in confidence over time,
- random selection,
- or a fixed local window around a suspected error.

These rules mostly use token probability or physical distance. They do not explicitly use the structural relationships inside the generated output. Two tokens can be far apart in the sequence but strongly connected in meaning or function.

## Our general idea: Structure-Guided ReMasking

**Working title:** *Structure-Guided ReMasking for Masked Diffusion Language Models*

Our method is a model-agnostic inference module. It can be placed on top of any masked diffusion language model that provides token logits and supports partial masking.

In simple words:

1. A diffusion language model generates an output.
2. A verifier checks the output and identifies a failure or suspicious region.
3. A structural analyzer constructs relationships between parts of the output.
4. Starting from the suspicious region, the method finds all structurally related parts, even if they occur far away.
5. It replaces this non-contiguous set of tokens with `[MASK]`.
6. The diffusion model regenerates the connected pieces together.
7. The verifier checks whether the new output is better.

```text
Generate → Verify → Build structure graph → Select related tokens
        → Re-mask → Diffusion refinement → Verify again
```

The diffusion model itself remains frozen. Only the adapter used to call the model changes between DiffuLLaMA, LLaDA, and Dream.

## First application: code repair

Code provides a clean initial application because it has explicit structure and executable verifiers.

Suppose the generated program contains:

```python
discount = 0.10  # possible cause

# many lines later
final_price = price - discount * quantity  # failure appears here
```

A local repair method may regenerate only the `final_price` line. Our method follows the program's data-flow relationships and also revisits the earlier definition of `discount`.

For code, the structure graph can contain:

- variable definition-to-use edges,
- data-flow edges,
- function-call and argument edges,
- control-flow dependencies,
- and AST parent/child relationships.

Compilation errors, tracebacks, and unit-test failures provide the verification signal.

## Main research question

> Across multiple masked diffusion language models, does structure-guided non-contiguous remasking repair generated code better than probability-based or spatially local remasking under the same inference-compute budget?

## Why this is more general than the original idea

The contribution is **not a DiffuLLaMA modification**. It consists of three reusable parts:

1. **Model adapter:** standard interface for getting logits, applying masks, freezing prompt tokens, and running refinement steps.
2. **Structure selector:** converts a task-specific relationship graph into token positions that should be regenerated together.
3. **Verification loop:** evaluates the revised output and optionally repeats repair.

The core algorithm stays unchanged across models. We should evaluate at least two backbones:

- LLaDA-8B-Instruct,
- Dream-7B or Dream-Coder-7B,
- with DiffuLLaMA-7B as an optional third historical baseline.

This demonstrates that improvements come from the repair mechanism rather than one model's sampling implementation.

## What we need to build

### Common model interface

- Load different diffusion language models.
- Expose their mask-token ID and model logits.
- Freeze prompt and unaffected output positions.
- Apply an arbitrary non-contiguous mask.
- Run a controlled number of refinement steps.

### Structure-guided repair

- Parse generated Python code.
- Compile and execute visible unit tests safely.
- Localize the failing statement using errors, traces, or returned values.
- Construct an AST/data-flow graph.
- Traverse backward from the failure to obtain the relevant program slice.
- Map source-code locations to tokenizer positions.
- Remask and jointly regenerate those positions.

### Baselines

- no repair,
- random remasking,
- low-confidence remasking,
- temporal-instability remasking,
- fixed local-window remasking,
- AST-subtree remasking,
- and our full structure-guided program-slice remasking.

## Evaluation

Recommended datasets:

- HumanEval,
- MBPP.

Primary measurements:

- `pass@1`,
- percentage of failed drafts successfully repaired,
- compilation rate,
- extra model forward passes,
- wall-clock latency,
- number of remasked/changed tokens,
- and initially correct programs accidentally broken.

All methods must receive the same remasking and model-call budget. Visible tests may guide repair, but separate hidden tests must measure final correctness.

## Scope and novelty

Generic adaptive remasking is already an active research area. Recent methods select tokens using confidence or instability and already work across models such as LLaDA and Dream. Therefore, we should **not** claim that model-agnostic remasking itself is new.

Our proposed gap is narrower:

> Existing generic remasking methods do not explicitly use an external semantic dependency graph to choose a non-contiguous repair scope.

For the code application, this becomes test-guided program-slice remasking. The method should be designed generally, evaluated across multiple diffusion backbones, and compared against strong generic remasking baselines.

If time permits, the same framework could later be tested on another structured task, such as equation-dependency repair in mathematical reasoning. This is an extension, not required for the main semester project.
