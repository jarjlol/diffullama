# Provenance

| | |
|---|---|
| Questions | GSM8K (Cobbe et al., 2021), MIT licence, `openai/grade-school-math` `grade_school_math/data/{train,test}.jsonl` |
| dev | 500 items from train, `random.Random(0)` shuffle, first 500; uids `gsm8k-train-NNNN` (index into train) |
| test | all 1319 test items; uids `gsm8k-test-NNNN` |
| few-shot | next 4 shuffled train items (disjoint from dev); calculator annotations `<<…>>` removed |
| gold | number after `####`, normalised by `task.normalize` |
| candidates | `../pipeline/generate_candidates.py`, settings in `../pipeline/config.json` (N = 10, 256 new tokens) |
| features | `../pipeline/score_candidates.py` (AR NLL), `../pipeline/assemble.py` (merge + `len_chars`) |

Generation settings per model, as in `config.json`:

- DiffuGPT and DiffuLLaMA use the anchor paper's own sampler (`model.generate_samples`).
- Dream uses `alg="entropy"`, temperature 0.2 and top_p 0.95.
- LLaDA uses `block_length = gen_length` (fully non-autoregressive), temperature 0.7 and
  low-confidence remasking.

These are the decoding settings each model's authors describe for diverse sampling. The
known traps are guarded in code: the anchor's shifted output, Dream's random-order default,
and LLaDA's semi-AR blocks.
