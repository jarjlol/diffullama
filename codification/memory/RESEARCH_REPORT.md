# Research report for Arbor iteration 2 (N3 step 6) — read at ideation

Source: `codification/SOTA.md` (every number read in the full text; arXiv IDs given there).
Red lines: a selector scores **one candidate of one item at a time** from that item's own N candidates and their
cached features. It may not read other models' answers or gold, and it may not regenerate.
Published numbers use **Instruct** checkpoints (ours are Base), so they are context, not a bar.

## Verified methods: mechanism, per-candidate signal, admissible?

| Method (arXiv) | Mechanism | Per-candidate signal | Admissible here? |
|---|---|---|---|
| Majority / self-consistency | plurality of extracted answers | none (answer agreement) | yes, this is the bar (= tuned majority on gate) |
| ORM Rerank (2609.35472) | trained final-state ORM picks the argmax | ORM score of the finished text | no as published (needs a trained ORM, and none is cached) |
| BMC best-of-N (2604.16565) | re-mask 90% of answer tokens, reconstruct with the same DLM, measure the drift | semantic similarity of the candidate to its reconstruction | yes in principle, but needs GPU; not cached |
| S³ composite verifier (2604.06260 App. C) | weighted sum of text heuristics | structure, arithmetic consistency, answer reachability, non-degeneracy (+ model confidence) | **yes, CPU**, if model confidence is dropped |
| Verifier-weighted voting (cited in 2609.35472) | vote weighted by a per-candidate score | any admissible score | yes; our merged 4.1 is one instance |
| Likelihood tie-break (HEX, 2510.05040 Table 5) | NLL breaks majority ties | NLL | yes, but HEX found "no significant benefit"; our Arbor node 2.1 lost too |
| Self-Repulsion (2609.39560), S³ search, Prism, PDC, HEX schedules | change or extend generation | — | **no** (regenerate or search during denoising) |
| Stitching (2602.22871) | PRM picks steps across traces, an AR solver recomputes the answer | PRM step scores | **no** (cross-candidate synthesis and new generation) |

## Leap ideas worth one depth-1 direction each

1. **S³-style composite score as a vote weight** (CPU). Signals that rerank within a vote may beat our single
   arithmetic check, because 4.1 used only one of the five components.
2. **Answer reachability alone**: does the final answer follow from the last explicit calculation? It is S³'s
   s3, our 4.1's second check, cheap.
3. **Non-degeneracy filter before voting**: drop candidates with repetition or junk after `#### N` (I-22 showed
   most have some) before counting votes; report how many votes change.

Do not propose: cross-model consensus (rejected twice in iteration 1), trained verifiers, or anything that
regenerates text.
