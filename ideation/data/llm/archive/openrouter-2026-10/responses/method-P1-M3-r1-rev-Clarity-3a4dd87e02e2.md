Review: The method is well-structured and clearly motivated by the research problem, with a detailed algorithmic description of ACE‑v2 and explicit FLOP modeling. However, internal contradictions and undefined terms hinder full comprehension and replication.

Feedback:  
1. **Acceptance logic contradiction:** Step 2d states the first candidate is “accepted as the provisional output,” but the rationale later claims “the first candidate is not automatically accepted.” Clarify whether “accepted” means stored as the current best or selected as the final output, and reconcile this with Step 4 (best‑so‑far selection).  
2. **Undefined hybrid score:** The sanity‑check mentions a “lightweight hybrid score (AR NLL + length penalty)” if the scorer is weakly predictive, but the length‑penalty formula and weighting are unspecified.  
3. **Constraint violation:** Section 7 proposes training a task‑specific reward model for cross‑task validation, which conflicts with the stated “inference‑only constraint (no training, adaptation, or continual pre‑training).”  
4. **Calibration target:** The 30% early‑acceptance rate for δ calibration is arbitrary; justify this choice or discuss sensitivity.

Rating (1-5): 3