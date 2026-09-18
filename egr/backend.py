"""DiffusionBackend protocol + MockBackend + DiffuLLaMABackend. See docs/01-architecture.md
Sec 8 and the confidence-baseline correctness note (H-5).

================================================================================================
STATUS as of 2026-08-31: DiffuLLaMABackend is a full implementation but IS NOT CONNECTED and
MUST NOT be run on a laptop. Loading `diffusionfamily/diffullama` needs a GPU with tens of GB
of VRAM/RAM, which the development machine for this POC does not have. By explicit instruction,
nothing in this session loads that model: every test and every acceptance criterion through
Phase 2 (docs/04-build-phases.md) runs against MockBackend, on CPU, and imports neither torch
nor transformers -- both are lazily imported only inside DiffuLLaMABackend.__init__, so the rest
of the package works with neither installed.

DiffuLLaMABackend.__init__ additionally refuses to run unless called with `allow_load=True`
(the CLI's `--allow-load` flag) -- a deliberate safeguard, not a placeholder oversight, so that
selecting `--backend diffullama` without that flag fails loudly instead of silently trying to
download a 6.7B-parameter checkpoint.

TODO(compute): once GPU access is available (project-docs/01-project-brief.md), this class
should work as written -- run docs/04-build-phases.md Phase 3's smoke test first. If anything
needs changing, it should be small: the tokenize/infill/confidence bodies below already wire up
model.py's real `generate_samples` and `get_anneal_attn_mask`, unmodified (P5). Nothing else in
the package (policy.py, loop.py, cli.py) needs to change -- they only depend on the
DiffusionBackend protocol.
================================================================================================
"""
from __future__ import annotations

import logging
import random
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from egr.types import Span

if TYPE_CHECKING:
    from egr.types import Config, Task

log = logging.getLogger(__name__)

# Newlines are their OWN token, deliberately not merged with surrounding horizontal
# whitespace: MockBackend's oracle mode splices replacement text in by line number (see
# `_oracle_replacement_ids`), which only stays exact if line boundaries fall on token
# boundaries. (The real SentencePiece tokenizer does NOT have this property -- it glues a
# trailing indent onto the next identifier, F-21/H-2 -- but reproducing that bleed here would
# only make the mock harder to reason about without adding coverage; H-2 is exercised against
# real AST line ranges in test_canvas.py regardless of which backend tokenizes.)
_TOKEN_RE = re.compile(r"\n|[ \t]+|[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z_]")


class DiffusionBackend(Protocol):
    name: str
    bos_id: int

    def tokenize(self, text: str) -> tuple[list[int], list[Span]]: ...
    def detokenize(self, ids: Sequence[int]) -> str: ...
    def infill(self, canvas_ids: Sequence[int], src_mask: Sequence[int],
               *, steps: int, temperature: float, seed: int) -> list[int]: ...
    def confidence(self, canvas_ids: Sequence[int]) -> list[float]: ...


class MockBackend:
    """Satisfies DiffusionBackend on CPU, deterministically, with no model weights.

    P2 (docs/01-architecture.md Sec 1): every module except the real backend is developed and
    tested without a GPU. Three modes:

    - "oracle": fills each hole from `oracle_program`, the known-correct source -- proves the
      loop closes at all.
    - "noisy": fills correctly with probability `p`, else emits a plausible wrong token.
    - "stuck": always returns the same filler token -- drives the no-progress detector (H-4).

    `oracle_program` is task-specific, so construct a fresh MockBackend per task when using
    "oracle"/"noisy" -- `make_backend()` below does this automatically.
    """

    name = "mock"
    bos_id = 0

    def __init__(self, mode: str = "oracle", p: float = 0.5,
                 oracle_program: str | None = None, seed: int = 0):
        if mode not in ("oracle", "noisy", "stuck"):
            raise ValueError(f"unknown mock mode {mode!r}")
        self.mode = mode
        self.p = p
        self.oracle_program = oracle_program
        self._rng = random.Random(seed)
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}
        self._last_text: str | None = None
        self._last_spans: list[Span] | None = None

    def _id_for(self, tok: str) -> int:
        if tok not in self._vocab:
            idx = len(self._vocab) + 10  # ids below 10 reserved (bos_id = 0)
            self._vocab[tok] = idx
            self._inv[idx] = tok
        return self._vocab[tok]

    def tokenize(self, text: str) -> tuple[list[int], list[Span]]:
        ids, spans = [], []
        for m in _TOKEN_RE.finditer(text):
            ids.append(self._id_for(m.group(0)))
            spans.append(Span(m.start(), m.end(), "char"))
        self._last_text, self._last_spans = text, spans
        return ids, spans

    def detokenize(self, ids: Sequence[int]) -> str:
        return "".join(self._inv.get(i, "") for i in ids)

    def infill(self, canvas_ids: Sequence[int], src_mask: Sequence[int],
               *, steps: int, temperature: float, seed: int) -> list[int]:
        """Returns `len(canvas_ids) - 1` tokens -- position 0 (BOS) is dropped, mirroring the
        real model's shift (`model.py:156`, `x0 = x0[:, 1:]`). `DiffuLLaMABackend.infill`
        returns `generate_samples`'s output as-is, which already has this shape; MockBackend
        must match it so `loop.py` treats both backends identically (F-3 / H-1).
        """
        rng = random.Random(seed)
        out = list(canvas_ids)
        holes = [i for i, m in enumerate(src_mask) if m == 0]
        if not holes:
            return out[1:]

        if self.mode == "stuck":
            stuck_id = self._id_for("<stuck>")
            for i in holes:
                out[i] = stuck_id
            return out[1:]

        replacement_ids = self._oracle_replacement_ids(holes)

        for k, i in enumerate(holes):
            use_oracle = self.mode == "oracle" or (self.mode == "noisy" and rng.random() < self.p)
            if use_oracle and replacement_ids is not None and k < len(replacement_ids):
                out[i] = replacement_ids[k]
            elif use_oracle and replacement_ids is not None:
                out[i] = self._id_for(" ")  # hole wider than the fix needs -- pad, don't fail
            else:
                out[i] = self._id_for(f"<wrong-{rng.randint(0, 999)}>")
        return out[1:]

    def _oracle_replacement_ids(self, holes: list[int]) -> list[int] | None:
        """Text-level splice: find the char range `holes` covers in the canvas that was just
        tokenized, take the SAME LINE RANGE from `oracle_program`, tokenize that.

        This works without token-level alignment between the (possibly still-buggy) current
        program and the oracle program, because every policy produces whole-line spans (H-2)
        -- "same lines" is a well-defined correspondence even though the two programs are
        different token sequences. `i - 1` mirrors canvas.py's shift convention (a canvas
        position is always one more than its token index) but is a local, test-only mirror,
        not a second copy of the real index authority -- MockBackend never touches a real
        Canvas object.
        """
        if self.oracle_program is None or self._last_text is None or self._last_spans is None:
            return None
        token_idx = [i - 1 for i in holes if 0 <= i - 1 < len(self._last_spans)]
        if not token_idx:
            return None
        lo = self._last_spans[min(token_idx)].start
        hi = self._last_spans[max(token_idx)].end
        # Whole-line-aligned holes (H-2) always end exactly at a line boundary, so counting
        # newlines strictly before `hi` already gives the right EXCLUSIVE stop index for a
        # 0-indexed line slice -- e.g. hi landing right after the 2nd newline means "2 whole
        # lines consumed", i.e. slice up to (not through) index 2. Adding another +1 here
        # would pull in one line too many, which is exactly the bug this comment replaced.
        start_line = self._last_text.count("\n", 0, lo)
        end_line = self._last_text.count("\n", 0, hi)
        oracle_lines = self.oracle_program.splitlines(keepends=True)
        if start_line >= len(oracle_lines):
            return None
        replacement_text = "".join(oracle_lines[start_line:end_line])
        return self.tokenize(replacement_text)[0]

    def confidence(self, canvas_ids: Sequence[int]) -> list[float]:
        # A fixed, seed-independent function of position -- MockBackend exists so
        # ConfidencePolicy has something to rank against off a GPU, not to model a real
        # model's uncertainty. H-5's "must equal max softmax, must be seed-invariant" test
        # applies to DiffuLLaMABackend.confidence(), not this stand-in.
        return [0.5 for _ in canvas_ids]


class DiffuLLaMABackend:
    """Wraps model.py's `generate_samples` over `diffusionfamily/diffullama`.

    NOT CONNECTED YET on this machine -- see the module docstring. Refuses to load unless
    `allow_load=True`.
    """

    name = "diffullama"

    def __init__(self, model_name: str = "diffusionfamily/diffullama", device: str = "cuda",
                 flash_attn: str = "eager", allow_load: bool = False):
        if not allow_load:
            raise RuntimeError(
                "DiffuLLaMABackend refused to load a model. This is a POC safeguard, not a "
                "bug: loading 'diffusionfamily/diffullama' needs a GPU and was deliberately "
                "never run on the development laptop (project-docs/02-decision-log.md P-5). "
                "Pass allow_load=True (`--backend diffullama --allow-load` on the CLI) once "
                "this is running on compute that can actually hold the model -- "
                "docs/04-build-phases.md Phase 3 is the checklist for that session."
            )
        try:
            import torch
            from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
        except ImportError as e:
            raise RuntimeError(
                "torch/transformers are not installed. This POC was developed entirely "
                "against MockBackend (see the module docstring); install the real "
                "dependencies before using DiffuLLaMABackend."
            ) from e

        # additive, unmodified (P5) -- the anchor paper's reproduction path stays byte-identical
        from model import DiscreteDiffusionModel, generate_samples, get_anneal_attn_mask

        self._torch = torch
        self._generate_samples = generate_samples
        self._get_anneal_attn_mask = get_anneal_attn_mask
        self.model_name = model_name
        self.device = device

        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = LlamaForCausalLM.from_pretrained(
            model_name, device_map="auto", _attn_implementation=flash_attn,
            torch_dtype=torch.bfloat16,
        )
        self.model = DiscreteDiffusionModel(
            model=base_model, config=config, tokenizer=self.tokenizer, device=device,
        ).to(device)
        self.bos_id = self.tokenizer.bos_token_id
        log.info("loaded %s on %s (flash_attn=%s)", model_name, device, flash_attn)

    def tokenize(self, text: str) -> tuple[list[int], list[Span]]:
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        spans = [Span(s, e, "char") for s, e in enc["offset_mapping"]]
        return ids, spans

    def detokenize(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def infill(self, canvas_ids: Sequence[int], src_mask: Sequence[int],
               *, steps: int, temperature: float, seed: int) -> list[int]:
        torch = self._torch
        torch.manual_seed(seed)

        class _Args:  # mirrors inf_diffullama.py's argparse namespace, minimally
            shift = True
            diffusion_steps = steps
            logits_temp = temperature
            topp_temp = 0.9

        inputs = {
            "input_ids": torch.tensor([list(canvas_ids)]),
            "src_mask": torch.tensor([list(src_mask)]),
        }
        out = self._generate_samples(self.model, _Args(), self.tokenizer, inputs)
        return out.tolist()[0]

    def confidence(self, canvas_ids: Sequence[int]) -> list[float]:
        """ADDITIVE max-softmax confidence -- a fresh forward pass, NOT model.py's x0_scores.

        H-5: x0_scores (model.py:116) is the log-prob of the SAMPLED token, which measures
        sampling luck, not model certainty, and model.py never returns it anyway. Building the
        confidence baseline on x0_scores would bias the comparison in our favour (M-10).
        """
        torch = self._torch
        x = torch.tensor([list(canvas_ids)]).to(self.device)
        x_embed = self.model.get_embeds(x)
        attn_mask = self._get_anneal_attn_mask(
            x.size(1), 1, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0,
        )
        with torch.no_grad():
            logits = self.model(x, attention_mask=attn_mask)
            probs = torch.softmax(logits.float(), dim=-1)
        return probs.max(dim=-1).values.squeeze(0).tolist()


def make_backend(cfg: "Config", task: "Task | None" = None) -> DiffusionBackend:
    """Construct the backend named by `cfg.backend`.

    For "mock", a fresh instance is returned per call (cheap, and required for oracle/noisy
    modes, which are task-specific -- see MockBackend's docstring). For "diffullama", callers
    should construct once and reuse across tasks (loading the checkpoint is the expensive
    part); this function still supports it but `task` is ignored.
    """
    if cfg.backend == "mock":
        return MockBackend(
            mode=cfg.mock_mode, p=cfg.mock_p, seed=cfg.seed,
            oracle_program=task.oracle_program if task else None,
        )
    if cfg.backend == "diffullama":
        return DiffuLLaMABackend(allow_load=bool(cfg.extra.get("allow_load", False)))
    raise ValueError(f"unknown backend {cfg.backend!r}")
