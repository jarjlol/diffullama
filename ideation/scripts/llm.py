"""LLM backends for the ResearchAgent pipeline.

The source paper uses GPT-4 (Nov 2023) for every role. No API key is available in
this build environment yet, so the backend is pluggable and every call goes through
one interface with an on-disk cache:

    manual   (default)  Writes each prompt to data/llm/requests/<id>.md and reads the
                        answer from data/llm/responses/<id>.md. If the answer is missing
                        the call raises PendingLLM and the pipeline stops cleanly; fill the
                        responses (with any model, or by hand) and re-run to continue.
    openai              Any OpenAI-compatible /chat/completions endpoint (OpenAI, Groq,
                        OpenRouter, Together, a local vLLM or Ollama server).
    gemini              Google's Generative Language REST API.
    mock                Deterministic canned output, for testing the pipeline wiring only.
                        Mock output is never valid research content.

Every response, from any backend, is written to data/llm/responses/<id>.md, so a run
is fully re-playable and auditable without calling a model again.

Generator and reviewer roles are configured separately (LLM_* and REVIEWER_* env vars)
so the ReviewingAgents can be a different model from the ResearchAgent -- the paper
uses one model for both, and REPORT.md records that as a self-evaluation risk.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from common import DATA, read_text, write_text

REQ_DIR = DATA / "llm" / "requests"
RES_DIR = DATA / "llm" / "responses"


class PendingLLM(Exception):
    """Raised by the manual backend when a response has not been supplied yet."""

    def __init__(self, request_ids: list[str]):
        self.request_ids = request_ids
        super().__init__(f"{len(request_ids)} LLM response(s) pending")


def _env(role: str, key: str, default: str | None = None) -> str | None:
    prefix = "REVIEWER_" if role == "reviewer" else "LLM_"
    return os.environ.get(prefix + key) or (os.environ.get("LLM_" + key) if role == "reviewer" else None) or default


def _post(url: str, payload: dict, headers: dict, retries: int = 4) -> dict:
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {e.read()[:300]!r}") from e
    raise RuntimeError("unreachable")


def _openai(system: str, user: str, role: str) -> str:
    base = _env(role, "BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = _env(role, "API_KEY")
    if not key:
        raise RuntimeError(f"{'REVIEWER' if role == 'reviewer' else 'LLM'}_API_KEY is not set")
    out = _post(f"{base}/chat/completions", {
        "model": _env(role, "MODEL", "gpt-4o"),
        "temperature": float(_env(role, "TEMPERATURE", "0.7")),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }, {"Authorization": f"Bearer {key}"})
    return out["choices"][0]["message"]["content"]


def _gemini(system: str, user: str, role: str) -> str:
    key = _env(role, "API_KEY")
    if not key:
        raise RuntimeError(f"{'REVIEWER' if role == 'reviewer' else 'LLM'}_API_KEY is not set")
    model = _env(role, "MODEL", "gemini-2.5-flash")
    out = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"systemInstruction": {"parts": [{"text": system}]},
         "contents": [{"role": "user", "parts": [{"text": user}]}],
         "generationConfig": {"temperature": float(_env(role, "TEMPERATURE", "0.7"))}},
        {"x-goog-api-key": key},
    )
    return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])


def _mock(system: str, user: str, role: str) -> str:
    if role == "reviewer":
        return "Review: [mock review]\nFeedback: [mock feedback]\nRating (1-5): 3"
    for field in ("Experiment", "Method", "Problem"):
        if f"in the format of\n{field}:" in user:
            return f"{field}: [mock {field.lower()}]\nRationale: [mock rationale]"
    return "Rating (1-5): 3"


BACKENDS = {"openai": _openai, "gemini": _gemini, "mock": _mock}


class LLM:
    """One interface for every model call in the pipeline, with a persistent cache."""

    def __init__(self, backend: str = "manual"):
        if backend not in ("manual", *BACKENDS):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.pending: list[str] = []
        self.calls = 0

    def __call__(self, request_id: str, system: str, user: str, role: str = "generator") -> str | None:
        req = REQ_DIR / f"{request_id}.md"
        res = RES_DIR / f"{request_id}.md"
        if not req.exists():
            write_text(req, f"<!-- role: {role} -->\n# SYSTEM\n\n{system}\n\n# USER\n\n{user}\n")
        if res.exists():
            return read_text(res)
        if self.backend == "manual":
            self.pending.append(request_id)
            return None
        text = BACKENDS[self.backend](system, user, role)
        self.calls += 1
        write_text(res, text)
        return text
