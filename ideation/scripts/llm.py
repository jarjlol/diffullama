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

Multi-key rotation (OpenRouter free tier, ~200 req/day/key): set LLM_API_KEYS
and/or REVIEWER_API_KEYS as comma-separated lists (singular LLM_API_KEY /
REVIEWER_API_KEY still work). Keys are tried in order starting from the persisted
index in data/llm/key_state.json; on HTTP 429 / 401 / 402 the pool advances to the
next key. When every key is rate-limited the call raises RuntimeError starting
with KEY_EXHAUSTED -- re-run after supplying the next key; cached responses are
never re-requested. Keys come only from env vars and are never written to disk.
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


KEY_STATE_FILE = DATA / "llm" / "key_state.json"


def _key_pool(role: str) -> list[str]:
    """Ordered key pool for a role. Reviewer falls back to the LLM pool."""
    if role == "reviewer":
        raw = os.environ.get("REVIEWER_API_KEYS") or os.environ.get("REVIEWER_API_KEY") or ""
        if not raw.strip():
            raw = os.environ.get("LLM_API_KEYS") or os.environ.get("LLM_API_KEY") or ""
    else:
        raw = os.environ.get("LLM_API_KEYS") or os.environ.get("LLM_API_KEY") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _load_key_state() -> dict:
    try:
        return json.loads(KEY_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_key_state(state: dict) -> None:
    KEY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _pool_index(role: str, n: int) -> int:
    if n <= 0:
        return 0
    try:
        return int(_load_key_state().get(role, 0)) % n
    except (ValueError, TypeError):
        return 0


def _advance_pool(role: str, n: int) -> None:
    if n <= 0:
        return
    state = _load_key_state()
    try:
        cur = int(state.get(role, 0))
    except (ValueError, TypeError):
        cur = 0
    state[role] = (cur + 1) % n
    _save_key_state(state)


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


class _BadResponse(Exception):
    """Transient malformed 200 response (no usable content); safe to retry."""


def _openai_once(base: str, key: str, model: str, temperature: float, system: str, user: str) -> str:
    """Single POST without retry; HTTPError/_BadResponse propagate to the caller."""
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base}/chat/completions", data=data, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    try:
        msg = out["choices"][0]["message"]
        text = msg.get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise _BadResponse(f"no choices in response (keys={list(out) if isinstance(out, dict) else type(out)})") from e
    if not text.strip():
        raise _BadResponse("empty message content")
    return text


def _openai(system: str, user: str, role: str) -> str:
    base = _env(role, "BASE_URL", "https://api.openai.com/v1").rstrip("/")
    pool = _key_pool(role)
    if not pool:
        name = "REVIEWER_API_KEY(S)" if role == "reviewer" else "LLM_API_KEY(S)"
        raise RuntimeError(f"{name} is not set")
    model = _env(role, "MODEL", "gpt-4o")
    temperature = float(_env(role, "TEMPERATURE", "0.7"))
    start = _pool_index(role, len(pool))
    rate_limited_keys = 0
    last_err: Exception | None = None
    for offset in range(len(pool)):
        key = pool[(start + offset) % len(pool)]
        for attempt in range(6):
            try:
                text = _openai_once(base, key, model, temperature, system, user)
                # Persist the working index so re-runs resume on a live key.
                state = _load_key_state()
                state[role] = (start + offset) % len(pool)
                _save_key_state(state)
                return text
            except urllib.error.HTTPError as e:
                body = e.read()[:500] if hasattr(e, "read") else b""
                last_err = e
                if e.code == 429 and b"upstream" in body:
                    # Provider-side free-model congestion, not our key's quota:
                    # retry the same key with backoff instead of rotating.
                    if attempt < 5:
                        time.sleep(2 ** attempt * 10)
                        continue
                    raise RuntimeError(f"LLM HTTP 429 (provider congested): {body[:300]!r}") from e
                if e.code in (429, 401, 402):
                    # Key-level limit / invalid / no credits: try the next key.
                    rate_limited_keys += 1
                    _advance_pool(role, len(pool))
                    time.sleep(5)
                    break
                if e.code in (500, 502, 503) and attempt < 2:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise RuntimeError(f"LLM HTTP {e.code}: {body!r}") from e
            except _BadResponse as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt * 10)
                    continue
                break  # same-key retries exhausted: try the next key
    raise RuntimeError(
        f"KEY_EXHAUSTED ({role}): all {len(pool)} key(s) rate-limited or invalid "
        f"(last: {last_err}). Supply the next key via "
        f"{'REVIEWER_API_KEYS' if role == 'reviewer' else 'LLM_API_KEYS'} and re-run; "
        "cached responses are never re-requested.")


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
