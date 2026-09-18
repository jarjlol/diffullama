"""Runs INSIDE the sandboxed subprocess: a fresh `python3` process, resource-limited, cwd'd
into a per-candidate temp directory, stripped environment, no network (H-8). It executes
model-generated code, so nothing here should be trusted with the real filesystem or an
unbounded clock -- see `_guarded_open` and `_set_resource_limits` below.

Talks to the parent (`egr.verify.SandboxVerifier`) over stdout: exactly one line of JSON.
Invoked as `python3 _sandbox_harness.py <sandbox_dir> <timeout_s> <mem_mb>`, with the repo
root on PYTHONPATH so `egr.trace` (stdlib only) is importable.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
import os
import resource
import signal
import sys

from egr.trace import LineTracer, extract_frames


class SandboxTimeout(Exception):
    """Raised by the SIGALRM handler. Caught like any other test failure, so a timing-out
    test looks like one more outcome rather than killing the harness (H-8).
    """


def _alarm_handler(signum, frame) -> None:
    raise SandboxTimeout("wall-clock timeout exceeded")


def _set_resource_limits(cpu_seconds: int, mem_bytes: int) -> None:
    for res, limit in (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_AS, (mem_bytes, mem_bytes)),
        (resource.RLIMIT_FSIZE, (50_000_000, 50_000_000)),
    ):
        try:
            resource.setrlimit(res, limit)
        except (ValueError, OSError):
            pass  # not supported on this platform -- best-effort, docs/01-architecture.md Sec 6.2


def _guarded_open(sandbox_dir: str):
    """A POC-level guard against writing outside the sandbox, not a real filesystem jail (that
    would need containers/chroot, out of scope here). Good enough to catch the common case and
    to make H-8's "writes outside the temp directory" hazard test deterministic.
    """
    real_open = builtins.open
    root = os.path.realpath(sandbox_dir)

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            target = str(file)
            resolved = os.path.realpath(target if os.path.isabs(target) else os.path.join(root, target))
            if resolved != root and not resolved.startswith(root + os.sep):
                raise PermissionError(f"sandbox: refusing to open path outside sandbox: {target!r}")
        return real_open(file, *args, **kwargs)

    return guarded


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sandbox_dir, timeout_s, mem_mb = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    os.chdir(sandbox_dir)
    _set_resource_limits(int(timeout_s) + 5, mem_mb * 1024 * 1024)

    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(timeout_s)))
    except (ValueError, AttributeError):
        pass  # no SIGALRM on this platform -- verify.py's subprocess-level timeout is the backstop

    solution_path = os.path.join(sandbox_dir, "solution.py")
    result: dict = {"outcomes": [], "spectra": {}, "exec_counts": {}, "timed_out": False}

    builtins.open = _guarded_open(sandbox_dir)

    try:
        _load("solution", solution_path)
    except SandboxTimeout:
        result["timed_out"] = True
        print(json.dumps(result))
        return
    except BaseException as e:  # the candidate program itself failed to even import
        result["outcomes"].append({
            "name": "<solution-import>", "passed": False,
            "exc_type": type(e).__name__, "message": str(e),
            "frames": list(extract_frames(e, solution_path)),
        })
        print(json.dumps(result))
        return

    try:
        tests_mod = _load("tests", os.path.join(sandbox_dir, "tests.py"))
    except BaseException as e:  # a broken rendered test suite is OUR bug (H-6), not the
        result["outcomes"].append({                     # candidate's -- verify.py maps this
            "name": "<harness-tests-import>", "passed": False,  # sentinel to HARNESS_ERROR
            "exc_type": type(e).__name__, "message": str(e), "frames": [],
        })
        print(json.dumps(result))
        return

    spectra: dict[int, list[int]] = {}
    exec_counts: dict[int, int] = {}

    for name in sorted(n for n in dir(tests_mod) if n.startswith("test_")):
        fn = getattr(tests_mod, name)
        passed, exc_type, message, frames = True, None, None, ()
        with LineTracer(solution_path) as tracer:
            try:
                fn()
            except SandboxTimeout as e:
                passed, exc_type, message = False, type(e).__name__, str(e)
                result["timed_out"] = True
            except BaseException as e:  # includes SystemExit: a candidate calling sys.exit()
                passed = False          # must not be allowed to kill the harness process
                exc_type = type(e).__name__
                message = str(e)
                frames = extract_frames(e, solution_path)

        result["outcomes"].append({
            "name": name, "passed": passed, "exc_type": exc_type,
            "message": message, "frames": list(frames),
        })
        for line, count in tracer.hit_counts.items():
            n_pass, n_fail = spectra.get(line, [0, 0])
            if passed:
                n_pass += 1
            else:
                n_fail += 1
                exec_counts[line] = exec_counts.get(line, 0) + count
            spectra[line] = [n_pass, n_fail]

        if result["timed_out"]:
            break  # the alarm already fired once; remaining tests do not get their own budget

    signal.alarm(0)
    result["spectra"] = spectra
    result["exec_counts"] = exec_counts
    print(json.dumps(result))


if __name__ == "__main__":
    main()
