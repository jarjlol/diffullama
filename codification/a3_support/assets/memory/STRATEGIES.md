# Operator fallback experimentation memory

T4 support; consult before repair. Sources are the repository's recorded operational
receipts, not newly reproduced failures. Final task must use the peer-reviewed T4 if available.

| Area / symptom | Cause | Fix / command | Evidence |
|---|---|---|---|
| vLLM context overflow after several Arbor turns | 32k insufficient for prompt plus16k output | --max-model-len 131072 | NEXT_ITERATION.md, Process lessons; results/iter1/ops/start_vllm.sh |
| TP2 hangs in first all-reduce | Blackwell PCIe communication path | --data-parallel-size 2 | HANDOFF.md §3; start_vllm.sh |
| Qwen tool calls not parsed | XML tool format | --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 | start_vllm.sh |
| Arbor gate free text delayed/never submitted | Routed to companion LLM | /approve, /skip, /edit TEXT, /answer TEXT; log exact reply | NEXT_ITERATION.md, Process lessons |
| Missed majority execution | Depth2 restatement confused with duplicate | Inspect node depth; approve concrete child of an approved direction | ITERATION_1.md node1 child; NEXT_ITERATION.md |
| HF unreachable / TLS interception | Lab network | HF_HUB_OFFLINE=1; download on laptop, scp, verify sha256; never disable TLS | HANDOFF.md §3 |
| datasets.load_dataset fails offline despite hub cache | Loader cache differs | Read cached parquet directly | ITERATION_1.md I-17 |
| corpus too small:1854<2000 | Validation lacks requested segments | Validation then held-out test WikiText segments | ITERATION_1.md I-17 |
| CUDA error: the launch timed out and was terminated | Display watchdog on both GPUs | Resume by UID; max8 retries; no duplicate card jobs | ITERATION_1.md I-18 |
| DiffuGPT-S prompt exceeds512 | 4shot prompt alone too large | Per-model1shot,192new-token config | ITERATION_1.md I-7 |
| Dream candidates identical / low diversity | T0.2 under generation budget | Experiment uses T0.7; label fidelity versus sampling confound | ITERATION_1.md I-16 |
| DiffuGPT earlier hidden states unaffected by later token | Accidental causal SDPA mask | All-zero4D mask; check_attention must pass | ITERATION_1.md I-14 |
| qwen0.5b selectors all zero | Scorer split at first dot | Parse scorer between ar. and final field | ITERATION_1.md I-19 |
| AR scores include unrelated trailing text | trim only stops new Question line | --cut; retain separate *-cut names and compare; do not overwrite originals | iter2/aaditya-worker ITERATION_2.md I-22 |
| GitHub/LFS push/network timeout | Workstation network unreliable | Local git bundle, scp, git fetch bundle in separate clone; never edit running source checkout | HANDOFF.md §3 |
| execvpe(/bin/bash) failed on Windows | WSL bash resolved first | Git Bash first in PATH | ITERATION_1.md I-10 |
| Shared GPU held idle8h | Server started before consumer ready | Start vLLM with Arbor; stop owned server as soon as consumer ends | NEXT_ITERATION.md, Process lessons |
| Free OpenRouter requests exhausted / retired model slug | Free endpoint instability | Preserve response cache; use approved local vLLM; do not mix generators mid-run | ideation/REPORT.md §7; decision D-2026-10-08-b |
| Duplicate future resume could spawn queue | Old scheduled controllers after early launch | Verify existing consumers; singleton startup wrappers; never restart a healthy lane | operator receipts21Nov02:42 |

One process/card; Tier1 first. Stop team work through STOP_gpu0/1 between steps.
No signals to another user's processes or inherited jobs. Never touch ~/diffullama
or modify ~/codify/diffullama while lanes run. Search runs only when prerequisites
and staffed review are ready; no unattended gate auto-approval.
