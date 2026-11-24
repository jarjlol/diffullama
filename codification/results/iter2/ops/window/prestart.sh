#!/usr/bin/env bash
# prestart.sh <output.jsonl>... : run by gen_k.sh / greedy_k.sh before every (re)start of a window job (rule of Mon 23 Nov 2026, Aaditya via Main).
# 1. Cool-down: wait until this job's GPU reads under 80 C, at most 120 s; at most 300 s when there were 3 or more Xid 8 crashes on it in the last 10 min.
# 2. Tail check: rows are appended only after a question finishes (one write + fsync per row); if the LAST line of an output file does not parse (a kill in the middle of a write),
#    the torn tail is moved to <file>.torn.<time> and the file is cut back to its last good line, so the resume-by-uid start does not fail or glue a row onto the torn one.
# Every action is logged to ~/codify/iter2/logs/prestart.log. The GPU is CUDA_VISIBLE_DEVICES (set by queue.sh, PCI bus order).
G=${CUDA_VISIBLE_DEVICES:-0}; LOG=/home/antpc/codify/iter2/logs/prestart.log
log() { echo "$(date '+%F %T') [gpu$G] $*" >> $LOG; }
PCI=$(nvidia-smi -i $G --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-F' 'a-f' | cut -c1-8)   # e.g. 00000000:27:00.0 -> 0000:27:00
BUS=$(nvidia-smi -i $G --query-gpu=pci.bus_id --format=csv,noheader | cut -c10-17 | tr 'A-F' 'a-f')                            # 27:00.0 -> 27:00.
BUS=${BUS%.}
n8=$( (journalctl -k --since "-10min" 2>/dev/null) | grep -c "Xid (PCI:0000:$BUS): 8,")
MAXW=120; [ "${n8:-0}" -ge 3 ] && MAXW=300
w=0; t=$(nvidia-smi -i $G --query-gpu=temperature.gpu --format=csv,noheader)
while [ "${t:-0}" -ge 80 ] && [ $w -lt $MAXW ]; do sleep 10; w=$((w+10)); t=$(nvidia-smi -i $G --query-gpu=temperature.gpu --format=csv,noheader); done
log "start: GPU temp ${t} C after waiting ${w}s (limit ${MAXW}s; Xid 8 in last 10 min: ${n8})"
for f in "$@"; do
  [ -s "$f" ] || continue
  python3 - "$f" <<'PY' >> $LOG 2>&1
import json, sys, time, shutil
p = sys.argv[1]
b = open(p, "rb").read()
lines = b.split(b"\n")
# a good file ends with "\n" (last element empty); every non-empty line must parse
good_end = 0; pos = 0; bad = None
for i, l in enumerate(lines):
    end = pos + len(l) + 1
    if l.strip():
        try: json.loads(l.decode("utf-8")); 
        except Exception: bad = (i, pos); break
        if i == len(lines) - 1: bad = (i, pos)        # last line without trailing newline: parseable but unterminated
    pos = end
if bad is None:
    print(f"tail check ok: {p} ({sum(1 for l in lines if l.strip())} rows)"); sys.exit(0)
i, pos = bad
if i == len(lines) - 1 and lines[i].strip():
    try: json.loads(lines[i].decode("utf-8")); open(p, "ab").write(b"\n"); print(f"tail check: added missing newline to {p}"); sys.exit(0)
    except Exception: pass
torn = b"\n".join(lines[i:])
open(f"{p}.torn.{int(time.time())}", "wb").write(torn)
open(p, "wb").write(b[:pos])
print(f"tail check: torn tail of {p} (from line {i+1}, {len(torn)} bytes) saved to .torn and cut off")
PY
done
exit 0
