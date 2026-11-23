#!/usr/bin/env bash
# xid_watch.sh : GPU 0 health watcher for window 2. Rule changed Mon 23 Nov ~17:25 IST (Aaditya via Main; operations only, no data looked at):
#   Xid 8 (launch timeout) on PCI 27:00: only LOG it (time + kernel line) to logs/xid_crashes.log; the queue restarts the job, prestart.sh cools GPU 0 first. No crash limit.
#   Xid 79 (GPU fell off the bus): cannot be restarted from user space. touch STOP_gpu0 (GPU 1 keeps running), write XID79_gpu0.txt, tell Aaditya at once (a reboot also hits krispy).
# Only touches our own STOP file. Needs no sudo.
D=/home/antpc/codify/iter2; LOG=$D/logs/xid_watch.log; CR=$D/logs/xid_crashes.log; touch $CR
while :; do
  k=$( (journalctl -k --since "2026-11-23 00:00" 2>/dev/null; dmesg 2>/dev/null) | grep -i "Xid (PCI:0000:27:00)" | sort -u)
  while IFS= read -r line; do [ -n "$line" ] && ! grep -qF -- "$line" $CR && echo "$line" >> $CR; done <<< "$k"
  n8=$(echo "$k" | grep -c "): 8,"); n79=$(echo "$k" | grep -c "): 79,")
  if [ "$n79" -ge 1 ]; then
    echo "$(date '+%F %T') Xid79=$n79 on GPU 0 (fell off the bus): touching STOP_gpu0; GPU 1 keeps running; tell Aaditya, a reboot would also hit krispy" | tee -a $LOG > $D/XID79_gpu0.txt
    touch $D/STOP_gpu0; exit 0
  fi
  echo "$(date '+%F %T') ok: Xid8=$n8 (logged, no limit) Xid79=$n79" >> $LOG
  sleep 30
done
