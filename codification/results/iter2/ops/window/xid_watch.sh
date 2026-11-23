#!/usr/bin/env bash
# xid_watch.sh : GPU 0 health rule for window 2 (Main, Mon 23 Nov 2026): on ANY Xid 79 on PCI 27:00, or a THIRD Xid 8 there today, touch STOP_gpu0 so GPU 0 stops for the rest of the window
# (no auto-restart) and write ~/codify/iter2/XID_STOP_gpu0.txt. Only touches our own STOP file. Needs no sudo (journalctl -k / dmesg as the lab allows).
D=/home/antpc/codify/iter2; LOG=$D/logs/xid_watch.log
while :; do
  k=$( (journalctl -k --since "2026-11-23 00:00" 2>/dev/null; dmesg 2>/dev/null) | grep -i "Xid (PCI:0000:27:00)" | sort -u)
  n8=$(echo "$k" | grep -c "): 8,"); n79=$(echo "$k" | grep -c "): 79,")
  if [ "$n79" -ge 1 ] || [ "$n8" -ge 3 ]; then
    echo "$(date '+%F %T') Xid8=$n8 Xid79=$n79 on GPU 0: touching STOP_gpu0, no restart" | tee -a $LOG > $D/XID_STOP_gpu0.txt
    touch $D/STOP_gpu0; exit 0
  fi
  echo "$(date '+%F %T') ok: Xid8=$n8 Xid79=$n79" >> $LOG
  sleep 30
done
