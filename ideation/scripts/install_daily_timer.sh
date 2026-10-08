#!/usr/bin/env bash
# Install (or remove) a systemd user timer that runs daily_run.sh at 05:45 IST every day.
# If the machine is off or asleep at 05:45, the run starts as soon as it is back (Persistent=true).
#
#   ideation/scripts/install_daily_timer.sh              # install and enable
#   ideation/scripts/install_daily_timer.sh --uninstall  # remove
#   systemctl --user list-timers ideation-daily.timer    # next run time
#   journalctl --user -u ideation-daily.service          # service log (full log: ~/.local/state/ideation/)
set -euo pipefail
UNIT_DIR="$HOME/.config/systemd/user"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/daily_run.sh"

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl --user disable --now ideation-daily.timer 2>/dev/null || true
  rm -f "$UNIT_DIR/ideation-daily.timer" "$UNIT_DIR/ideation-daily.service"
  systemctl --user daemon-reload
  echo "removed"; exit 0
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/ideation-daily.service" <<EOF
[Unit]
Description=ResearchAgent ideation run (one day of free-tier quota)

[Service]
Type=oneshot
ExecStart=$SCRIPT
TimeoutStartSec=infinity
EOF
cat > "$UNIT_DIR/ideation-daily.timer" <<EOF
[Unit]
Description=Run the ideation pipeline daily after the OpenRouter quota reset

[Timer]
OnCalendar=*-*-* 05:45:00 Asia/Kolkata
Persistent=true

[Install]
WantedBy=timers.target
EOF
chmod +x "$SCRIPT"
systemctl --user daemon-reload
systemctl --user enable --now ideation-daily.timer
systemctl --user list-timers ideation-daily.timer --no-pager
