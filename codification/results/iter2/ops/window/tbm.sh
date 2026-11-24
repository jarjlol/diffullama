#!/usr/bin/env bash
# tbm.sh <margin_s> <idle_s> <gpu> <venv> <python args...> : tb.sh with a stop margin before WINDOW_END and an idle requirement
# MIN_LEFT: a margin line needs 15 min of work left after its margin; a margin-0 line needs 60 s.
# (generation lines: margin 1500, idle 300; the scoring lines that follow our own generation: margin 0, idle 20).
M=${1:?margin}; I=${2:?idle}; shift 2
MIN_LEFT=$(( M > 0 ? M + 900 : 60 )) MARGIN=$M IDLE_SECONDS=$I exec bash "$(dirname "$0")/tb.sh" "$@"
