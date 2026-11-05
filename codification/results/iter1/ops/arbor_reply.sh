#!/usr/bin/env bash
# arbor_reply.sh -- type a review reply (stdin, newlines folded to spaces) into the Arbor tmux session.
# Start the reply with a gate command (/approve, /skip, /edit <note>, /answer <text>); plain text goes to Arbor's companion LLM instead.
t=$(tr "\n" " " | sed "s/ *$//")
tmux send-keys -t arbor -l "$t" && tmux send-keys -t arbor Enter
