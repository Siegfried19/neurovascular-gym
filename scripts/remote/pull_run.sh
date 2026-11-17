#!/usr/bin/env bash
# Pull a run's files from the rented machine back to the laptop (logs, checkpoints, the offline wandb copy).
# Run on the laptop; only copies towards the laptop, never deletes anything on either side.
#
#   bash scripts/remote/pull_run.sh <host> <port> <run-name>            # user root, key ~/.ssh/vast_ed25519
#   LOGS_ONLY=1 bash scripts/remote/pull_run.sh <host> <port> <run-name>   # skip checkpoints (quick look)
set -euo pipefail
HOST="${1:?usage: pull_run.sh <host> <port> <run-name>}"; PORT="${2:?port}"; NAME="${3:?run name}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vast_ed25519}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_REPO="Projects/contact-aware-tdmpc2"
EXCLUDE=(); [ -n "${LOGS_ONLY:-}" ] && EXCLUDE=(--exclude 'checkpoints/')
mkdir -p "$REPO/runs/$NAME"
# --partial-dir: a checkpoint cut off mid-transfer never sits under its final name
rsync -a --partial-dir=.rsync-partial "${EXCLUDE[@]}" \
  -e "ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30" \
  "$SSH_USER@$HOST:$REMOTE_REPO/runs/$NAME/" "$REPO/runs/$NAME/"
rsync -a -e "ssh -p $PORT -i $SSH_KEY -o ServerAliveInterval=30" \
  "$SSH_USER@$HOST:$REMOTE_REPO/runs/$NAME.launch.log" "$REPO/runs/" 2>/dev/null || true
ls -la "$REPO/runs/$NAME/checkpoints" 2>/dev/null | tail -n +2 || true
