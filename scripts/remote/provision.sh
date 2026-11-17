#!/usr/bin/env bash
# One command, run on the laptop, that turns a fresh rented machine into a training machine:
# copies our files over SSH, then runs scripts/remote/setup_machine.sh there (official downloads + installs).
#
#   bash scripts/remote/provision.sh <host> [port]            # user root, key ~/.ssh/vast_ed25519
#   SSH_USER=ubuntu SSH_KEY=~/.ssh/other bash scripts/remote/provision.sh <host> [port]
#   DEMOS="runs/demos_round2f runs/other" bash scripts/remote/provision.sh <host> [port]
#
# Copied (nothing else leaves the laptop; no GitHub, wandb or Vast key goes to the machine):
#   the files in this repository that git tracks or would track (the working copy: uncommitted edits and
#   new files not yet added included; whatever .gitignore excludes stays here)
#   stEVE/ (without its .git)
#   ~/opt/BeamAdapter-v25.12-dev (our patched build)
#   the demonstration folders in DEMOS (default: the one the dagger config reads)
# On the machine everything is at the same place relative to its home as here:
#   ~/Projects/contact-aware-tdmpc2, ~/opt/BeamAdapter-v25.12-dev, ~/opt/SOFA_v25.12.00_Linux.
set -euo pipefail
HOST="${1:?usage: provision.sh <host> [port]}"
PORT="${2:-22}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vast_ed25519}"
DEMOS="${DEMOS:-runs/demos_round2f}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BEAM="$HOME/opt/BeamAdapter-v25.12-dev"
REMOTE_REPO="Projects/contact-aware-tdmpc2"            # relative to the remote home
SSH_OPTS=(-p "$PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
TARGET="$SSH_USER@$HOST"
T0=$SECONDS

[ -f "$BEAM/BeamAdapter/lib/libBeamAdapter.so" ] || { echo "no BeamAdapter build at $BEAM"; exit 1; }
[ -d "$REPO/stEVE/eve" ] || { echo "no stEVE/ in $REPO"; exit 1; }
for d in $DEMOS; do [ -d "$REPO/$d" ] || { echo "no $d in $REPO"; exit 1; }; done

echo "=== rsync on the machine"
ssh "${SSH_OPTS[@]}" "$TARGET" 'command -v rsync >/dev/null || { export DEBIAN_FRONTEND=noninteractive; SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo; $SUDO apt-get update -qq && $SUDO apt-get install -y -qq rsync; }; mkdir -p ~/'"$REMOTE_REPO"' ~/opt'

echo "=== copy files"
RSYNC=(rsync -a --info=stats1 -e "ssh ${SSH_OPTS[*]}")
git -C "$REPO" ls-files -z --cached --others --exclude-standard | "${RSYNC[@]}" --from0 --files-from=- "$REPO/" "$TARGET:$REMOTE_REPO/"
"${RSYNC[@]}" --delete --exclude .git --exclude __pycache__ --exclude '*.egg-info' "$REPO/stEVE/" "$TARGET:$REMOTE_REPO/stEVE/"
"${RSYNC[@]}" --delete "$BEAM/" "$TARGET:opt/BeamAdapter-v25.12-dev/"
for d in $DEMOS; do
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p ~/$REMOTE_REPO/$d"
  "${RSYNC[@]}" "$REPO/$d/" "$TARGET:$REMOTE_REPO/$d/"
done
COPY=$((SECONDS - T0))

echo "=== install on the machine"
ssh "${SSH_OPTS[@]}" "$TARGET" "bash ~/$REMOTE_REPO/scripts/remote/setup_machine.sh"

echo
echo "copy from the laptop: $COPY s"
echo "provision.sh total: $((SECONDS - T0)) s"
