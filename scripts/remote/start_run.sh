#!/usr/bin/env bash
# Start (or resume) a two-device training run ON the rented machine, detached from the SSH session,
# together with the guard that ends an evaluation held up by one or two crawling simulators.
#
#   bash scripts/remote/start_run.sh <run-name> [extra scripts/train.py arguments]
#   bash scripts/remote/start_run.sh round2_dagger_v3
#   bash scripts/remote/start_run.sh round2_dagger_v3 --resume runs/round2_dagger_v3/checkpoints/step_40000.pt
#
# wandb is offline here (no wandb key on the rented machine): the copy under runs/<run>/wandb is pulled
# back to the laptop and synced from there. CONFIG and NUM_ENVS can be overridden from the environment.
# The guard (~/tools/exam_straggler_guard.py, copied from the laptop) is optional: skipped if not there.
set -euo pipefail
NAME="${1:?usage: start_run.sh <run-name> [train.py arguments]}"; shift
CONFIG="${CONFIG:-configs/two_devices_ica_dagger.yaml}"
NUM_ENVS="${NUM_ENVS:-20}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA="$(command -v conda || echo "$HOME/miniforge3/condabin/conda")"
cd "$REPO"
mkdir -p runs
if pgrep -f "scripts/train.py .*--run-name $NAME( |\$)" >/dev/null; then
  echo "a training named $NAME is already running"; exit 1
fi
setsid env --default-signal=INT nohup "$CONDA" run --no-capture-output -n guidewire_rl \
  python scripts/train.py --config "$CONFIG" --run-name "$NAME" \
  --set training.num_envs="$NUM_ENVS" --set training.async_envs=true --set logging.wandb.mode=offline "$@" \
  >> "runs/$NAME.launch.log" 2>&1 < /dev/null &
echo "training $NAME started, log runs/$NAME.launch.log"
if [ -f "$HOME/tools/exam_straggler_guard.py" ]; then
  sleep 20                                              # the guard exits when it finds no trainer
  setsid nohup python3 "$HOME/tools/exam_straggler_guard.py" "$NAME" --silent-min 18 \
    >> "runs/$NAME.guard.out" 2>&1 < /dev/null &
  echo "guard started, kills in runs/$NAME/straggler_guard.log"
fi
