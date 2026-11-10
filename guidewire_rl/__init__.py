"""Modular TD-MPC2 training for stEVE guidewire navigation.

Layout (each part maps to one section of the YAML config):

* ``env``   – builds the MDP from YAML: stEVE scene, observation terms, reward,
              terminal/truncation, labels (training targets the controller never sees).
* ``model`` – TD-MPC2 world model, latent-MPPI planner and updates (copied from the
              reference release) plus optional auxiliary prediction heads.
* ``train`` – episode replay, the online training loop, checkpoints and evaluation.
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = WORKSPACE_ROOT / "reference"
REFERENCE_G13_CHECKPOINT = REFERENCE_ROOT / "checkpoints_g13" / "step_260000.pt"
RUNS_ROOT = WORKSPACE_ROOT / "runs"
