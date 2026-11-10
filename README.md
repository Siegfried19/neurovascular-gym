# neurovascular-gym

Reinforcement learning for navigating a guidewire and a catheter from the descending aorta into the right internal carotid artery, inside the BIRO lab's neurovascular phantom (a physical model of the head-and-neck vessels). The simulator is SOFA + BeamAdapter (with our patches) + stEVE (upstream, unmodified) + this repository's fixes on top of them. The algorithm is TD-MPC2 (a learned world model with planning in latent space); the world model and planner are copied from Wilson Wan's contact-aware TD-MPC2. One experiment = one YAML file in `configs/`.

This README is written for people and for coding agents (Claude Code, Codex) alike: paths, commands and the things that must not be touched are spelled out. Read it to the end before changing anything.

## What is in the repository

| Location | What it is |
| --- | --- |
| `guidewire_rl/` | Our code, one Python package: `env/` (the MDP side), `model/` (the solver side), `train/` (the training loop), `config.py` (reads the YAML) |
| `simulator_modifications/` | Our changes to the simulator: 7 Python files that register fixed versions under stEVE's own class names, and `beamadapter/` with 6 patches for BeamAdapter plus the build script |
| `configs/` | 10 YAML files, a chain of inheritance (see "Configs") |
| `scripts/` | Entry points: `train.py`, `evaluate.py`, `inspect_env.py`, `record_demos.py`, `make_snapshots.py` |
| `tests/` | pytest, roughly one file per part (see "Tests") |
| `visual/workbench/` | A viser workbench: pick a phantom version and a setting, deploy a policy, look at the instruments and centerlines, an X-ray mode. Viewed in the browser, the simulation runs in Python |
| `docs/` | Three technical documents, in Chinese: `training_framework.md` (how to use the framework), `simulator_changes.md` (every simulator problem we found, what we did about it, numbered), `simulation_speed.md` (the speed-up work) |
| `environment.yml`, `pyproject.toml` | The conda environment and the package definition |
| `THIRD_PARTY_NOTICES.md` | Where the copied code and data come from and under which licence |

Not in the repository, to be installed separately: SOFA 25.12, stEVE (pinned commit), our build of BeamAdapter. Optional: `reference/` (Wilson's original package and the G13 checkpoint, only needed to reproduce G13). Output directories `runs/` and `outputs/` are not committed.

The folders have their own `README.md` (`guidewire_rl/env/two_device/`, `env/c_arm/`, `env/vessels/biro_phantom/`, `env/instruments/data/`, `simulator_modifications/`, `visual/workbench/`), written in Chinese: what every file in that folder does, where the data came from. Some links in them and in `docs/` point to internal planning and log files (`todo_list/`, `agent_log/`); those are not part of this repository, skip them.

## Installation (Linux; the versions in use on 2026-11-10)

Paths are the ones on our machine (`~/opt/…`, `~/Project/…`). If you install elsewhere, change the environment variables below and the variables at the top of `build.sh` accordingly.

1. **The SOFA v25.12.00 Linux binary**, unpacked to `~/opt/SOFA_v25.12.00_Linux` (download from the GitHub releases of `sofa-framework/sofa`). It ships SofaPython3 (Python 3.12) and BeamAdapter; we do not use its BeamAdapter (step 5).

2. **The conda environment**:
   ```bash
   conda env create -f environment.yml          # env name guidewire_rl, Python 3.12, torch built for CUDA 12.8
   ```

3. **stEVE** (package name `eve`), cloned into `stEVE/` at the root of this repository (already in `.gitignore`), pinned to the upstream commit `c3704ccf8f5f6deca443ca98f51a023c2f2b4838` (branch `feat/sofa25.12_compatibility`, 2026-04-14), then two editable installs:
   ```bash
   git clone https://github.com/lkarstensen/stEVE.git stEVE
   git -C stEVE checkout c3704ccf8f5f6deca443ca98f51a023c2f2b4838
   conda run -n guidewire_rl python -m pip install -e stEVE
   conda run -n guidewire_rl python -m pip install -e .        # installs guidewire_rl and simulator_modifications
   ```

4. **SOFA's environment variables**, stored in the conda env (`conda run` and `conda activate` both apply them). The order matters: the first entry of `SOFA_PLUGIN_PATH` must be our BeamAdapter build, and `LD_LIBRARY_PATH` must end with the env's own `lib` (SofaPython3 looks for `libpython3.12.so.1.0` there):
   ```bash
   SOFA=$HOME/opt/SOFA_v25.12.00_Linux
   ENV=$(conda run -n guidewire_rl python -c 'import sys; print(sys.prefix)')
   conda env config vars set -n guidewire_rl \
     SOFA_ROOT=$SOFA \
     SOFA_PLUGIN_PATH=$HOME/opt/BeamAdapter-v25.12-dev:$SOFA/plugins \
     PYTHONPATH=$SOFA/plugins/SofaPython3/lib/python3/site-packages \
     LD_LIBRARY_PATH=$SOFA/lib:$SOFA/plugins/SofaPython3/lib:$ENV/lib
   ```

5. **Build our BeamAdapter**:
   ```bash
   bash simulator_modifications/beamadapter/build.sh
   ```
   It checks out upstream BeamAdapter at the commit SOFA 25.12 ships into `simulator_modifications/beamadapter/build/`, applies the 6 patches in order, builds, and installs to `~/opt/BeamAdapter-v25.12-dev/BeamAdapter`. Needs: the system g++ 11, cmake, ninja and the Boost headers (the script looks for `bin/cmake`, `bin/ninja` and `include/` under the directory in the environment variable `TOOLS`, default `~/anaconda3/envs/sofa-build`; a conda env with `cmake ninja boost-cpp` installed is enough); Eigen 3.3.7 is downloaded by the script. Why the build is mandatory: the BeamAdapter shipped with SOFA segfaults when an instrument is pushed by exactly 0.01 mm (fixed upstream later; one of the patches), and our four patches provide switches the configs use (minimum beam length, densest node sampling where two devices overlap, no collision points on the part of the wire sheathed by the catheter, immediate re-mapping of per-device collision points). `build/` is an intermediate and can be deleted at any time.

6. **Run every Python command as**
   ```bash
   conda run --no-capture-output -n guidewire_rl python …
   ```
   Calling the env's `python` by its path does not pick up the variables of step 4 and `import Sofa` fails.

### Check the installation

```bash
# tests that do not start SOFA, a few minutes
conda run --no-capture-output -n guidewire_rl python -m pytest -q -m "not sofa and not checkpoint"
# everything, including real SOFA scenes (whole scripted episodes etc.), about 20-30 minutes
conda run --no-capture-output -n guidewire_rl python -m pytest -q
```

Expected: no failures. The tests marked `checkpoint` / `reference` are skipped automatically without `reference/`; those marked `slow` (the C-arm angle search, minutes each) run only with `--run-slow`. The markers are defined in `pyproject.toml`, the skipping logic in `tests/conftest.py`.

### Optional: `reference/` and the old environment

`guidewire_rl/__init__.py` defines `REFERENCE_ROOT = <repo root>/reference` and `REFERENCE_G13_CHECKPOINT = reference/checkpoints_g13/step_260000.pt`. With Wilson's `contact-aware-tdmpc2` repository and the G13 checkpoint placed there, `tests/test_reference_fidelity.py` and the bit-for-bit comparison of `configs/g13.yaml` run, and `scripts/train.py --init-from-reference` can read the checkpoint. Reproducing G13 exactly also needs a second conda env `contact_aware_tdmpc2` (Python 3.10, SOFA 23.06, `eve` imported from Wilson's package); see `docs/training_framework.md`, section "两个环境" (the two environments). Without any of this, new training is unaffected.

## Running

```bash
# train (outputs in runs/<run.name>/: config.yaml, episodes.jsonl, metrics.jsonl, checkpoints/)
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/two_devices_ica.yaml
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/two_devices_ica.yaml \
    --set training.total_steps=1000 --run-name smoke          # override one value, name the run
conda run --no-capture-output -n guidewire_rl python scripts/train.py --config configs/two_devices_ica.yaml \
    --resume runs/<run>/checkpoints/step_N.pt
# evaluate a checkpoint (fixed seeds, the planner's mean action)
conda run --no-capture-output -n guidewire_rl python scripts/evaluate.py --config configs/two_devices_ica.yaml \
    --checkpoint runs/<run>/checkpoints/step_N.pt --episodes 20 --workers 8
# show what the policy sees (observation layout, reward and labels for a few random actions)
conda run --no-capture-output -n guidewire_rl python scripts/inspect_env.py --config configs/two_devices_ica.yaml
# record demonstrations with the scripted controller (demos for the trainer)
conda run --no-capture-output -n guidewire_rl python scripts/record_demos.py --config configs/two_devices_ica.yaml \
    --seeds 1000-1019 --workers 10 --out runs/demos_ica
# build a library of episode-start snapshots (simulator states of the scripted controller at each stage, for FromSnapshot)
conda run --no-capture-output -n guidewire_rl python scripts/make_snapshots.py --help
# the workbench (then open http://localhost:8080 in Chrome; needs WebGL)
conda run --no-capture-output -n guidewire_rl python visual/workbench/workbench.py --env two_devices_ica
```

Every script's module docstring lists all of its arguments.

## Map: which question each folder answers

### `guidewire_rl/` is split along the three parts of reinforcement learning

```
guidewire_rl/
├─ env/      the MDP side: simulation scene, observations, actions, reward, how an episode starts and ends,
│            labels (training targets the policy never sees)
├─ model/    the solver side: world model, planner, update step (agent.py, world_model.py), auxiliary heads (aux_heads.py)
├─ train/    putting them together: replay, single-env loop (trainer), multi-process (parallel_trainer + env_pool),
│            curriculum, demos, checkpoint, evaluate, metrics, logger, setup
└─ config.py reads the YAML: `extends` inheritance + `--set` overrides
```

The `env:` section of a YAML maps to `env/`, the `model:` and `training:` sections to the other two.

### `env/` mixes two ways of splitting

- **By kind of part**: `observations/` (observation terms), `labels/`, `instruments/`, `vessels/`, `c_arm/` (imaging angle). Each folder is one kind of replaceable part, chosen by name in the config. At the top, `builder.py` (assembles the scene), `registry.py` (name → class), `guidewire_env.py` (the gymnasium wrapper) and `actions.py` (policy action → stEVE command) are the assembly framework.
- **By task**: `two_device/` keeps every part of the two-device task together: `route.py` (the route and its ruler), `phases.py` (the referee: five phases, success, hard failure), `observations.py`, `rewards.py`, `targets.py`, `start.py`, `terminal.py`, `truncation.py`, `wall.py` (distance to the wall, wall breach), `labels.py`, `actions.py` (the safety shell), `snapshot.py` (episode start from a saved simulator state), `script.py` (the scripted controller, the teacher; not a config part). The table of parts is in that folder's `README.md`.
- `vessels/` holds three different things: the class that puts the vessels into stEVE at runtime (`biro_phantom.py`), offline scripts that generate meshes and centerlines from the STL (in use: `biro_rebuild.py`, `biro_spheres.py`, `biro_cut.py`; earlier versions: `biro_phantom_root.py`, `biro_phantom_all.py`, `biro_phantom_lumen*.py`), and the data folder `biro_phantom/` (what every file is, its hash, how to regenerate it: the `README.md` there).

### How a name becomes a class

A config says `type: SofaBeamAdapter`, `type: BiroPhantomArch`, `type: PhaseDetector` and so on; `env/registry.py` looks the name up: stEVE's own parts under stEVE's names, ours registered with `@register("observation")` and the like. `simulator_modifications/` registers its fixed versions under **stEVE's original class names** (`SofaBeamAdapter`, `CenterlineRandom`, `Manual`, `Tracking2D`, `NormalizeTracking2DEpisode`, `BruteForceBFS`, `TrackingOnly`), so configs keep writing the original names and get the fixed classes. This only happens with upstream stEVE (`simulator_modifications.UPSTREAM`); with Wilson's old eve nothing is registered.

### Configs

A chain of inheritance: dictionaries merge key by key, lists are replaced whole, `null` deletes an inherited key:

```
g13.yaml                      G13 reproduction: every number from the training config stored in the G13 checkpoint (Wilson's eve, old env)
└─ g13_upstream.yaml          the same task on upstream stEVE (the guidewire becomes upstream's Angled)
   └─ phantom_g13.yaml        vessels replaced by the phantom
      └─ phantom_friction.yaml      friction 0.001 -> 0.1
         ├─ phantom_root.yaml -> phantom_all_vessels.yaml   aorta centerline extended to the root / a centerline for every vessel (lab-input version)
         ├─ phantom_real_mass.yaml   single guidewire, real mass, 2 ms physics step
         └─ two_devices_round1.yaml  guidewire + hooked catheter, target right common carotid (round one)
            └─ two_devices_real_mass.yaml   both devices at real mass, 4 ms physics step, all simulator switches on
               └─ two_devices_ica.yaml      target right internal carotid, C-arm angle per target, gamma pinned -- the one in use
```

The header comment of every file says what changed relative to its parent, why, and when it was decided. `g13.yaml` only runs in the old environment.

### Data flow

```
configs/*.yaml -> scripts/train.py -> config.py
  -> env/builder.py finds the parts through the registry -> stEVE scene (simulator_modifications underneath) -> GuidewireEnv
  -> train/trainer runs the loop, calling model/agent for planning and updates
  -> runs/<name>/
```

## Where to change things

- **Add an observation / reward / label / terminal / truncation**: subclass the corresponding stEVE base class, register it with `@register("observation")` etc., add one line to the YAML. The observation size and the model's input layer follow automatically; after an observation change, old checkpoints cannot be resumed.
- **Change simulator behaviour** (contact, collision, speed, seeding, coordinate transforms): `simulator_modifications/`. If Python suffices, write a subclass and register it under the original name; if BeamAdapter's C++ must change, add a patch: edit in `beamadapter/build/src` (the baseline with the existing patches applied and `git add -A`), `git -C build/src diff > <name>.patch`, list it in `PATCHES` in `build.sh`, rebuild. `build.sh` resets to the pinned commit and re-applies all patches every run, so hand edits not saved as a patch are lost. Every change gets a problem number in `docs/simulator_changes.md`.
- **Replace vessel data**: never edit the data files by hand. Regenerate (commands in `env/vessels/biro_phantom/README.md`), update the hashes in `tests/test_biro_phantom.py` and the README's table.
- **Replace the C-arm angle table**: recompute with `env/c_arm/choose_angle.py`, write to `env/c_arm/data/`; see `env/c_arm/README.md`.
- **A new experiment**: a new YAML whose first line is `extends: two_devices_ica.yaml`, listing only what changes. For a one-off change of a value use `--set section.key=value`.

### Files that must not be edited by hand

- Copied code (kept identical to its source; any change must be documented): `guidewire_rl/model/agent.py`, `model/world_model.py`, `env/labels/force_reconstruction.py`, `env/labels/distal_force.py`, `env/observations/vessel_geometry.py`; the two official BeamAdapter patches `official_irc_fix.patch`, `official_rod_straight_fix.patch`.
- Hash-locked data: `env/vessels/biro_phantom/*` (`tests/test_biro_phantom.py`), `env/instruments/data/simmons.csv` (`tests/test_instruments.py`), `env/c_arm/data/*.json`, `visual/workbench/assets/makehuman_base.obj`.

## Tests

`pytest -q -m "not sofa and not checkpoint"` takes a few minutes; the full run about 20-30 minutes. After a change, first run the files that cover it directly; run the full suite before committing.

| Changed | Run |
| --- | --- |
| `config.py` / `registry.py` | `test_config.py`, `test_registry.py` |
| `model/` | `test_agent.py` |
| `train/` | `test_trainer.py`, `test_parallel.py`, `test_replay.py`, `test_two_device_training.py` (fake env) |
| parts in `two_device/` | `test_two_device_env.py`; the scripted controller `test_two_device_script.py` (includes a whole SOFA episode); wall distance / breach `test_wall_monitor.py`; force labels `test_tip_force_fast.py`; the force gate `test_force_blowup.py`; snapshots `test_snapshot.py` |
| `simulator_modifications/` | `test_simulator_modifications.py`, `test_device_contact.py`, `test_trim_sheathed.py` (all need SOFA) |
| `vessels/`, `instruments/`, `c_arm/`, `observations/` | `test_biro_phantom.py`, `test_instruments.py`, `test_c_arm*.py`, `test_vessel_geometry.py` |
| scene assembly as a whole | `test_env_sofa.py` |
| the workbench | `visual/workbench/tests/test_parts.py` |

The two-device SOFA tests share `tests/two_device_scenes.py` (building scenes, pushing devices, reading the collision chain).

## State of the work (a snapshot of 2026-11-10; the commit history is authoritative after that)

- **Round one of two-device training** (2026-10-31 to 11-02, `two_devices_round1.yaml`, 1.7 M steps) had a success rate of 0; the post-mortem attributed it to the direction of the reward. Round two is being redesigned block by block along the data flow.
- **Already in code and config** (`two_devices_ica.yaml`): target in the right internal carotid; both devices at real mass; physics step 4 ms, collision alarm distance 1.0 mm, no collision points on the wire where it is sheathed by the catheter (together about 2.3x faster than the 2 ms configuration before the speed-up work, numbers in `docs/simulation_speed.md`; real mass itself is slower than round one's default mass, so still slower than round one's 0.1 s per step); a wall breach is treated as a simulator fault (that step dropped, the episode closed as `breach`, counted separately); any contact point's normal force above 30 N truncates the episode as a simulation error; the scripted controller gives up; the C-arm angle is looked up per target branch; the discount factor is pinned at 0.99375.
- **Built and verified, not yet in a training config**: episode start from a saved simulator state (`FromSnapshot` in `two_device/snapshot.py`, library built by `scripts/make_snapshots.py`).
- **Decided, code not yet changed** (block 1 of round two): the safety shell keeps only the insertion-length limits (the code still has round one's shell: speed limit, force limit, equal-length guard and more); contact forces go into the reward, not into the shell; the junction map gives all 6 junctions of the route at once; the task's 2D box is computed from the whole vessel tree; the observation grows from 110 to 120 values. Where the code and this paragraph disagree, the code is the current state and this paragraph the direction.
- **Known pitfalls**: the BeamAdapter shipped with SOFA segfaults on a push of exactly 0.01 mm (hence our build); when both devices have the same insertion length BeamAdapter prints "Case 1" warnings, its own 0.01 mm threshold, harmless; `conda run` without `--no-capture-output` shows no live output; a NaN in the simulation (`SimError`), a wall breach (`WallBreach`) and a force blow-up (`TipForceBlowup`) are truncations, not terminations, and the trainer drops that step; the workbench needs Chrome's WebGL, the desktop app's embedded browser cannot render it.

## Origin and licences

This repository's own code is MIT (`LICENSE`). The copied code and data (Wilson's TD-MPC2 implementation, the lab's phantom data and catheter code, the official BeamAdapter patches, the Simmons catheter trace, the MakeHuman body mesh) are listed with their sources and licences in `THIRD_PARTY_NOTICES.md`.
