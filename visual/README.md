# visual

Only one thing lives here: the workbench, the one interface we keep using.

| Folder | What it is | Open |
| --- | --- | --- |
| [`workbench/`](workbench/) | Pick a phantom version and a simulation setting, edit the main parameters, deploy a policy in the simulator, see the instruments at rest, the centerlines (training's and every other vessel's, with their radii as tubes) and an X-ray mode that looks along the C-arm (no contrast: only the wire; road map; what the policy gets). Chinese and English UI, one-click switch. Modular: environments, sources, policies, instrument window, 3D layers, X-ray mode are separate parts | `python visual/workbench/workbench.py` in env `guidewire_rl`, then http://localhost:8080 in Chrome |

One-off figures are not kept in the repository (user decision 2026-10-28): they are drawn in the agent's
session scratchpad and sent to the user. Everything that was here before (21 figure folders) is in commit
`f7b7440` ("visual: snapshot of every figure folder before the cleanup"); get one back with
`git checkout f7b7440 -- visual/<folder>`.

The C-arm angle selection that used to be `visual/c_arm_angle/` is not a figure: it was restored on 2026-10-28
as [`guidewire_rl/env/c_arm/`](../guidewire_rl/env/c_arm/) (same method, reads the vendored centerlines).
