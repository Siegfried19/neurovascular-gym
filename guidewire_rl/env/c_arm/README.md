# C-arm angle

Picks the C-arm angle for the guidewire task on the lab phantom, from the vessel centerlines alone.
Each target's angle is chosen for its whole route from the aorta (since 2026-11-03; "Method").
Training never runs the search. A config either fixes one angle for the whole run
(`env.scene.fluoroscopy.image_rot_zx: [rot_z, rot_x]`; every existing config does) or, since 2026-11-03,
asks for one angle per target (`image_rot_zx: per_target`, the direction agreed on 2026-10-28): the angle
of the episode's target branch is read from a table this script wrote, when the target is drawn. See
"In training" below.

The angle each existing config runs with:

| Config | Devices, vessels | `image_rot_zx` | On a C-arm | Where the angle comes from |
| --- | --- | --- | --- | --- |
| `g13.yaml`, `g13_upstream.yaml` | guidewire; stEVE's generated arch, not the phantom | [20, 5] | — (the names below assume the phantom's frame) | G13's |
| `phantom_g13.yaml`, `phantom_friction.yaml`, `phantom_real_mass.yaml`, `phantom_root.yaml`, `phantom_all_vessels.yaml` | guidewire; lab phantom | [20, 5], inherited from `g13.yaml` | RAO 18°, caudal 5° | G13's, not chosen for the phantom |
| `two_devices_round1.yaml` | guidewire + hooked catheter; model-wall phantom (`rebuilt_cut`) | [61.0, −21.0] | LAO 25°, cranial 10° | this script's angle for all 5 targets on the model-wall version, routes before 2026-11-03 (table below): computed 2026-10-30, set for the two-device run 2026-10-31 so the five branches lie apart in the image |

A new config gets the angle of the config it extends unless it sets its own. No config sets
`per_target` yet.

History: written 2026-10-26 as `visual/c_arm_angle/`, deleted with the other figure folders on 2026-10-28
(snapshot in commit `f7b7440`), restored here the same day with the same method. Only the data paths and
the figure output changed: it reads the vendored centerlines next door instead of copies, and draws a
figure only when asked (`--figure`), into the session scratchpad, not the repository.

Since 2026-10-28 it reads every vessel of the phantom (`phantom_arch_all.json`, 75 centerlines), so a route
drawn over any vessel counts, not only over the 6 training branches. The routes still run along the 6.

```bash
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle                         # ~3 min, all vessels
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle --centerline original   # ~20 s, the lab's 6 only
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle --centerline rebuilt_cut  # ~2 min, model-wall version (rebuilt_cut.json, 60 vessels)
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle --figure /path/out.png  # also draw
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle --centerline rebuilt_cut --write-table  # also write data/angles_rebuilt_cut.json
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle --route own_ostium      # the routes before 2026-11-03
conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle --past-mm 30           # also 30 mm of aorta past the first ostium
```

It prints the tables below: per angle every target's length kept / share covered over its route, and for
each target's own angle its route piece by piece (aorta, trunk, the branch). With `--figure`: left, how
much the worst route is covered at each angle; middle and right, the phantom as stEVE draws it now and at
the chosen angle.

| File | |
| --- | --- |
| `choose_angle.py` | the method, the printout, the optional figure and the table (`--write-table`) |
| `per_target.py` | training side of `image_rot_zx: per_target`: finds the table, sets the angle each episode |
| `data/angles_rebuilt_cut.json`, `data/angles_all_vessels.json` | the tables training reads (below); written by `--write-table` |
| `../vessels/biro_phantom/phantom_arch_all.json` | the default: the 6 training branches (aorta extended to the root's bottom) plus a centerline for each of the phantom's other vessels; provenance in that folder's README; lab-internal |
| `../vessels/biro_phantom/phantom_arch.json` | `--centerline original`: the lab's 6 branches; lab-internal |
| `tests/test_c_arm.py` | angle conventions round-trip against stEVE, routes (both definitions, the overshoot), scores per route and per piece, the vendored tables (the full grid only with `--run-slow`) |

## In training: one angle per target (2026-11-03)

One line in a config (or `--set env.scene.fluoroscopy.image_rot_zx=per_target` on the command line):

```yaml
env: {scene: {fluoroscopy: {image_rot_zx: per_target}}}
```

Then at every episode start, after the target is drawn and before the first observation, the
fluoroscopy's `image_rot_zx` becomes the angle of the target's branch in the table of the scene's phantom
version; the target, which stEVE keeps in the image frame, is put back into the new frame. A target on a
continuation of a target branch (`rcca_rest`, the lab's `v10_rest`) gets that branch's angle; a target on
any other branch is an error. Before the first reset the fluoroscopy sits at the all-5 angle. Everything
that projects into the image (the 2D observations, the two-device task box, `VesselGeometry`'s heading,
`NormalizeTracking2DEpisode`'s range) follows the angle. Details: `per_target.py`'s docstring.

Which table a phantom version reads (by its centerline file; the table records that file's sha256, and a
table made from another version of the file stops the env build with the command that rewrites it):

| `BiroPhantomArch(...)` | Centerline file | Chooser input | Table |
| --- | --- | --- | --- |
| `version: rebuilt_cut` (`two_devices_round1.yaml`) | `rebuilt_cut.json` | `--centerline rebuilt_cut` | `data/angles_rebuilt_cut.json` |
| `version: lab`, or `centerline: all_vessels` (`phantom_all_vessels.yaml`) | `phantom_arch_all.json` | default (`all_vessels`) | `data/angles_all_vessels.json` |
| default `centerline: original` (`phantom_g13.yaml`, `phantom_friction.yaml`, `phantom_real_mass.yaml`) | `phantom_arch.json` | `--centerline original` | none yet: the env build says to run `--centerline original --write-table` |
| `centerline: root_extended`, `version: lab_cut`, `rebuilt`, `spheres`, `spheres_cut` | their own | none | none: add the file to `DATA` in `choose_angle.py`, then `--write-table` |

The angle each target gets (tables written 2026-11-03, whole routes; "Result" sections below). Length kept /
share covered are the target's own route's, at its angle; per piece in the result sections:

| Target | `rebuilt_cut`: C-arm | stEVE | kept / covered | `all_vessels`: C-arm | stEVE | kept / covered |
| --- | --- | --- | --- | --- | --- | --- |
| before the first reset (all 5) | LAO 20°, cranial 0° | [57.7, 0.0] | worst 0.93 / 0.28 | LAO 20°, cranial 0° | [57.7, 0.0] | worst 0.93 / 0.29 |
| bct | LAO 20°, cranial 0° | [57.7, 0.0] | 0.97 / 0.05 | LAO 27.5°, cranial 15° | [61.2, −32.5] | 0.97 / 0.00 |
| rcca (and its continuation) | LAO 15°, cranial 7.5° | [52.0, −12.2] | 0.95 / 0.16 | LAO 15°, cranial 15° | [50.2, −23.8] | 0.93 / 0.15 |
| rsa | LAO 27.5°, cranial 10° | [63.4, −22.8] | 0.94 / 0.13 | LAO 20°, cranial 10° | [56.3, −18.2] | 0.95 / 0.17 |
| lcca | LAO 20°, caudal 10° | [56.3, 18.2] | 0.97 / 0.06 | LAO 25°, caudal 22.5° | [55.2, 42.1] | 0.98 / 0.00 |
| lsa | LAO 32.5°, caudal 25° | [58.5, 54.0] | 0.96 / 0.00 | LAO 27.5°, caudal 22.5° | [57.0, 44.6] | 0.95 / 0.02 |

The two-device task (`two_devices_round1.yaml`, target in the right carotid) with the switch gets the rcca
row of `rebuilt_cut`, [52.0, −12.2]. The config itself still fixes [61.0, −21.0], the all-5 angle with the
routes before 2026-11-03.

A table holds, per row (`all` and each target): `image_rot_zx` (full precision; training uses it as is),
`lao_deg` / `cranial_deg` and the clinical name, and every target's length kept and share covered at that
angle, over its route and per piece (`pieces`: aorta, bct, the branch, each with its length); and
`continuations` (a branch whose first point lies within 0.05 mm of a target branch's last point).
Provenance: `command`, `date`, `centerline` (chooser input, file, sha256, branch count, frame rotation)
and `method` (the route definition, `route: whole` and `route_past_mm: 0`, and every number of the method
below). To write a table again, run the `command` it records (the `--write-table` line above); `tests/test_c_arm.py` checks the vendored tables against today's
centerline files, and with `--run-slow` the `rebuilt_cut` one against a fresh run (about 2 min).

stEVE's `TrackingOnly.tracking3d_space` turns the vessel box by two corners only and is not a valid box at
most of these angles (`rebuilt_cut`: all but lcca and lsa; `all_vessels`: every row), so a single-device
phantom config (stEVE's `Tracking2D` / `Target2D` read it when the env is built) stopped at the build
(gymnasium: "Box all low values must be less than or equal to high"). Fixed on this branch since
2026-11-03 (problem 21 of `docs/simulator_changes.md`, `simulator_modifications/fluoroscopy.py`, brought
over from `harry-repro`): `phantom_all_vessels.yaml` + the switch builds and runs, every episode at its
target branch's angle. The two-device observations never read that box.

## Method

For each of the 5 targets (bct, rcca, rsa, lcca, lsa) the route is the whole way from the aorta (user's
decision, 2026-11-03; in the two-device task the hook is engaged at the trunk's ostium on the arch, which
the right carotid's route left out before): the 60 mm of aorta before the first ostium on the way,
then every vessel on the way up to where the next one leaves it, then the whole target branch.

| Target | Route | Length (`rebuilt_cut`) |
| --- | --- | --- |
| bct | aorta 60 mm + bct | 120 mm |
| rcca | aorta 60 mm + bct + rcca | 157 mm |
| rsa | aorta 60 mm + bct + rsa | 162 mm |
| lcca, lsa | aorta 60 mm + the branch | 119, 117 mm |

Before 2026-11-03 the route was the 60 mm before the target's *own* ostium plus the target branch, so the
right carotid's and subclavian's routes began on the trunk and left out the aorta and the trunk's ostium
(bct, lcca and lsa are the same either way). `--route own_ostium` still gives that definition and the
numbers of the two older result sections, to the bit. For each C-arm angle and each route:

- **length kept** = projected length / true length (1 = not shortened), over the whole route. Tu et al. 2011
  use 1 − this.
- **share covered** = share of the route's length drawn on top of another vessel's lumen while the two are
  more than 5 mm apart in 3D, so a branch touching its parent at the ostium does not count. A closed arch
  (ascending over descending) and crowded ostia both raise it. Same idea as Sun et al. 2021's overlap ratio.
  Over the whole route, weighted by length.
- Both also **per piece** of the route (aorta, bct, the target branch), printed and in the tables, so what
  the whole-route number trades (for example the trunk against the carotid) stays visible. They are not
  used for the choice.

One angle for all 5 targets is scored by the worst target. Every angle in C-arm reach is tried (RAO ≤ 50°,
LAO ≤ 90°, cranial/caudal ≤ 40°, as Tu et al. 2012), each scored by its worst value within ±5° so a C-arm
or phantom set a few degrees off still sees what was planned. Then, as Sun et al. 2021 and Chen & Carroll
2000, in steps rather than a weighted sum:

1. keep the angles whose share covered is within 5 points of the lowest (zero is not reachable for all
   5 targets at once on this phantom);
2. of those, keep the ones within 98 % of the best length kept;
3. of those, take the one closest to straight-on.

Angles are given two ways: as on a real C-arm, and as stEVE's `image_rot_zx`. The C-arm names assume the
phantom's scan frame is the patient's LPS (x left, y posterior, z head); the anatomy agrees (the
ascending aorta lies right-anterior of the descending one, the brachiocephalic trunk is on the right).

## Result, model-wall version, whole routes (2026-11-03; `--centerline rebuilt_cut`, `data/angles_rebuilt_cut.json`)

`rebuilt_cut.json`: 60 vessels, centerlines and radii from the phantom model's own inner wall, right arm cut
off (`../vessels/biro_phantom/README.md`); `two_devices_round1.yaml` runs on it. Per target: length kept /
share covered over its whole route.

| Angle | C-arm | stEVE | bct | rcca | rsa | lcca | lsa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G13, `[20, 5]` | RAO 17.6°, caudal 4.7° | [20, 5] | 0.82 / 0.57 | 0.85 / 0.50 | 0.82 / 0.42 | 0.78 / 0.87 | 0.79 / 0.94 |
| lab, `[25, 0]` | RAO 12.7° | [25, 0] | 0.85 / 0.51 | 0.88 / 0.50 | 0.85 / 0.38 | 0.80 / 0.84 | 0.81 / 0.93 |
| **chosen, all 5** | **LAO 20°, cranial 0°** | **[57.7, 0.0]** | 0.97 / 0.05 | 0.96 / 0.28 | 0.93 / 0.25 | 0.95 / 0.06 | 0.95 / 0.20 |
| chosen, bct alone | LAO 20°, cranial 0° | [57.7, 0.0] | 0.97 / 0.05 | | | | |
| chosen, rcca alone | LAO 15°, cranial 7.5° | [52.0, −12.2] | | 0.95 / 0.16 | | | |
| chosen, rsa alone | LAO 27.5°, cranial 10° | [63.4, −22.8] | | | 0.94 / 0.13 | | |
| chosen, lcca alone | LAO 20°, caudal 10° | [56.3, 18.2] | | | | 0.97 / 0.06 | |
| chosen, lsa alone | LAO 32.5°, caudal 25° | [58.5, 54.0] | | | | | 0.96 / 0.00 |

Each target's route piece by piece at its own angle (length kept / share covered, piece length):

| Target | aorta | bct | the target branch |
| --- | --- | --- | --- |
| bct | 0.96 / 0.00 (60 mm) | 0.98 / 0.10 (60 mm) | — |
| rcca | 0.94 / 0.10 (60 mm) | 0.99 / 0.01 (60 mm) | 0.90 / **0.49** (38 mm) |
| rsa | 0.97 / 0.00 (60 mm) | 0.97 / 0.00 (60 mm) | 0.85 / **0.51** (42 mm) |
| lcca | 0.95 / 0.00 (60 mm) | — | 0.99 / 0.13 (59 mm) |
| lsa | 1.00 / 0.00 (59 mm) | — | 0.93 / 0.00 (59 mm) |

- bct, lcca and lsa: same routes and angles as before 2026-11-03.
- The right carotid moves from RAO 20°, cranial 12.5° ([17.2, −13.1]) to LAO 15°, cranial 7.5°. What the
  whole-route number trades, piece by piece (aorta / trunk / carotid covered):

  | Angle | C-arm | stEVE | whole route | aorta | bct | rcca |
  | --- | --- | --- | --- | --- | --- | --- |
  | rcca, whole route (now) | LAO 15°, cranial 7.5° | [52.0, −12.2] | 0.16 | 0.10 | 0.01 | 0.49 |
  | rcca alone, routes before | RAO 20°, cranial 12.5° | [17.2, −13.1] | 0.44 | 1.00 | 0.00 | 0.26 |
  | bct + rcca jointly (work log 2026-11-02 23:53) | LAO 10°, cranial 10° | [46.7, −14.7] | 0.25 | 0.45 | 0.00 | 0.33 |
  | all 5, routes before (`two_devices_round1.yaml`) | LAO 25°, cranial 10° | [61.0, −21.0] | 0.20 | 0.00 | 0.00 | 0.85 |
  | all 5 (now) | LAO 20°, cranial 0° | [57.7, 0.0] | 0.28 | 0.00 | 0.11 | 1.00 |

  The share is weighted by length, so the 120 mm of aorta and trunk outweigh the 38 mm carotid: at the
  chosen angle the carotid itself is half covered. The same holds for the right subclavian (51 %) and,
  at the all-5 angle, for both right-side branches (rcca 100 %, rsa 81 %).
- The all-5 angle moves from LAO 25°, cranial 10° to LAO 20°, cranial 0° (= bct alone). With whole routes,
  LAO 25°, cranial 10° itself has its worst route (lsa) 21 % covered, but within ±5° of it a route is up
  to 41 % covered; LAO 20°, cranial 0° stays at 29 % within ±5° (the lowest such value anywhere is 26 %).

## Result, all vessels, whole routes (2026-11-03; default input, `data/angles_all_vessels.json`)

| Angle | C-arm | stEVE | bct | rcca | rsa | lcca | lsa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G13, `[20, 5]` | RAO 17.6°, caudal 4.7° | [20, 5] | 0.84 / 0.49 | 0.87 / 0.47 | 0.84 / 0.37 | 0.79 / 0.82 | 0.79 / 0.91 |
| lab, `[25, 0]` | RAO 12.7° | [25, 0] | 0.87 / 0.43 | 0.90 / 0.46 | 0.87 / 0.32 | 0.82 / 0.69 | 0.81 / 0.89 |
| **chosen, all 5** | **LAO 20°, cranial 0°** | **[57.7, 0.0]** | 0.97 / 0.08 | 0.96 / 0.29 | 0.93 / 0.24 | 0.96 / 0.12 | 0.94 / 0.18 |
| chosen, bct alone | LAO 27.5°, cranial 15° | [61.2, −32.5] | 0.97 / 0.00 | | | | |
| chosen, rcca alone | LAO 15°, cranial 15° | [50.2, −23.8] | | 0.93 / 0.15 | | | |
| chosen, rsa alone | LAO 20°, cranial 10° | [56.3, −18.2] | | | 0.95 / 0.17 | | |
| chosen, lcca alone | LAO 25°, caudal 22.5° | [55.2, 42.1] | | | | 0.98 / 0.00 | |
| chosen, lsa alone | LAO 27.5°, caudal 22.5° | [57.0, 44.6] | | | | | 0.95 / 0.02 |

| Target | aorta | bct | the target branch |
| --- | --- | --- | --- |
| bct | 0.97 / 0.00 (60 mm) | 0.97 / 0.00 (62 mm) | — |
| rcca | 0.96 / 0.18 (60 mm) | 0.98 / 0.00 (62 mm) | 0.82 / 0.36 (38 mm) |
| rsa | 0.97 / 0.00 (60 mm) | 0.98 / 0.07 (62 mm) | 0.87 / 0.56 (42 mm) |
| lcca | 0.98 / 0.00 (60 mm) | — | 0.97 / 0.00 (59 mm) |
| lsa | 0.99 / 0.00 (59 mm) | — | 0.91 / 0.03 (59 mm) |

The lab's 6 branches only (`--centerline original`, printout only, no table): all 5 LAO 20°, cranial 0°
[57.7, 0.0]; bct LAO 27.5°, cranial 15° [61.2, −32.5]; rcca LAO 15°, caudal 2.5° [52.6, 4.1]; rsa
LAO 12.5°, cranial 12.5° [48.6, −19.1]; lcca LAO 25°, caudal 22.5° [55.2, 42.1]; lsa LAO 27.5°, caudal
22.5° [57.0, 44.6].

## 30 mm past the trunk's ostium (checked 2026-11-03, not used)

In the two-device task the hooked catheter goes at least 30 mm past the trunk's ostium into the ascending
aorta before it hooks back. `--past-mm 30` puts that piece of aorta (29 mm, to the centerline point) on
every route as well, after the first ostium on the way. Angles that move (`rebuilt_cut`; covered = share
covered of the 60-mm-before route and per piece, at each angle):

| Target | 60 mm before (used) | + 30 mm past | at "60 before": route, branch, past piece | at "+ 30 past": route, branch, past piece |
| --- | --- | --- | --- | --- |
| all 5 | LAO 20°, cranial 0° | the same | | |
| bct | LAO 20°, cranial 0° [57.7, 0.0] | LAO 17.5°, caudal 2.5° [55.1, 4.4] | 0.05, bct 0.10, past 0.00 | 0.06, bct 0.11, past 0.00 |
| rcca | LAO 15°, cranial 7.5° [52.0, −12.2] | LAO 15°, cranial 2.5° [52.6, −4.1] | 0.16, rcca 0.49, past 0.00 | 0.24, rcca **0.90**, past 0.00 |
| rsa | LAO 27.5°, cranial 10° | the same | | |
| lcca | LAO 20°, caudal 10° [56.3, 18.2] | LAO 20°, caudal 7.5° [56.9, 13.8] | 0.06, lcca 0.13, past 0.00 | 0.06, lcca 0.13, past 0.00 |
| lsa | LAO 32.5°, caudal 25° [58.5, 54.0] | LAO 32.5°, caudal 15° [65.3, 38.3] | 0.00, lsa 0.00, past 0.13 | 0.00, lsa 0.00, past 0.00 |

The piece past the ostium is uncovered at almost every angle, so adding it mostly thins out the covered
share and lets the choice drift towards straight-on; for the right carotid that drift covers 90 % of the
carotid. Not in favour of including it, so the tables use the 60-mm-before route. On `all_vessels` the
past piece moves all 5 to LAO 17.5°, caudal 2.5°, rcca to LAO 12.5°, cranial 7.5°, rsa to LAO 17.5°,
cranial 12.5°, bct, lcca and lsa by 2.5–15°; the past piece is uncovered there too.

## Result, model-wall version, routes before 2026-11-03 (2026-10-30; `--centerline rebuilt_cut --route own_ostium` reproduces it)

`rebuilt_cut.json`: 60 vessels, centerlines and radii from the phantom model's own inner wall, right arm cut
off (provenance in `../vessels/biro_phantom/README.md`). This is the version `two_devices_round1.yaml` runs on.
Rerun 2026-11-02: the all-5 and rcca-alone angles are the ones the work log recorded on 2026-10-30.
Per target: length kept / share covered.

| Angle | C-arm | stEVE | bct | rcca | rsa | lcca | lsa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G13, `[20, 5]` | RAO 18°, caudal 5° | [20, 5] | 0.82 / 0.57 | 0.96 / 0.19 | 0.91 / 0.08 | 0.78 / 0.87 | 0.79 / 0.94 |
| lab, `[25, 0]` | RAO 13° | [25, 0] | 0.85 / 0.51 | 0.97 / 0.19 | 0.92 / 0.01 | 0.80 / 0.84 | 0.81 / 0.93 |
| **chosen, all 5** (`two_devices_round1.yaml`) | **LAO 25°, cranial 10°** | **[61.0, −21.0]** | 0.97 / 0.00 | 0.94 / 0.32 | 0.93 / 0.23 | 0.94 / 0.05 | 0.94 / 0.21 |
| chosen, bct alone | LAO 20°, cranial 0° | [57.7, 0.0] | 0.97 / 0.05 | | | | |
| chosen, rcca alone | RAO 20°, cranial 12.5° | [17.2, −13.1] | | 0.96 / 0.10 | | | |
| chosen, rsa alone | RAO 10°, cranial 2.5° | [27.6, −2.8] | | | 0.93 / 0.00 | | |
| chosen, lcca alone | LAO 20°, caudal 10° | [56.3, 18.2] | | | | 0.97 / 0.06 | |
| chosen, lsa alone | LAO 32.5°, caudal 25° | [58.5, 54.0] | | | | | 0.96 / 0.00 |

## Result, all vessels, routes before 2026-11-03 (2026-10-28; `--route own_ostium`)

Per target: length kept / share covered.

| Angle | C-arm | stEVE | bct | rcca | rsa | lcca | lsa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G13, `[20, 5]` | RAO 18°, caudal 5° | [20, 5] | 0.84 / 0.49 | 0.96 / 0.15 | 0.91 / 0.00 | 0.79 / 0.82 | 0.79 / 0.91 |
| lab, `[25, 0]` | RAO 13° | [25, 0] | 0.87 / 0.43 | 0.97 / 0.21 | 0.93 / 0.00 | 0.82 / 0.69 | 0.81 / 0.89 |
| **chosen, all 5** | **LAO 30°, cranial 17.5°** | **[61.9, −39.7]** | 0.97 / 0.00 | 0.89 / 0.36 | 0.93 / 0.26 | 0.92 / 0.11 | 0.91 / 0.19 |
| chosen, bct alone | LAO 27.5°, cranial 15° | [61.2, −32.5] | 0.97 / 0.00 | | | | |
| chosen, rcca alone | RAO 22.5°, cranial 0° | [15.2, 0.0] | | 0.97 / 0.10 | | | |
| chosen, rsa alone | RAO 10°, cranial 2.5° | [27.6, −2.8] | | | 0.93 / 0.00 | | |
| chosen, lcca alone | LAO 25°, caudal 22.5° | [55.2, 42.1] | | | | 0.98 / 0.00 | |
| chosen, lsa alone | LAO 27.5°, caudal 22.5° | [57.0, 44.6] | | | | | 0.95 / 0.02 |

- Against the lab's 6 only (below): bct, lcca and lsa keep their angles. The right carotid moves from
  RAO 7.5°, cranial 5° to RAO 22.5°, cranial 0°, the right subclavian to RAO 10°, cranial 2.5°, and the
  all-5 angle from LAO 20°, cranial 2.5° to LAO 30°, cranial 17.5°.
- The vessel that moves the right carotid is `v18`: 48 mm long, radius about 2.6 mm, leaving the right
  carotid's continuation (`v10_rest`) 61 mm past the carotid's opening and running down and forward to a
  dead end, 20–35 mm in front of the carotid. At RAO 7.5°, cranial 5° it lies over 33 % of the carotid's
  route (5–37 mm past the opening); at RAO 22.5°, cranial 0° over 10 % (28–37 mm), and no reachable angle
  clears it fully.

## Result, the lab's 6 branches only, routes before 2026-11-03 (2026-10-26; `--centerline original --route own_ostium` reproduces it)

Per target: length kept / share covered.

| Angle | C-arm | stEVE | bct | rcca | rsa | lcca | lsa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G13, `[20, 5]` | RAO 18°, caudal 5° | [20, 5] | 0.84 / 0.49 | 0.96 / 0.00 | 0.91 / 0.00 | 0.79 / 0.81 | 0.79 / 0.87 |
| lab, `[25, 0]` | RAO 13° | [25, 0] | 0.87 / 0.43 | 0.97 / 0.00 | 0.93 / 0.00 | 0.82 / 0.69 | 0.81 / 0.82 |
| **chosen, all 5** | **LAO 20°, cranial 2.5°** | **[57.6, −4.7]** | 0.97 / 0.07 | 0.94 / 0.23 | 0.92 / 0.24 | 0.95 / 0.12 | 0.93 / 0.16 |
| chosen, bct alone | LAO 27.5°, cranial 15° | [61.2, −32.5] | 0.97 / 0.00 | | | | |
| chosen, rcca or rsa alone | RAO 7.5°, cranial 5° | [30.0, −5.8] | | 0.96 / 0.00 | 0.93 / 0.00 | | |
| chosen, lcca alone | LAO 25°, caudal 22.5° | [55.2, 42.1] | | | | 0.98 / 0.00 | |
| chosen, lsa alone | LAO 27.5°, caudal 22.5° | [57.0, 44.6] | | | | | 0.95 / 0.00 |

- Stable: changing one tuning number at a time (5 mm → 3 or 8 mm; 60 mm → 40 or 90 mm; 5 points → 3
  or 10; 98 % → 95 %) keeps the all-5 angle within LAO 12.5–25°, caudal 2.5° to cranial 7.5°.
- Facing the arch plane alone (the plane fitted to the aorta ±60 mm around its top) would be LAO 38°,
  caudal 13–23°, inside the clinical LAO 30–60° for arch angiography. The all-5 angle sits lower because
  the right-side branches want RAO and the left ones LAO; each target alone can be seen with nothing on top.

## Not covered

- With the default and `original` data, the other vessels' centerlines and radii are the lab's skeleton
  (radius at least 1 mm), not measured on the phantom model's own wall (`rebuilt_cut` is); vessels count
  as tubes around their centerlines.
- The whole-route share is weighted by length, so a short target branch at the end of a long route can be
  drawn largely over other vessels while the route as a whole is little covered (the right carotid at its
  own angle, 49 % of it; the right-side branches at the all-5 angle, 81–100 %). The per-piece numbers show it; the choice does not use them (the user's
  decision of 2026-11-03 scores the route as one).
- One angle serving several targets: the literature has no method for it (each work picks one view per
  vessel segment); scoring by the worst target is our own choice.
- Naming in the stEVE papers disagrees: Karstensen et al. 2023 call their view LAO, the stEVE paper an RAO
  plane. On this phantom, under the LPS assumption above, `[20, 5]` and `[25, 0]` come out as RAO.

## References

- Tu S, et al. 2011. EuroIntervention 7(1):112–120. doi:10.4244/EIJV7I1A19 — foreshortening formula.
- Tu S, et al. 2012. Int J Cardiovasc Imaging 28(7):1617–1625. doi:10.1007/s10554-011-9996-x — C-arm reach.
- Sun, Yang, Chen, Shu 2021. BMC Med Imaging 21:141. doi:10.1186/s12880-021-00676-3 — arch view for TEVAR:
  overlap of lcca/lsa as a hard condition first, then least shortening, then the smaller angle.
- Chen & Carroll 2000. IEEE TMI 19(4):318–336. doi:10.1109/42.848183 — least shortening first, then least overlap.
- Wang L, et al. 2018. J Thorac Dis 10(4):2100–2107. doi:10.21037/jtd.2018.03.72 — CT-measured arch LAO, 55–70°.
- Karstensen et al. 2023. IJCARS 18(9):1735–1744. doi:10.1007/s11548-023-02938-7 — stEVE-line view, no optimisation.
- The lab's `000-app/engine/sim/carm.py` `working_view`: looks along the route's least-spread direction
  (least shortening of one route), then ±35° / ±25° random turns for training.
