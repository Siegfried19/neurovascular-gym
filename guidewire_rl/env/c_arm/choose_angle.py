"""Pick the C-arm angle for the guidewire task on the lab phantom.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle [--centerline original] [--figure out.png] [--write-table]

Tries every C-arm angle in reach and scores how well the routes to the 5 target branches show in
the image; prints the chosen angle (clinical LAO/RAO + cranial/caudal, and stEVE's image_rot_zx).
A route is the whole way from the aorta (since 2026-11-03): 60 mm of aorta before the first ostium
on the way, every vessel on the way, the target branch. `--route own_ostium` is the definition before
that (60 mm of the parent vessel before the target's own ostium + the target branch); `--past-mm 30`
also puts the aorta 30 mm past the first ostium on the route (a hooked catheter's overshoot).
`--figure` also draws the score map and the phantom at the current and the chosen angle (one-off
figures go to the session scratchpad, not the repository). Method and references: README.md.

The vessels are phantom_arch_all.json from ../vessels/biro_phantom/ by default: the 6 training branches
plus a centerline for every other vessel of the phantom, so a route drawn over any of them counts.
`--centerline original` reads the lab's 6-branch phantom_arch.json instead; `--centerline rebuilt_cut` the
model-wall version rebuilt_cut.json (its centerlines and junctions all from the phantom model, right arm cut off).

`--write-table` also writes the chosen angles as a JSON table (default ``data/angles_<centerline>.json``
next to this file), which training reads for ``env.scene.fluoroscopy.image_rot_zx: per_target``
(per_target.py; training never runs this search).

The image is stEVE's: a parallel projection, so only the direction of the beam matters.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

PHANTOM = Path(__file__).resolve().parents[1] / "vessels" / "biro_phantom"
DATA = {"all_vessels": PHANTOM / "phantom_arch_all.json", "original": PHANTOM / "phantom_arch.json",
        "rebuilt_cut": PHANTOM / "rebuilt_cut.json"}   # the model-wall version (biro_phantom/README.md)
TASK = ["aorta", "bct", "rsa", "rcca", "lcca", "lsa"]  # the training branches; routes run along these only
TARGETS = ["bct", "rcca", "rsa", "lcca", "lsa"]          # configs/phantom_g13.yaml (g13's list, no "co")

ROUTE = "whole"          # route = ROUTE_BEFORE_MM of aorta before the first ostium on the way + every vessel
                         # on the way + the target branch; "own_ostium" (before 2026-11-03): ROUTE_BEFORE_MM of
                         # the parent vessel(s) before the target's own ostium + the target branch
ROUTES = ("whole", "own_ostium")
ROUTE_BEFORE_MM = 60.0
ROUTE_PAST_MM = 0.0      # aorta past the first ostium (on, away from the inlet) also on the route; 30 checked
APART_MM = 5.0          # drawn on top of another vessel only counts if the two are this far apart in 3D
STEP_DEG = 2.5
LAO_RANGE = (-50.0, 90.0)   # RAO <= 50, LAO <= 90, cranial/caudal <= 40: C-arm reach as in Tu et al. 2012
CRAN_RANGE = (-40.0, 40.0)
ROBUST_DEG = 5.0         # score an angle by the worst value within +-5 deg (C-arm or phantom set a bit off)
COVER_TOL = 0.05         # step 1: keep angles whose covered share is within 5 points of the lowest
KEEP_FRAC = 0.98         # step 2: of those, keep the ones within 98 % of the best length kept
                         # step 3: of those, take the one closest to straight-on (AP), as Sun et al. 2021

TABLE_DIR = Path(__file__).with_name("data")   # the tables training reads (--write-table)
CONTINUES_MM = 0.05      # a branch starting this close to a target branch's last point continues it


# ------------------------------------------------------------------ vessels ----
def arclen(p):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _drop_repeats(c, r):
    keep = np.r_[True, np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-6]
    return c[keep], r[keep]


def load_phantom(path=DATA["all_vessels"]):
    """Centerlines and radii in the training (arch) frame, and the angle that frame is turned by
    about z relative to the phantom's own (scan) frame: p_arch = Rz(theta) (p_scan - origin)."""
    spec = json.loads(Path(path).read_text())
    branches = {k: _drop_repeats(np.asarray(v["coordinates"], float), np.asarray(v["radii"], float))
                for k, v in spec["branches"].items()}
    rotation = spec["rotation_z_deg"] if "rotation_z_deg" in spec else spec["frame_transform"]["rotation_z_deg"]
    return branches, np.radians(rotation)


def parent_of(branches, name):
    """(parent branch, index on it) where this branch leaves; a sibling that starts at the same
    point is not a parent."""
    p0, best = branches[name][0][0], None
    for k in TASK:
        c = branches[k][0]
        if k == name:
            continue
        i = int(np.argmin(np.linalg.norm(c - p0, axis=1)))
        if i == 0 and k != "aorta":
            continue
        dist = np.linalg.norm(c[i] - p0)
        if best is None or dist < best[0]:
            best = (dist, k, i)
    return best[1], best[2]


def chain_to(branches, target):
    """The vessels on the way from the aorta to `target`: aorta, (bct), target."""
    chain = [target]
    while chain[-1] != "aorta":
        chain.append(parent_of(branches, chain[-1])[0])
    return chain[::-1]


def route(branches, target, before_mm=ROUTE_BEFORE_MM, kind=ROUTE, past_mm=ROUTE_PAST_MM):
    """The route to `target` as polylines [(points, radii, piece of each point)], a piece being the vessel
    the point lies on. The first polyline runs along the way:

    - ``whole``: the aorta's last `before_mm` before the first ostium on the way, every vessel on the way up
      to where the next one leaves it, the whole target branch (rcca: aorta 60 mm + bct + rcca; bct, lcca,
      lsa: aorta 60 mm + the branch);
    - ``own_ostium`` (the definition before 2026-11-03): the last `before_mm` of the vessels on the way
      before the target's own ostium, the whole target branch (rcca: the last 60 mm of bct + rcca).

    `past_mm` > 0 (``whole`` only) adds a second polyline, piece ``aorta_past``: the aorta from that first
    ostium on for `past_mm`, away from the inlet (a hooked catheter goes this far past the ostium before
    it hooks back into it)."""
    if kind not in ROUTES:
        raise ValueError(f"route kind must be one of {ROUTES}, got {kind!r}")
    chain = chain_to(branches, target)
    pts, rad, names = [], [], []
    for a, b in zip(chain[:-1], chain[1:]):
        _, i = parent_of(branches, b)
        pts.append(branches[a][0][: i + 1])
        rad.append(branches[a][1][: i + 1])
        names.append(np.full(i + 1, a))
    if kind == "whole":
        m = arclen(pts[0]) >= arclen(pts[0])[-1] - before_mm
        pts[0], rad[0], names[0] = pts[0][m], rad[0][m], names[0][m]
    pts, rad, names = np.concatenate(pts), np.concatenate(rad), np.concatenate(names)
    if kind == "own_ostium":
        m = arclen(pts) >= arclen(pts)[-1] - before_mm
        pts, rad, names = pts[m], rad[m], names[m]
    c = np.concatenate([pts, branches[target][0]])
    keep = np.r_[True, np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-6]      # as _drop_repeats
    lines = [(c[keep], np.concatenate([rad, branches[target][1]])[keep],
              np.concatenate([names, np.full(len(branches[target][0]), target)])[keep])]
    if past_mm > 0:
        if kind != "whole":
            raise ValueError("past_mm is for the whole route only")
        _, i = parent_of(branches, chain[1])
        c, r = branches["aorta"]
        s = arclen(c)
        m = (s >= s[i]) & (s <= s[i] + past_mm)
        lines.append((c[m], r[m], np.full(int(m.sum()), "aorta_past")))
    return lines


# ------------------------------------------------------------------- angles ----
def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def clinical_to_beam(lao_deg, cran_deg, theta):
    """Beam axis in the arch frame for a C-arm at LAO (+) / RAO (-) and cranial (+) / caudal (-).
    Assumes the phantom's scan frame is the patient's LPS (x left, y posterior, z head), which the
    anatomy agrees with: the ascending aorta lies right-anterior of the descending one."""
    a, b = np.radians(lao_deg), np.radians(cran_deg)
    detector = np.array([np.sin(a) * np.cos(b), -np.cos(a) * np.cos(b), np.sin(b)])
    return _rz(theta) @ detector


def beam_to_clinical(d, theta):
    u = _rz(theta).T @ d
    if u[1] > 0:                                          # same line; put the detector in front
        u = -u
    return float(np.degrees(np.arctan2(u[0], -u[1]))), float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))


def steve_to_beam(rot_z, rot_x):
    """stEVE image_rot_zx: image = drop y of Rz(-rot_z) Rx(-rot_x) p, so the beam is Rx(rot_x) Rz(rot_z) e_y."""
    z, x = np.radians(rot_z), np.radians(rot_x)
    return np.array([-np.sin(z), np.cos(z) * np.cos(x), np.cos(z) * np.sin(x)])


def beam_to_steve(d):
    d = d / np.linalg.norm(d)
    rot_x = np.degrees(np.arctan2(d[2], d[1]))
    if abs(rot_x) > 90:                                   # same line, other sign
        return beam_to_steve(-d)
    return float(np.degrees(np.arctan2(-d[0], np.hypot(d[1], d[2])))), float(rot_x)


def steve_image(points, rot_z, rot_x):
    from eve.util.coordtransform import _get_rot_matrix
    return np.delete((_get_rot_matrix((rot_z, rot_x)) @ points.T).T, 1, -1)


def _plane(d):
    a = np.cross(d, [0.0, 0.0, 1.0])
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-9 else np.array([1.0, 0.0, 0.0])
    return np.stack([a, np.cross(d, a)])


# ------------------------------------------------------------------ scoring ----
_ROUTES = {}


def _route_parts(branches, target, apart_mm=APART_MM, **route_kw):
    """What scoring a route needs and does not depend on the angle, once per route: its points, their
    weights (half the segments on either side) and pieces; its segments, their lengths and pieces (the
    end point's); the 3D test (which other vessel points lie more than `apart_mm` apart)."""
    key = (id(branches), target, apart_mm, tuple(sorted(route_kw.items())))
    if key not in _ROUTES:
        lines = route(branches, target, **route_kw)
        seg = [np.diff(p, axis=0) for p, _, _ in lines]
        L = [np.linalg.norm(s, axis=1) for s in seg]
        p = np.concatenate([p for p, _, _ in lines])
        r = np.concatenate([r for _, r, _ in lines])
        q = np.concatenate([c for c, _ in branches.values()])
        s = np.concatenate([rr for _, rr in branches.values()])
        reach = r[:, None] + s[None]
        far = np.linalg.norm(p[:, None] - q[None], axis=2) > reach + apart_mm
        cols = far.any(0)
        _ROUTES[key] = dict(
            p=p, piece=np.concatenate([n for _, _, n in lines]),
            w=np.concatenate([np.r_[l, 0] / 2 + np.r_[0, l] / 2 for l in L]),
            seg=np.concatenate(seg), L=np.concatenate(L), seg_piece=np.concatenate([n[1:] for _, _, n in lines]),
            q=q[cols], reach=reach[:, cols], far=far[:, cols])
    return _ROUTES[key]


def _scores(parts, d):
    """Per segment its projected length, per point whether it is drawn on top of another vessel."""
    seg, L = parts["seg"], parts["L"]
    projected = L * np.sqrt(np.clip(1 - ((seg / L[:, None]) @ d) ** 2, 0, 1))
    plane = _plane(d)
    q, reach, far, p = parts["q"], parts["reach"], parts["far"], parts["p"]
    on_top = ((np.linalg.norm((p @ plane.T)[:, None] - (q @ plane.T)[None], axis=2) < reach) & far).any(1)
    return projected, on_top


def score_route(branches, target, d, apart_mm=APART_MM, **route_kw):
    """(length kept, share covered) of the route to `target` (``route``) for beam axis d.
    length kept: projected length / true length (1 = no foreshortening; Tu et al. 2011).
    share covered: share of the route's length drawn on top of another vessel's lumen while the two
    are more than `apart_mm` apart in 3D (closed arch, crowded ostia; Sun et al. 2021). Every vessel in
    `branches` counts, not only the training branches."""
    parts = _route_parts(branches, target, apart_mm, **route_kw)
    projected, on_top = _scores(parts, d)
    w = parts["w"]
    return float(projected.sum() / parts["L"].sum()), float((w * on_top).sum() / w.sum())


def score_pieces(branches, target, d, apart_mm=APART_MM, **route_kw):
    """The same two numbers per piece of the route (aorta, bct, the target, ...; in route order), with
    the piece's length: {piece: {length_mm, length_kept, share_covered}}. A segment counts to the piece of
    its end point, a point's weight (half the segments on either side) to its own piece."""
    parts = _route_parts(branches, target, apart_mm, **route_kw)
    projected, on_top = _scores(parts, d)
    out = {}
    for name in dict.fromkeys(parts["piece"]):
        s, k = parts["seg_piece"] == name, parts["piece"] == name
        if not s.any():                                    # a single point where the route starts
            continue
        w = parts["w"][k]
        out[str(name)] = {"length_mm": float(parts["L"][s].sum()),
                          "length_kept": float(projected[s].sum() / parts["L"][s].sum()),
                          "share_covered": float((w * on_top[k]).sum() / w.sum())}
    return out


def grid(branches, theta, targets=TARGETS, **route_kw):
    lao = np.arange(LAO_RANGE[0], LAO_RANGE[1] + 1e-9, STEP_DEG)
    cran = np.arange(CRAN_RANGE[0], CRAN_RANGE[1] + 1e-9, STEP_DEG)
    res = np.array([[[score_route(branches, t, clinical_to_beam(a, b, theta), **route_kw) for t in targets]
                     for a in lao] for b in cran])          # (cran, lao, target, [kept, covered])
    return lao, cran, res


def select(kept, covered, lao, cran):
    """Index (i_cran, j_lao) of the chosen angle; kept/covered are already the worst over targets."""
    k = int(round(ROBUST_DEG / STEP_DEG)) * 2 + 1
    covered_r = maximum_filter(covered, size=k, mode="nearest")
    kept_r = minimum_filter(kept, size=k, mode="nearest")
    ok = covered_r <= covered_r.min() + COVER_TOL
    ok &= kept_r >= KEEP_FRAC * kept_r[ok].max()
    a, b = np.meshgrid(np.radians(lao), np.radians(cran))
    off_ap = np.degrees(np.arccos(np.cos(a) * np.cos(b)))
    return np.unravel_index(np.argmin(np.where(ok, off_ap, np.inf)), kept.shape)


# --------------------------------------------------------------------- main ----
def _check_conventions():
    from eve.util.coordtransform import _get_rot_matrix
    for rz, rx in [(20, 5), (25, 0), (-40, 12)]:
        assert np.allclose(steve_to_beam(rz, rx), _get_rot_matrix((rz, rx)).T @ [0, 1, 0])
        assert np.allclose(beam_to_steve(steve_to_beam(rz, rx)), (rz, rx))
    for lao, cran in [(30, 10), (-20, -15)]:
        assert np.allclose(beam_to_clinical(clinical_to_beam(lao, cran, 0.6), 0.6), (lao, cran))


def choose(branches, theta, **route_kw):
    """The chosen angles as clinical (LAO, cranial) degrees: ``"all"`` for the 5 targets together
    (each angle scored by its worst target), then each target alone; and the grid for the figure.
    `route_kw` (``kind``, ``past_mm``) picks the route definition (``route``)."""
    lao, cran, res = grid(branches, theta, **route_kw)
    kept, covered = res[..., 0], res[..., 1]
    i, j = select(kept.min(-1), covered.max(-1), lao, cran)
    chosen = {"all": (float(lao[j]), float(cran[i]))}
    for k, t in enumerate(TARGETS):
        i2, j2 = select(kept[..., k], covered[..., k], lao, cran)
        chosen[t] = (float(lao[j2]), float(cran[i2]))
    return chosen, (lao, cran, res)


def clinical_words(lao_deg, cran_deg):
    a, b = round(lao_deg, 1) + 0.0, round(cran_deg, 1) + 0.0        # no "-0.0"
    return (f"LAO {a:.1f}" if a >= 0 else f"RAO {-a:.1f}",
            f"cranial {b:.1f}" if b >= 0 else f"caudal {-b:.1f}")


def describe(branches, theta, lao_deg, cran_deg, **route_kw):
    """One angle as the table holds it: stEVE's image_rot_zx, the clinical angle, and every target's
    length kept / share covered there, over its whole route and per piece of it."""
    d = clinical_to_beam(lao_deg, cran_deg, theta)
    rz, rx = beam_to_steve(d)
    scores = {}
    for t in TARGETS:
        kept, covered = score_route(branches, t, d, **route_kw)
        scores[t] = {"length_kept": kept, "share_covered": covered,
                     "pieces": score_pieces(branches, t, d, **route_kw)}
    return {"image_rot_zx": [rz, rx], "lao_deg": float(lao_deg), "cranial_deg": float(cran_deg),
            "clinical": ", ".join(clinical_words(lao_deg, cran_deg)), "scores": scores}


def continuations(branches):
    """{branch: target} for every branch that carries a target branch on past its last point (its
    first point within CONTINUES_MM of it): the rebuilt phantom's ``rcca_rest`` and the like, the
    lab's ``v10_rest`` and the like. A target drawn there gets that target's angle."""
    out = {}
    for t in TARGETS:
        end = branches[t][0][-1]
        for name, (c, _) in branches.items():
            if name not in TASK and np.linalg.norm(c[0] - end) < CONTINUES_MM:
                out[name] = t
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_path(which):
    return TABLE_DIR / f"angles_{which}.json"


def _route_args(kind=ROUTE, past_mm=ROUTE_PAST_MM):
    """The command-line options for a route definition other than the default."""
    return (f" --route {kind}" if kind != ROUTE else "") + (f" --past-mm {past_mm:g}" if past_mm != ROUTE_PAST_MM else "")


def command(which, **route_kw):
    return ("conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.c_arm.choose_angle "
            f"--centerline {which}{_route_args(**route_kw)} --write-table")


def make_table(which, branches, theta, chosen, **route_kw):
    """What --write-table writes (README.md, "In training")."""
    path = DATA[which]
    kind, past_mm = route_kw.get("kind", ROUTE), route_kw.get("past_mm", ROUTE_PAST_MM)
    return {
        "what": "C-arm angle per target branch (scored over the target's route; method.route) and for all 5 "
                "targets together, from guidewire_rl/env/c_arm/choose_angle.py (method: README.md next to it); "
                "training reads it for env.scene.fluoroscopy.image_rot_zx: per_target (per_target.py)",
        "command": command(which, **route_kw),
        "date": datetime.date.today().isoformat(),
        "centerline": {"input": which, "file": path.name, "sha256": sha256(path),
                       "branches": len(branches), "rotation_z_deg": round(float(np.degrees(theta)), 6)},
        "method": {"route": kind, "route_before_mm": ROUTE_BEFORE_MM, "route_past_mm": past_mm,
                   "apart_mm": APART_MM, "step_deg": STEP_DEG,
                   "lao_range_deg": list(LAO_RANGE), "cranial_range_deg": list(CRAN_RANGE),
                   "robust_deg": ROBUST_DEG, "cover_tol": COVER_TOL, "keep_frac": KEEP_FRAC,
                   "route_branches": TASK, "targets": TARGETS, "continues_mm": CONTINUES_MM},
        "conventions": "image_rot_zx: stEVE's [rot_z, rot_x], degrees; lao_deg: LAO + / RAO -; "
                       "cranial_deg: cranial + / caudal -; the clinical names assume the phantom's scan "
                       "frame is the patient's LPS (README.md, Method)",
        "continuations": continuations(branches),
        "angles": {key: describe(branches, theta, *chosen[key], **route_kw) for key in ["all", *TARGETS]},
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--centerline", choices=list(DATA), default="all_vessels")
    ap.add_argument("--route", choices=ROUTES, default=ROUTE,
                    help="whole: from the aorta (default); own_ostium: from the target's own ostium (before 2026-11-03)")
    ap.add_argument("--past-mm", type=float, default=ROUTE_PAST_MM,
                    help="also put this much aorta past the first ostium on the route (whole route only)")
    ap.add_argument("--figure", type=Path, default=None, help="also draw the score map and the two images here")
    ap.add_argument("--write-table", nargs="?", const=True, default=None, type=Path, metavar="PATH",
                    help="also write the chosen angles as a JSON table (default: data/angles_<centerline>.json)")
    args = ap.parse_args()
    which = args.centerline
    route_kw = {"kind": args.route, "past_mm": args.past_mm}
    if args.write_table is True and _route_args(**route_kw):
        ap.error("--route / --past-mm other than the default: give --write-table a PATH; "
                 "data/ holds the tables training reads")
    _check_conventions()
    branches, theta = load_phantom(DATA[which])
    print(f"vessels: {DATA[which].name}, {len(branches)} branches; route: {args.route}"
          + (f", aorta {args.past_mm:g} mm past the first ostium" if args.past_mm else ""))
    chosen, (lao, cran, res) = choose(branches, theta, **route_kw)

    def row(name, a, b, own=None):
        d = clinical_to_beam(a, b, theta)
        sc = np.array([score_route(branches, t, d, **route_kw) for t in TARGETS])
        rz, rx = beam_to_steve(d)
        side, tilt = clinical_words(a, b)
        print(f"{name:<22} {side:<10} {tilt:<13} [{rz:6.1f}, {rx:6.1f}]   "
              + "  ".join(f"{t} {k:.2f}/{c:.2f}" for t, (k, c) in zip(TARGETS, sc)))
        if own is not None:                                 # the target's route, piece by piece
            print(f"{'':<22} {own} by piece: " + "  ".join(
                f"{n} {v['length_kept']:.2f}/{v['share_covered']:.2f} ({v['length_mm']:.0f} mm)"
                for n, v in score_pieces(branches, own, d, **route_kw).items()))

    print("angle                  C-arm                   stEVE image_rot_zx   per target: length kept / share covered")
    for name, (rz, rx) in [("G13 [20, 5]", (20, 5)), ("lab [25, 0]", (25, 0))]:
        row(name, *beam_to_clinical(steve_to_beam(rz, rx), theta))
    row("chosen, all 5 targets", *chosen["all"])
    for t in TARGETS:
        row(f"chosen for {t} alone", *chosen[t], own=t)
    if args.write_table is not None:
        out = table_path(which) if args.write_table is True else args.write_table
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(make_table(which, branches, theta, chosen, **route_kw), indent=2) + "\n")
        print(f"table: {out}")
    if args.figure is not None:
        draw(branches, theta, lao, cran, res[..., 1].max(-1), chosen["all"], args.figure)
        print(f"figure: {args.figure}")


def draw(branches, theta, lao, cran, worst_covered, chosen, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle

    colour = {"aorta": "#9a9893", "bct": "#2a78d6", "rcca": "#eb6834", "rsa": "#1baf7a",
              "lcca": "#eda100", "lsa": "#e87ba4"}
    seq = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    h = STEP_DEG / 2
    im = ax[0].imshow(worst_covered * 100, origin="lower", cmap=seq, vmin=0, vmax=100, aspect="auto",
                      extent=[lao[0] - h, lao[-1] + h, cran[0] - h, cran[-1] + h])
    fig.colorbar(im, ax=ax[0]).set_label("worst target: % of route drawn over another vessel")
    cur = beam_to_clinical(steve_to_beam(20, 5), theta)
    lab = beam_to_clinical(steve_to_beam(25, 0), theta)
    for (a, b), text, mk in [(cur, "current [20, 5]", "o"), (lab, "lab [25, 0]", "s"), (chosen, "chosen", "*")]:
        ax[0].plot(a, b, mk, ms=13 if mk == "*" else 8, mfc="white", mec="#0b0b0b", mew=1.5)
        ax[0].annotate(text, (a, b), xytext=(7, 6), textcoords="offset points", fontsize=9)
    ax[0].set_xlabel("RAO  <-  0  ->  LAO  (deg)")
    ax[0].set_ylabel("caudal  <-  0  ->  cranial  (deg)")
    ax[0].set_title("Phantom: how much the worst of 5 routes is covered")

    def image(a_, rz, rx, title):
        for n, (c, r) in branches.items():                  # every other vessel, as a contrast shot shows it
            if n in colour:
                continue
            p = steve_image(c, rz, rx)
            a_.add_collection(PatchCollection([Circle(x, rr) for x, rr in zip(p, r)], facecolor="#b9b7b1",
                                              edgecolor="none", alpha=0.35))
        for n in ["aorta", "bct", "rcca", "rsa", "lcca", "lsa"]:
            c, r = branches[n]
            p = steve_image(c, rz, rx)
            a_.add_collection(PatchCollection([Circle(x, rr) for x, rr in zip(p, r)], facecolor=colour[n],
                                              edgecolor="none", alpha=0.28 if n == "aorta" else 0.45))
            a_.plot(p[:, 0], p[:, 1], color=colour[n], lw=1.2)
            if n != "aorta":
                a_.annotate(n, p[-1], xytext=(3, 3), textcoords="offset points", fontsize=9, color="#0b0b0b")
        a_.set_aspect("equal")
        box = np.concatenate([steve_image(branches[n][0], rz, rx) for n in ["bct", "rcca", "rsa", "lcca", "lsa"]])
        lo, hi = box.min(0) - 25, box.max(0) + 25
        a_.set_xlim(lo[0], hi[0])
        a_.set_ylim(lo[1] - 30, hi[1])
        a_.set_xticks([])
        a_.set_yticks([])
        a_.set_title(title, fontsize=10)

    image(ax[1], 20, 5, f"Now: stEVE [20, 5] = RAO {-cur[0]:.0f}, caudal {-cur[1]:.0f}")
    rz, rx = beam_to_steve(clinical_to_beam(*chosen, theta))
    image(ax[2], rz, rx, f"Chosen: LAO {chosen[0]:.0f}, cranial {chosen[1]:.1f} = stEVE [{rz:.0f}, {rx:.0f}]")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
