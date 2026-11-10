"""The lab's BIRO phantom as a stEVE vessel tree.

Built exactly as the lab's ``47-steve-agent-phantom/evaluate.py`` (``make_phantom_env``)
builds it: stEVE's ``FromMesh`` on ``phantom_arch.obj`` with the six centerlines
(coordinates and radii) and the insertion point from ``phantom_arch.json``, no
rotation. Both files are vendored unchanged in ``biro_phantom/`` (provenance and
sha256 in its README; ``tests/test_biro_phantom.py`` checks them).

The frame is the lab's "paper test arch" frame: z towards the head, start of the
aorta at the origin, mm. The collision mesh is the whole phantom (head and arms
included); only the six branches aorta, bct, rcca, rsa, lcca, lsa have centerlines.

``centerline="root_extended"`` reads ``phantom_arch_root.json`` instead: the same tree with
the aorta's centerline extended to the bottom of the aortic root (made from the two files
above by ``biro_phantom_root.py``; the mesh is unchanged).

``centerline="all_vessels"`` reads ``phantom_arch_all.json``: the root-extended six plus a
centerline for every other vessel in the mesh (``biro_phantom_all.py``), so the branch tubes
fill the collision mesh. The six stay the task branches (``task_branch_names``): the fixed
``CenterlineRandom`` and ``NormalizeTracking2DEpisode`` (``simulator_modifications``) take the
targets and the 2D coordinate range from them only, so both are as with ``root_extended``.
Everything else that reads ``branches`` sees all of them: the vessel geometry observation,
the path length in the reward, and where stEVE stops the push (open vessel ends).

``version=`` picks a whole phantom version (wall and centerlines together, ``VERSIONS``;
what each is: ``biro_phantom/README.md``):

  - ``"lab"``: the lab's algorithm, whole model: the wall of balls + every vessel's centerline
    (the same as ``centerline="all_vessels"``);
  - ``"lab_cut"``: the same without the right arm (``biro_cut.py``);
  - ``"rebuilt"``: everything from the phantom model with our code (``biro_rebuild.py``): the
    model's own inner wall and centerlines through it, whole model;
  - ``"rebuilt_cut"``: the same without the right arm;
  - ``"spheres"``: the inscribed-sphere wall built from the model alone (``biro_spheres.py``):
    the union of the inscribed spheres along the rebuilt centerlines, as the lab built its own,
    with those centerlines (radii at least 1 mm); whole model;
  - ``"spheres_cut"``: the same without the right arm.

The rebuilt and spheres versions list their ``junctions`` (where pieces meet); the vessel tree is joined
there instead of where stEVE guesses from the radii (``calc_branching_with_radii`` joins two
branches whenever a point of one lies within the other's radius, which merges neighbouring
ostia once the radii are measured to the real wall).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from eve.intervention.vesseltree import FromMesh
from eve.intervention.vesseltree.util.branch import BranchingPoint, BranchWithRadii

from ... import WORKSPACE_ROOT
from ..registry import register

DATA_DIR = Path(__file__).with_name("biro_phantom")
SPEC_PATH = DATA_DIR / "phantom_arch.json"
MESH_PATH = DATA_DIR / "phantom_arch.obj"
CENTERLINES = {"original": SPEC_PATH, "root_extended": DATA_DIR / "phantom_arch_root.json",
               "all_vessels": DATA_DIR / "phantom_arch_all.json"}
VERSIONS = {"lab": (MESH_PATH, CENTERLINES["all_vessels"]),
            "lab_cut": (DATA_DIR / "lab_cut.obj", DATA_DIR / "lab_cut.json"),
            "rebuilt": (DATA_DIR / "rebuilt_full.obj", DATA_DIR / "rebuilt_full.json"),
            "rebuilt_cut": (DATA_DIR / "rebuilt_cut.obj", DATA_DIR / "rebuilt_cut.json"),
            "spheres": (DATA_DIR / "spheres_full.obj", DATA_DIR / "spheres_full.json"),
            "spheres_cut": (DATA_DIR / "spheres_cut.obj", DATA_DIR / "spheres_cut.json")}
JUNCTION_RADIUS_MM = 1.0     # a branch end this close to a listed junction is not a vessel end


def load_branches(spec: dict) -> list[BranchWithRadii]:
    return [BranchWithRadii(name=name, coordinates=np.asarray(branch["coordinates"], np.float32),
                            radii=np.asarray(branch["radii"], np.float32))
            for name, branch in spec["branches"].items()]


@register("vessel_tree")
class BiroPhantomArch(FromMesh):
    """The six-branch aortic arch of the lab phantom (lab directory 47)."""

    def __init__(self, centerline: str = "original", version: str | None = None,
                 mesh: str | None = None) -> None:
        # mesh: another wall file for the same centerlines, insertion and junctions (an experiment's
        # decimated copy of the version's wall; docs/simulation_speed.md); None = the version's own
        wall_override = mesh
        if version is not None:
            if version not in VERSIONS:
                raise ValueError(f"version must be one of {list(VERSIONS)}, got {version!r}")
            mesh, spec_path = VERSIONS[version]
        elif centerline in CENTERLINES:
            mesh, spec_path = MESH_PATH, CENTERLINES[centerline]
        else:
            raise ValueError(f"centerline must be one of {list(CENTERLINES)}, got {centerline!r}")
        self.mesh_override = None
        if wall_override is not None:
            mesh = Path(wall_override)
            if not mesh.is_absolute():
                mesh = WORKSPACE_ROOT / mesh
            if not mesh.exists():
                raise FileNotFoundError(f"BiroPhantomArch mesh override not found: {mesh}")
            self.mesh_override = str(mesh)
        spec = json.loads(spec_path.read_text())
        super().__init__(str(mesh),
                         insertion_position=spec["insertion"]["position"],
                         insertion_direction=spec["insertion"]["direction"],
                         branch_list=load_branches(spec),
                         # FromMesh indexes this argument itself when rotating the
                         # insertion point, so None would crash (the lab passes zeros too).
                         rotation_yzx_deg=(0.0, 0.0, 0.0))
        # the branches the task is defined on (targets, 2D coordinate range); the rest only
        # describe the lumen (simulator_modifications reads this)
        self.task_branch_names = tuple(spec.get("task_branches", spec["branches"]))
        # the centerline file, so the per-target C-arm finds its angle table (c_arm/per_target.py)
        self.spec_path = spec_path
        if "junctions" in spec:
            self.branching_points = listed_junctions(self.branches, spec["junctions"])


def listed_junctions(branches, junctions) -> list[BranchingPoint]:
    """stEVE branching points where the spec says pieces meet."""
    by_name = {b.name: b for b in branches}
    return [BranchingPoint(np.asarray(j["point"], dtype=np.float64), JUNCTION_RADIUS_MM,
                           tuple(by_name[n] for n in j["branches"])) for j in junctions]
