"""Tests for the workbench's parts.

    conda run --no-capture-output -n guidewire_rl python -m pytest -q visual/workbench/tests
    ... -m "not sofa"      # skip the one test that builds a SOFA scene
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from parts.frame import (EpisodeLog, Frame, SceneInfo, arc_length, fluoro_rotation, inserted_wire,  # noqa: E402
                         instrument_polylines, polyline_head, project_2d, resample_polyline,
                         tracked_points_3d)
from parts.instrument import (InstrumentInfo, SectionInfo, instrument_svg, instrument_table_html,  # noqa: E402
                              instruments_html)
from parts.i18n import POLICY_TEXT, TEXT, has_cjk, other, pick, policy_text, tr  # noqa: E402
from parts.policies import (ConstantPolicy, Filtered, PassThroughFilter, RandomPolicy,  # noqa: E402
                            checkpoint_label, policy_menu)
from parts.plots import PLOTS, plot_data  # noqa: E402
from parts.text import action_html, force_html, status_html  # noqa: E402

NAMES = ([f"position[{i}]" for i in range(10)] + ["target[0]", "target[1]", "rotation[0]", "rotation[1]",
         "d0_normalized", "d1_normalized", "d2_normalized", "sin_delta_theta", "cos_delta_theta",
         "kappa_distal_normalized", "r_local_normalized", "direction_valid"])
LAYOUT = [("position", 10), ("target", 2), ("rotation", 2), ("geometry", 8)]


def make_instrument() -> InstrumentInfo:
    """A 450 mm wire: 420 mm shaft, then 15 mm straight and a 72 deg arc of radius 12 mm."""
    arc = np.radians(np.linspace(0, 72, 30))
    tip = np.vstack([[[0.0, 0, 0]], np.stack([15 + 12 * np.sin(arc), 12 * (1 - np.cos(arc)), 0 * arc], axis=1)])
    shape = np.vstack([[[-420.0, 0, 0]], tip])
    return InstrumentInfo(
        name="guidewire", kind="Angled", shape_params={"tip_radius": 12.0, "tip_angle_deg": 72.0},
        sections=[SectionInfo("shaft", 420.0, 0.89, 0.0, 80e3, 6.45e-9),
                  SectionInfo("tip", 30.0, 0.7, 0.0, 17e3, 6.45e-9)],
        rest_shape_mm=shape)


def make_info() -> SceneInfo:
    s = np.linspace(0, 1, 30)[:, None]
    aorta = np.hstack([0 * s, 0 * s, 200 * s])
    branch = aorta[-1] + np.hstack([40 * s, 0 * s, 30 * s])
    head = branch[-1] + np.hstack([0 * s, 10 * s, 60 * s])
    return SceneInfo(
        vessel_vertices=np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32),
        vessel_faces=np.array([[0, 1, 2]]), centerlines={"aorta": aorta, "lcca": branch, "v7": head},
        insertion_position=np.zeros(3), insertion_direction=np.array([0, 0, 1.0]),
        target_threshold_mm=5.0, image_rot_zx_deg=(20.0, 5.0), observation_names=NAMES,
        observation_layout=LAYOUT, action_low=np.array([-50.0, -3.14]),
        action_high=np.array([50.0, 3.14]), max_steps=200, step_seconds=1 / 7.5,
        vessel_key="k", description="test", task_branches=("aorta", "lcca"),
        centerline_radii={"aorta": np.full(30, 12.0), "lcca": np.full(30, 4.0), "v7": np.full(30, 2.0)},
        instruments=[make_instrument()])


def make_frame(step: int, action=None) -> Frame:
    wire = np.stack([np.zeros(20), np.zeros(20), np.linspace(10 + step, 0, 20)], axis=1)
    return Frame(episode=0, step=step, obs=np.linspace(-1, 1, 22).astype(np.float32), wire_mm=wire,
                 target_mm=np.array([40.0, 0, 230]), action=action,
                 raw_action=None if action is None else np.asarray(action) * [50, 3.14],
                 reward=0.01 * step, episode_return=0.01 * step * (step + 1) / 2,
                 labels={"distal_force": 0.1 * step} if step else {},
                 predictions={"distal_force": 0.09 * step} if step else {},
                 inserted_mm=np.array([10.0 + step]), rotation_rad=np.array([0.5]), seed=7,
                 extras={"nominal_action": np.array([1.0, 0.0]), "filter": {"changed": True}})


# ------------------------------------------------------------------ frame --
def test_inserted_wire_drops_parked_tail():
    tip_part = np.stack([np.zeros(5), np.zeros(5), np.linspace(20, 4, 5)], axis=1)
    parked = np.tile([[0.0, 0.0, 2.0]], (7, 1))
    out = inserted_wire(np.vstack([tip_part, parked]))
    assert len(out) == 6 and np.allclose(out[-1], [0, 0, 2]) and np.allclose(out[0], [0, 0, 20])
    assert len(inserted_wire(parked)) == 1


def test_projection_matches_stEVE():
    coordtransform = pytest.importorskip("eve.util.coordtransform")
    points = np.random.default_rng(0).normal(size=(12, 3)) * 50
    expected = coordtransform.tracking3d_to_2d(
        coordtransform.vessel_cs_to_tracking3d(points, (20, 5), (0, 0, 0), None))
    assert np.allclose(fluoro_rotation((20, 5)), coordtransform._get_rot_matrix((20, 5)))
    assert np.allclose(project_2d(points, (20, 5)), expected, atol=1e-4)


def test_resample_polyline_spacing():
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    out = resample_polyline(line, 1.0, 5)
    assert np.allclose(out[:, 0], [0, 1, 2, 3, 4])
    assert np.allclose(resample_polyline(line, 4.0, 5)[-1], [10, 0])  # repeats the end


def test_episode_log_roundtrip(tmp_path):
    log = EpisodeLog(make_info())
    log.append(make_frame(0))
    for k in range(1, 4):
        log.append(make_frame(k, action=np.array([0.5, -0.2])))
    log.frames[-1].terminated = log.frames[-1].success = True
    path = log.save(tmp_path / "ep.npz")
    back = EpisodeLog.load(path)
    assert len(back) == 4 and back.info.observation_layout == LAYOUT
    assert back.frames[0].action is None and np.allclose(back.frames[2].action, [0.5, -0.2])
    assert all(np.allclose(a.wire_mm, b.wire_mm) for a, b in zip(log.frames, back.frames))
    assert back.frames[3].success and back.frames[3].labels == {"distal_force": pytest.approx(0.3)}
    assert back.frames[1].extras["nominal_action"] == [1.0, 0.0]
    assert np.allclose(back.info.centerlines["lcca"], log.info.centerlines["lcca"])
    assert back.info.task_branches == ("aorta", "lcca") and np.allclose(back.info.centerline_radii["v7"], 2.0)
    inst = back.info.instruments[0]
    assert inst.kind == "Angled" and inst.length_mm == pytest.approx(450.0)
    assert np.allclose(inst.rest_shape_mm, make_instrument().rest_shape_mm, atol=1e-4)


def test_tracked_points_3d_sit_on_the_wire_and_match_the_image_spacing():
    wire = np.stack([np.zeros(40), np.linspace(0, 20, 40), np.linspace(20, 0, 40)], axis=1)   # tip first
    points = tracked_points_3d(wire, (20.0, 5.0), spacing_mm=1.0, count=5)
    assert np.allclose(points[0], wire[0])
    image = project_2d(points, (20.0, 5.0))
    assert np.allclose(np.linalg.norm(np.diff(image, axis=0), axis=1), 1.0, atol=1e-6)   # 1 mm apart in the image
    assert np.allclose(np.cross(points - wire[0], wire[-1] - wire[0]), 0, atol=1e-9)    # on the (straight) wire


def make_catheter() -> InstrumentInfo:
    return InstrumentInfo(name="catheter", kind="HookCatheter", shape_params={},
                          sections=[SectionInfo("shaft", 400.0, 2.0, 1.0, 4e3, 5e-6),
                                    SectionInfo("tip", 50.0, 2.0, 1.0, 4e3, 5e-6)],
                          rest_shape_mm=np.zeros((2, 3)))


def bent_beam() -> np.ndarray:
    """120 mm tip-first beam: 60 mm straight, then a quarter circle of radius 38 mm."""
    arc = np.linspace(0, np.pi / 2, 40)
    bend = np.stack([38 * (1 - np.cos(arc)), 0 * arc, 60 + 38 * np.sin(arc)], axis=1)
    straight = np.stack([np.zeros(30), np.zeros(30), np.linspace(0, 60, 30, endpoint=False)], axis=1)
    return np.vstack([straight, bend])[::-1]


def test_instrument_polylines_count_the_shorter_one_back_from_the_leading_tip():
    beam = bent_beam()
    wire, cath = instrument_polylines(beam, [100.0, 70.0])     # wire leads by 30 mm
    assert np.allclose(wire, beam)
    assert arc_length(beam)[-1] - arc_length(cath)[-1] == pytest.approx(30.0, abs=1e-6)
    assert np.allclose(cath[-1], beam[-1])
    lead_cath, inner_wire = instrument_polylines(beam, [70.0, 100.0])[::-1]    # catheter leads
    assert np.allclose(lead_cath, beam) and arc_length(inner_wire)[-1] == pytest.approx(arc_length(beam)[-1] - 30)
    assert len(instrument_polylines(beam, [100.0])) == 1 and len(instrument_polylines(beam, None)) == 1


def test_instrument_polylines_match_the_two_device_env():
    pytest.importorskip("eve")
    from guidewire_rl.env.two_device.phases import instrument_polylines as env_split
    beam = bent_beam()
    for inserted in ([100.0, 70.0], [70.0, 100.0], [100.0, 100.0], [300.0, 10.0]):
        for ours, theirs in zip(instrument_polylines(beam, inserted), env_split(beam, inserted)):
            assert np.allclose(ours, theirs)


def test_polyline_head_cuts_at_the_length():
    beam = bent_beam()
    head = polyline_head(beam, 25.0)
    assert np.allclose(head[0], beam[0]) and arc_length(head)[-1] == pytest.approx(25.0)
    assert len(polyline_head(beam, 0.0)) == 0 and len(polyline_head(beam, -5.0)) == 0
    assert np.allclose(polyline_head(beam, 1e3), beam)


class FakeHandle:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.removed = False

    def remove(self):
        self.removed = True


class FakeScene:
    def __init__(self):
        self.splines, self.spheres, self.meshes = [], [], []

    def add_spline_catmull_rom(self, name, points, thickness, color, segments, visible):
        h = FakeHandle(name=name, points=np.asarray(points), thickness=thickness, color=color, visible=visible)
        self.splines.append(h)
        return h

    def add_icosphere(self, name, radius, color, cast_shadow):
        h = FakeHandle(name=name, radius=radius, color=color, visible=True, position=None)
        self.spheres.append(h)
        return h

    def add_mesh_simple(self, name, vertices, faces, color, opacity, side, cast_shadow, receive_shadow, visible):
        h = FakeHandle(name=name, vertices=np.asarray(vertices), faces=np.asarray(faces), color=color,
                       opacity=opacity, side=side, visible=visible)
        self.meshes.append(h)
        return h

    def live(self, kind):
        return [h for h in getattr(self, kind) if not h.removed]


def test_wire_layer_one_instrument_is_drawn_as_before():
    from parts.scene import MM, WIRE_COLOR, WireLayer
    server = type("S", (), {"scene": FakeScene()})()
    layer = WireLayer(server)
    frame = make_frame(3)
    layer.update(frame, None, make_info().instruments)
    (spline,), (tip,) = server.scene.live("splines"), server.scene.live("spheres")
    assert spline.name == "/wire/body" and spline.thickness == pytest.approx(1.8 * MM)
    assert np.allclose(spline.points, frame.wire_mm * MM) and spline.color == WIRE_COLOR
    assert tip.radius == pytest.approx(1.1 * 1.8 * MM) and np.allclose(tip.position, frame.wire_mm[0] * MM)


def test_wire_layer_draws_the_catheter_and_only_the_wire_that_sticks_out():
    from parts.scene import CATHETER_COLOR, MM, WireLayer
    server = type("S", (), {"scene": FakeScene()})()
    layer = WireLayer(server)
    instruments = [make_instrument(), make_catheter()]
    beam = bent_beam()
    frame = Frame(episode=0, step=1, obs=np.zeros(3, np.float32), wire_mm=beam, inserted_mm=np.array([100.0, 70.0]))
    layer.update(frame, None, instruments)
    wire, cath = server.scene.live("splines")
    assert wire.thickness == pytest.approx(1.0 * MM) and cath.thickness == pytest.approx(2.0 * MM)
    assert cath.color == CATHETER_COLOR and arc_length(wire.points / MM)[-1] == pytest.approx(30.0)
    assert np.allclose(wire.points[-1] / MM, cath.points[0] / MM)          # the wire ends where the catheter starts
    assert all(t.visible for t in server.scene.live("spheres"))
    # catheter ahead: the wire is inside it and not drawn, its tip hidden
    frame.inserted_mm = np.array([70.0, 100.0])
    layer.update(frame, None, instruments)
    (cath,) = [h for h in server.scene.live("splines") if h.visible]
    assert np.allclose(cath.points / MM, beam) and cath.thickness == pytest.approx(2.0 * MM)
    wire_tip, cath_tip = server.scene.live("spheres")
    assert not wire_tip.visible and cath_tip.visible
    layer.set_visible(False)
    layer.set_visible(True)
    assert not wire_tip.visible and cath_tip.visible
    assert [h for h in server.scene.live("splines") if h.visible] == [cath]  # the hidden wire stays hidden
    assert len(server.scene.splines) == 2               # tubes changed in place, not removed and re-added


def test_status_lists_each_instrument():
    info = make_info()
    info.instruments = [make_instrument(), make_catheter()]
    frame = make_frame(2)
    frame.inserted_mm, frame.rotation_rad = np.array([100.0, 70.0]), np.array([0.0, np.pi / 2])
    zh, en = status_html(info, frame, "p", "s"), status_html(info, frame, "p", "s", "en")
    assert "导丝 100.0 mm · 0°" in zh and "导管 70.0 mm · 90°" in zh
    assert "catheter 70.0 mm" in en and not has_cjk(en)
    assert "12.0 mm · 29°" in status_html(make_info(), make_frame(2), "p", "s")    # one instrument: as before


# ------------------------------------------------------------------- body --
def inside_mesh(points, vertices, faces, chunk=32):
    """Generalised winding number > 1/2: the points a closed triangle mesh encloses."""
    tri = np.asarray(vertices, np.float64)[faces]
    wn = np.empty(len(points))
    for k in range(0, len(points), chunk):
        a, b, c = (tri[None, :, i] - points[k:k + chunk, None] for i in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        dot = lambda x, y: np.einsum("pti,pti->pt", x, y)  # noqa: E731
        wn[k:k + chunk] = np.arctan2(dot(a, np.cross(b, c)),
                                     la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb).sum(1) / (2 * np.pi)
    return np.abs(wn) > 0.5


def test_body_skin_is_the_closed_makehuman_body_only():
    from parts.body import load_skin
    skin = load_skin()
    assert len(skin.vertices) == 13380 and len(skin.faces) == 2 * 13378      # no eyes, teeth, helpers
    edges = np.sort(np.concatenate([skin.faces[:, [0, 1]], skin.faces[:, [1, 2]], skin.faces[:, [2, 0]]]), axis=1)
    assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()      # closed surface
    assert np.ptp(skin.vertices[:, 1]) == pytest.approx(1666, abs=2)        # a 1.67 m adult, in mm
    assert skin.joints["l-clavicle"][0] > 0 and skin.joints["neck"][1] < skin.joints["head"][1]


def test_trunk_is_closed_and_has_no_arms_or_legs():
    from parts.body import HIP_CUT_ABOVE_CROTCH_MM, load_skin, trunk
    skin = load_skin()
    v, f = trunk(skin)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()      # cuts closed
    crotch = skin.vertices[(np.abs(skin.vertices[:, 0]) < 8) & (skin.vertices[:, 1] > -300), 1].min()
    assert v[:, 1].min() == pytest.approx(crotch + HIP_CUT_ABOVE_CROTCH_MM, abs=0.5)
    assert np.abs(v[:, 0]).max() < 245 < np.abs(skin.vertices[:, 0]).max()  # arms cut at the shoulders
    assert v[:, 1].max() == pytest.approx(skin.vertices[:, 1].max())        # the head is all there


def test_body_keeps_its_shape():
    """Placed = turned and moved only: distances between skin points stay what MakeHuman has."""
    from parts.body import REFERENCE, _branches, load_skin, place
    skin = load_skin()
    placed = place(skin, _branches(REFERENCE))
    pick = np.random.default_rng(0).choice(len(placed), 300, replace=False)
    d = lambda v: np.linalg.norm(v[pick, None] - v[None, pick], axis=-1)  # noqa: E731
    assert np.allclose(d(placed), d(skin.vertices), atol=1e-6)
    assert np.linalg.det(np.linalg.lstsq(np.c_[skin.vertices, np.ones(len(placed))], placed, rcond=None)[0][:3]) \
        == pytest.approx(1.0)                                                # not mirrored


def test_body_axes_follow_the_anatomy():
    from parts.body import REFERENCE, _branches, body_axes
    branches = _branches(REFERENCE)
    left, up, front = body_axes(branches)
    assert np.linalg.det(np.c_[left, up, front]) == pytest.approx(1.0)
    aorta = branches["aorta"]
    assert (aorta[-1] - aorta[0]) @ left < 0                                 # ascending aorta right of the descending
    assert branches["lsa"][-1] @ left > branches["rsa"][-1] @ left


def test_body_holds_the_phantom_vessels():
    """The numbers in parts/body.py: every vessel point inside, most well under the skin."""
    from parts.body import REFERENCE, _branches, body_axes, body_mesh
    vertices, faces = body_mesh()
    branches = _branches(REFERENCE)
    points = np.concatenate(list(branches.values()))[::8]
    inside = inside_mesh(points, vertices, faces)
    up = points @ body_axes(branches)[1]
    assert inside.all() and up.max() > 280                                   # head vessels included


def test_body_only_around_the_biro_phantom():
    import dataclasses
    from parts.body import VESSELS, _branches, fits
    for name in ("rebuilt_cut", "spheres_cut", "spheres_full", "lab_cut", "phantom_arch"):
        assert fits(dataclasses.replace(make_info(), centerlines=_branches(VESSELS / f"{name}.json"))), name
    moved = {k: v + [0, 0, 40.0] for k, v in _branches(VESSELS / "rebuilt_cut.json").items()}
    assert not fits(dataclasses.replace(make_info(), centerlines=moved))
    assert not fits(make_info())


def test_body_layer_draws_its_inner_side_and_follows_the_vessel():
    import dataclasses
    from parts.body import BODY_COLOR, VESSELS, BodyLayer, _branches
    from parts.xray import visible_in_xray
    server = type("S", (), {"scene": FakeScene()})()
    layer = BodyLayer(server)
    biro = dataclasses.replace(make_info(), centerlines=_branches(VESSELS / "rebuilt_cut.json"))
    layer.set_scene(biro)
    (mesh,) = server.scene.live("meshes")
    assert mesh.side == "back" and mesh.color == BODY_COLOR and mesh.opacity == pytest.approx(0.15)
    layer.set_opacity(0.4)
    layer.set_visible(False)
    assert mesh.opacity == pytest.approx(0.4) and not mesh.visible
    layer.set_scene(make_info())                                             # another vessel: no body
    assert server.scene.live("meshes") == [] and not layer.shown
    layer.set_scene(biro)
    (mesh,) = server.scene.live("meshes")
    assert not mesh.visible and mesh.opacity == pytest.approx(0.4)
    assert visible_in_xray("body", contrast=True, policy_marks=True) is False


# ------------------------------------------------------------------- text --
def test_panels_render_with_and_without_data():
    info = make_info()
    first, later = make_frame(0), make_frame(3, action=np.array([0.5, -0.2]))
    for frame in (first, later):
        for out in (action_html(info, frame), force_html(frame), status_html(info, frame, "随机动作", "暂停")):
            assert isinstance(out, str) and out
    assert "+25.0 mm/s" in action_html(info, later)               # 0.5 of ±50 mm/s
    assert "原本" in action_html(info, later, np.array([1.0, -0.2]))  # filtered action shows the nominal one
    assert "导丝尖端接触力" in force_html(later) and "0.300 N" in force_html(later) and "0.270 N" in force_html(later)


def test_two_instruments_show_their_own_output_and_force():
    info = make_info()
    info.instruments = [make_instrument(), make_catheter()]
    info.action_low, info.action_high = np.array([-40.0, -3.14, -25.0, -1.57]), np.array([40.0, 3.14, 25.0, 1.57])
    frame = make_frame(3)
    frame.action = np.array([0.5, 0.0, -0.4, 1.0])
    frame.labels = {"distal_force": 0.2, "catheter_tip_force": 1.5, "wire_tip_clearance": 2.0}
    frame.predictions = {}
    out = action_html(info, frame)
    assert "导丝 · 推送" in out and "导管 · 推送" in out and "导管 · 旋转" in out
    assert "+20.0 mm/s" in out and "-10.0 mm/s" in out and "+1.6 rad/s" in out   # each with its own limits
    forces = force_html(frame)
    assert "导丝尖端接触力" in forces and "导管尖端接触力" in forces and "1.500 N" in forces
    assert "clearance" not in forces                                         # not a force: not shown
    assert "catheter · push" in action_html(info, frame, lang="en")


def test_plot_series_line_up_with_frames():
    log = EpisodeLog(make_info())
    log.append(make_frame(0))
    log.append(make_frame(1, action=np.array([1.0, 0.0])))
    for key, (_, series) in PLOTS.items():
        data = plot_data(log, key)
        assert len(data) == 1 + len(series) and len({len(d) for d in data}) == 1
    x, measured, predicted = plot_data(log, "force_guidewire")
    assert list(x) == [1] and measured[0] == pytest.approx(0.1)   # step 0 has no force: left out


def test_plots_send_no_nan_and_their_own_y_range():
    """No model predicts the force in a replay: that series is left out, the rest has no NaN."""
    import contextlib
    from parts.plots import PlotPanel, drawable

    class Plots:
        def __init__(self):
            self.handles = []

        def add_uplot(self, **kw):
            self.handles.append(FakeHandle(visible=True, **kw))
            return self.handles[-1]

    server = type("S", (), {"gui": Plots(), "atomic": staticmethod(contextlib.nullcontext)})()
    panel = PlotPanel(server)
    log = EpisodeLog(make_info())
    log.append(make_frame(0))
    for k in (1, 2, 3):
        frame = make_frame(k, action=np.array([0.5, -0.2]))
        frame.labels, frame.predictions = {"distal_force": [0.0, 0.0, 0.004][k - 1]}, {}
        log.append(frame)
    panel.update(log)
    force = panel._handles["force_guidewire"]
    assert [series["label"] for series in force.series] == ["步", "实测"]     # no predicted line
    assert len(force.data) == 2 and all(np.isfinite(d).all() for d in force.data)
    assert force.scales["y"]["auto"] and "range" not in force.scales["y"]
    scales, series = force.scales, force.series
    panel.update(log)
    assert force.scales is scales and force.series is series                # only the data changes (no flashing)
    assert drawable((np.array([1.0, 2.0]), np.array([np.nan, 3.0])))[1][1].tolist() == [3.0, 3.0]


def test_catheter_plots_only_with_a_catheter():
    from parts.plots import SHOWN_IF
    info = make_info()
    assert not SHOWN_IF["force_catheter"](info) and not SHOWN_IF["action_catheter"](info)
    info.instruments = [make_instrument(), make_catheter()]
    assert SHOWN_IF["force_catheter"](info) and SHOWN_IF["action_catheter"](info)
    log = EpisodeLog(info)
    frame = make_frame(1)
    frame.action = np.array([0.1, 0.2, 0.3, 0.4])
    frame.labels = {"distal_force": 0.2, "catheter_tip_force": 1.5}
    log.append(frame)
    _, push, rotate = plot_data(log, "action_catheter")
    assert push[0] == pytest.approx(0.3) and rotate[0] == pytest.approx(0.4)
    assert plot_data(log, "force_catheter")[1][0] == pytest.approx(1.5)


# --------------------------------------------------------------- language --
def test_every_text_has_both_languages():
    from parts import environments, text
    # the toggle button is the one place that shows the other language on purpose
    pairs = [p for k, p in TEXT.items() if k != "switch_to"] + list(POLICY_TEXT.values())
    pairs += [name for name, _ in text.ACTION_TEXT] + [name for name, _ in text.FORCE_TEXT.values()]
    pairs += [title for title, _ in PLOTS.values()] + [name for _, series in PLOTS.values() for name, _, _ in series]
    pairs += [spec[k] for spec in environments.PHANTOMS.values() for k in ("label", "note")]
    pairs += list(environments.SETTING_TEXT.values())
    from parts import parameters
    pairs += [p.label for p in parameters.PARAMS]
    for zh, en in pairs:
        assert zh and en and not has_cjk(en), (zh, en)
    assert other("zh") == "en" and other("en") == "zh"


def test_english_panels_have_no_chinese():
    info = make_info()
    later = make_frame(3, action=np.array([0.5, -0.2]))
    done = make_frame(4, action=np.array([0.5, -0.2]))
    done.success = True
    failed = make_frame(5, action=np.array([0.5, -0.2]))
    failed.extras["error"] = "solver diverged"
    bare = make_frame(1, action=np.array([0.1, 0.1]))
    bare.labels, bare.predictions = {}, {}
    for frame in (make_frame(0), later, done, failed, bare):
        policy = policy_text("随机动作 + 不过滤", "en")
        outputs = [action_html(info, frame, np.array([1.0, -0.2]), "en"), force_html(frame, "en"),
                   status_html(info, frame, policy, tr("state_paused", "en"), "en")]
        for out in outputs:
            assert out and not has_cjk(out), out
    assert "guidewire tip contact force" in force_html(later, "en")
    assert "+25.0 mm/s" in action_html(info, later, lang="en")
    assert "导丝尖端接触力" in force_html(later)                       # Chinese stays the default


def test_policy_names_translate_but_checkpoints_keep_theirs():
    assert policy_text("只往前推", "en") == "push forward only"
    assert policy_text("只往前推", "zh") == "只往前推"
    assert policy_text("run / step_20000", "en") == "run / step_20000"
    assert policy_text("随机动作 + 不过滤", "en") == "random actions + no filter"


def _panel_text(server) -> list:
    """Every label, option, hint and text block in the viewer's side panel."""
    import dataclasses
    found = []
    gui = server.gui
    handles = list(gui._container_handle_from_uuid.values())
    handles += [child for c in handles for child in getattr(c, "_children", {}).values()]
    for handle in handles:
        props = getattr(getattr(handle, "_impl", None), "props", None)
        if props is None:
            continue
        for field in dataclasses.fields(props):
            value = getattr(props, field.name)
            items = value if isinstance(value, tuple) else (value,)
            for item in items:
                if isinstance(item, str) and not item.startswith("<svg"):
                    found.append(item)
                elif isinstance(getattr(item, "label", None), str):   # uPlot series
                    found.append(item.label)
    return found


def _free_port() -> int:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_workbench_panel_switches_language_and_keeps_settings(tmp_path):
    from workbench import Workbench

    menu = policy_menu(2, [], lambda p: None)
    bench = Workbench(None, menu=menu, initial_policy="只往前推", port=_free_port(), record_dir=tmp_path,
                      seed=7, live=True, lang="en")
    try:
        english = _panel_text(bench.server)
        assert "Play" in english and "push forward only" in english and "Guidewire tip contact force (N)" in english
        assert "X-ray mode" in english and "Instrument" in english
        assert [t for t in english if has_cjk(t)] == [tr("switch_to", "en")]
        bench.speed.value, bench.layer_boxes["trail"].value, bench.contrast_box.value = 12.0, False, True
        bench.lang = other(bench.lang)
        bench._rebuild_gui()
        chinese = _panel_text(bench.server)
        assert "播放" in chinese and "只往前推" in chinese and "English" in chinese and "X 光模式" in chinese
        assert bench.speed.value == 12.0 and bench.layer_boxes["trail"].value is False
        assert bench.contrast_box.value is True
        assert bench._policy_keys[bench.policy_dd.value] == "只往前推"
        bench.lang = other(bench.lang)
        bench._rebuild_gui()
        assert [t for t in _panel_text(bench.server) if has_cjk(t)] == ["中文"]
    finally:
        bench.server.stop()


def test_workbench_environment_folder_and_parameter_edits(tmp_path):
    pytest.importorskip("guidewire_rl")
    from guidewire_rl.config import load_config
    from parts import parameters as P
    from parts.environments import phantoms, settings
    from workbench import Workbench

    bench = Workbench(None, menu=policy_menu(2, [], lambda p: None), initial_policy="不动", port=_free_port(),
                      record_dir=tmp_path, seed=1, live=True, lang="en", phantom_list=phantoms(),
                      setting_list=settings(), phantom_key="spheres_cut", setting_key="phantom_g13", friction=0.1)
    try:
        bench.param_base = P.read(load_config(ROOT / "configs" / "phantom_friction.yaml"))
        bench.param_values = dict(bench.param_base, friction=0.05)
        bench._rebuild_gui()
        english = _panel_text(bench.server)
        assert "Phantom version" in english and "Instrument setting" in english and "Friction (instruments on the wall)" in english
        assert "guidewire only: default mass, 6 ms step (G13's)" in english and "inscribed spheres (chest and head)" in english
        assert "Friction (wire on wall) (edited)" in english and "Physics step (ms)" in english
        assert [t for t in english if has_cjk(t)] == [tr("switch_to", "en")]
        bench._ask_env(phantom="rebuilt_cut")                       # a new phantom keeps the edits
        assert bench.pending_env == ("rebuilt_cut", "phantom_g13", bench.param_values)
        bench._ask_env(setting="phantom_real_mass")                 # a new setting starts from its own
        assert bench.pending_env == ("spheres_cut", "phantom_real_mass", None)
        bench.param_inputs["dt"].value = 2.0
        bench._ask_params(apply=True)
        phantom, setting, values = bench.pending_env
        assert values["dt"] == pytest.approx(0.002) and values["friction"] == pytest.approx(0.05)
        assert bench.friction == pytest.approx(0.05)                 # the friction menu follows the parameter
        bench._ask_friction(0.001)                                  # another friction: same environment, edits kept
        assert bench.pending_env[:2] == ("spheres_cut", "phantom_g13")
        assert bench.pending_env[2]["friction"] == 0.001 and bench.pending_env[2]["dt"] == bench.param_values["dt"]
    finally:
        bench.server.stop()


def test_a_failed_switch_does_not_lock_the_environment_menus(tmp_path):
    """Wilson's arch with the two-device setting cannot open; picking another phantom after that
    must open it with the setting that was asked for (it used to fail on a missing setting)."""
    from parts.environments import phantoms, settings
    from workbench import Workbench

    def configure(phantom, setting):
        if phantom.key == "wilson" and setting.key == "two_devices_round1":
            raise ValueError("Route needs a vessel tree with branches and junctions")
        return {}

    bench = Workbench(None, menu=policy_menu(2, [], lambda p: None), initial_policy="不动", port=_free_port(),
                      record_dir=tmp_path, seed=1, live=True, lang="en", phantom_list=phantoms(),
                      setting_list=settings(), phantom_key="rebuilt_cut", setting_key="two_devices_round1",
                      configure=configure, opener=lambda p, s, c: (_ for _ in ()).throw(RuntimeError("no simulator")))
    try:
        bench._open_env("wilson", "two_devices_round1", None)
        assert bench.state == "switch_failed" and bench.env_failed
        assert (bench.phantom_key, bench.setting_key) == ("wilson", "two_devices_round1")
        bench._ask_env(phantom="spheres_cut")
        assert bench.pending_env[:2] == ("spheres_cut", "two_devices_round1")
        bench._ask_env(phantom="wilson")                            # the same pick tries again
        assert bench.pending_env[:2] == ("wilson", "two_devices_round1")
        bench._ask_env(setting="phantom_g13")
        assert bench.pending_env == ("wilson", "phantom_g13", None)
    finally:
        bench.server.stop()


# --------------------------------------------------------------- policies --
def test_keyboard_keys_drive_each_instrument():
    from parts.keyboard import KeyboardPolicy, KeyState, keyboard_action
    two = keyboard_action(frozenset({"ArrowUp", "ArrowLeft", "KeyS"}), 4, 0.5)
    assert two.tolist() == [0.5, -0.5, -0.5, 0.0]               # wire push, wire rotate back, catheter pull
    assert keyboard_action(frozenset({"ArrowUp", "ArrowDown"}), 4, 1.0).tolist() == [0, 0, 0, 0]
    assert keyboard_action(frozenset({"KeyW", "ArrowRight"}), 2, 1.0).tolist() == [0.0, 1.0]  # one instrument
    keys = KeyState(timeout_s=0.05)
    keys.set(["ArrowUp", ""])
    assert keys.held() == {"ArrowUp"}
    policy = KeyboardPolicy(4, keys)
    keys.speed = 0.25
    assert policy.act(np.zeros(3), first_step=True).tolist() == [0.25, 0, 0, 0]
    import time
    time.sleep(0.08)
    assert keys.held() == frozenset()                           # the browser went quiet: released


def test_scripted_policy_wraps_a_controller():
    from parts.scripted import ScriptedPolicy, staged_factory

    class Controller:
        stage, substage, tries, last_extra = "wire", "probe", 2, {"d": 1.5, "skip": [1]}

        def reset(self, env):
            self.reset_with = env

        def act(self, env):
            return [0.5, 0, 0.1, 0]

    ctl = Controller()
    policy = ScriptedPolicy("env", ctl)
    policy.reset()
    assert ctl.reset_with == "env" and policy.act(np.zeros(3), first_step=True).tolist() == pytest.approx([0.5, 0, 0.1, 0])
    assert policy.debug() == {"stage": "wire:probe", "tries": 2, "d": 1.5}
    assert staged_factory(type("Source", (), {"action_dim": 2})()) is None       # one instrument: not listed


def test_keyboard_server_and_script():
    import urllib.request
    from parts.keyboard import KeyServer, KeyState, capture_html
    keys = KeyState()
    server = KeyServer(keys, host="127.0.0.1")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/keys?s=ArrowUp,KeyD") as reply:
            assert reply.status == 204 and reply.headers["Access-Control-Allow-Origin"] == "*"
        assert keys.held() == {"ArrowUp", "KeyD"}
    finally:
        server.close()
    on = capture_html(8123, active=True, two=True)
    assert "onerror=" in on and "8123" in on and "wb-keys-held" in on and "导管 W S" in on
    assert '"' not in on.split('onerror="', 1)[1].split('">', 1)[0]      # the script is escaped into the attribute
    off = capture_html(8123, active=False, two=True)
    assert "display:none" in off and "&quot;active&quot;: false" in off and "导丝" not in off
    assert capture_html(None, active=True, two=True) == ""
    assert "catheter W S" in capture_html(8123, active=True, two=True, lang="en")


def test_policies_and_filter():
    obs = np.zeros(22, np.float32)
    rnd = RandomPolicy(2, seed=0).act(obs, first_step=True)
    assert rnd.shape == (2,) and np.all(np.abs(rnd) <= 1)
    filtered = Filtered(ConstantPolicy([1.0, 0.0], "推"), PassThroughFilter())
    assert np.allclose(filtered.act(obs, first_step=True), [1, 0])
    assert np.allclose(filtered.debug()["nominal_action"], [1, 0])
    menu = policy_menu(2, [Path("runs/a/checkpoints/step_10.pt")], lambda p: ConstantPolicy([0, 0], str(p)),
                       extra={"键盘手动": lambda: ConstantPolicy([0, 0], "键盘手动")})
    assert list(menu) == ["随机动作", "不动", "只往前推", "键盘手动", "a / step_10"]
    assert np.allclose(menu["只往前推"]().act(obs, first_step=False), [1, 0])
    assert checkpoint_label(Path("runs/a/checkpoints/step_10.pt")) == "a / step_10"


def test_camera_fluoro_looks_along_the_projection_axis():
    from parts.scene import camera_pose
    info = make_info()
    position, look_at, up = camera_pose("fluoro", info)
    forward = (look_at - position) / np.linalg.norm(look_at - position)
    assert np.allclose(project_2d(forward, info.image_rot_zx_deg), 0, atol=1e-9)  # depth axis
    assert np.allclose(project_2d(up, info.image_rot_zx_deg), [0, 1], atol=1e-9)  # image up


# -------------------------------------------------------------- live env --
@pytest.mark.sofa
def test_live_source_runs_the_simulator():
    pytest.importorskip("Sofa")
    from guidewire_rl.config import load_config
    from parts.sources import LiveEnvSource

    config = load_config(ROOT / "configs" / "g13_upstream.yaml")
    source = LiveEnvSource(config, ConstantPolicy([1.0, 0.0], "只往前推"))
    try:
        first = source.reset(seed=3)
        info = source.scene_info()
        assert len(info.observation_names) == source.observation_dim == len(first.obs)
        frames = [source.step() for _ in range(8)]
        lo, hi = info.vessel_vertices.min(0) - 5, info.vessel_vertices.max(0) + 5
        for f in frames:
            assert np.all((f.wire_mm >= lo) & (f.wire_mm <= hi))  # wire is in the vessel frame
        assert frames[-1].inserted_mm[0] > first.inserted_mm[0]
        assert len(frames[-1].wire_mm) > len(first.wire_mm)       # more wire inside after pushing
        assert "distal_force" in frames[-1].labels and frames[-1].step == 8
        # the target is drawn in the vessel frame: it is a point of a centerline
        lines = np.concatenate(list(info.centerlines.values()))
        assert np.linalg.norm(lines - frames[-1].target_mm, axis=1).min() < 0.01
    finally:
        source.close()


@pytest.mark.sofa
def test_live_source_reports_instruments_and_training_branches():
    pytest.importorskip("Sofa")
    from guidewire_rl.config import load_config
    from parts.sources import LiveEnvSource

    source = LiveEnvSource(load_config(ROOT / "configs" / "phantom_all_vessels.yaml"))
    try:
        source.set_policy(ConstantPolicy([0.0, 0.0], "不动"))
        source.reset(seed=1)
        info = source.scene_info()
        assert set(info.task_branches) == {"aorta", "bct", "rcca", "rsa", "lcca", "lsa"}
        assert len(info.centerlines) > len(info.task_branches) and set(info.centerline_radii) == set(info.centerlines)
        (wire,) = info.instruments
        assert wire.kind == "Angled" and wire.length_mm == pytest.approx(450.0, abs=0.5)
        assert [s.role for s in wire.sections] == ["shaft", "tip"] and wire.sections[1].length_mm == pytest.approx(30.0)
        end = wire.rest_shape_mm[-1] - wire.rest_shape_mm[-2]
        assert np.degrees(np.arctan2(end[1], end[0])) == pytest.approx(72.0, abs=1.0)   # the bend at rest
    finally:
        source.close()


@pytest.mark.sofa
def test_a_new_phantom_setting_pair_and_edited_parameters_run():
    """Wilson's vessel (G13's generated arch) with the phantom setting, friction and physics step edited."""
    pytest.importorskip("Sofa")
    from parts import parameters as P
    from parts.environments import build_config, phantoms, settings
    from parts.sources import LiveEnvSource

    phantom = next(p for p in phantoms() if p.key == "wilson")
    setting = next(s for s in settings() if s.key == "phantom_g13")
    config = build_config(phantom, setting)
    config = P.apply(config, dict(P.read(config), friction=0.05, dt=0.004))
    source = LiveEnvSource(config, ConstantPolicy([1.0, 0.0], "只往前推"))
    try:
        source.reset(seed=2)
        sim = source.env.intervention.simulation
        assert sim.friction == 0.05 and sim.dt_simulation == 0.004
        assert type(source.env.intervention.vessel_tree).__name__ == "AorticArch"
        frames = [source.step() for _ in range(4)]
        assert frames[-1].inserted_mm[0] > 0 and "distal_force" in frames[-1].labels   # 33 substeps, no error
    finally:
        source.close()


def test_replay_source_plays_a_saved_episode(tmp_path):
    from parts.sources import ReplaySource
    log = EpisodeLog(make_info())
    log.append(make_frame(0))
    for k in range(1, 4):
        log.append(make_frame(k, action=np.array([0.5, -0.2])))
    source = ReplaySource(log.save(tmp_path / "ep.npz"))
    assert source.reset().step == 0 and source.scene_info().max_steps == 200
    steps = [source.step() for _ in range(5)]
    assert [f.step for f in steps] == [1, 2, 3, 3, 3]     # stays on the last frame
    assert steps[-1].done                                 # recording end counts as the end


# ------------------------------------------------------------- instrument --
def test_instrument_window_draws_and_counts():
    info = make_instrument()
    assert info.mass_g == pytest.approx(6.45e-9 * np.pi / 4 * (0.89 ** 2 * 420 + 0.7 ** 2 * 30) * 1e6)
    for lang in ("zh", "en"):
        svg, table = instrument_svg(info, lang), instrument_table_html(info, lang)
        assert "<svg" in svg and "polyline" in svg and "10 mm" in svg
        assert "450.0 mm" in table and "g/cm³" in table and ">6.45<" in table and "1.76 g" in table
        assert "72" in table
    assert not has_cjk(instrument_svg(info, "en") + instrument_table_html(info, "en"))
    assert not has_cjk(instruments_html([], "en")) and "没有器械" in instruments_html([], "zh")


def test_instrument_info_reads_stEVE_angled():
    angled = pytest.importorskip("eve.intervention.instrument.angled")
    from parts.instrument import instrument_info
    info = instrument_info(angled.Angled(mass_density=6.45e-9, flex_mass_density=6.45e-9))
    assert info.length_mm == pytest.approx(450.0, abs=0.5)
    assert info.sections[0].length_mm == pytest.approx(420.0) and info.sections[1].diameter_outer_mm == 0.7
    assert np.allclose(info.rest_shape_mm[0], [-420.0, 0, 0]) and info.shape_params["tip_angle_deg"] == 72.0
    straight = np.linalg.norm(info.rest_shape_mm[1:] - info.rest_shape_mm[:-1], axis=1).sum()
    assert straight == pytest.approx(450.0, abs=0.5)


# ---------------------------------------------------------- centerlines --
def test_outside_side_follows_the_normals():
    import trimesh
    from parts.scene import outside_side
    box = trimesh.creation.box()
    assert outside_side(box.vertices, box.faces) == "front"
    assert outside_side(box.vertices, box.faces[:, ::-1]) == "back"


def test_centerline_groups_and_radius_tubes():
    from parts.scene import tube_mesh
    info = make_info()
    assert [n for n in info.centerlines if info.is_task_branch(n)] == ["aorta", "lcca"]
    line = np.stack([np.zeros(10), np.zeros(10), np.linspace(0, 90, 10)], axis=1)
    vertices, faces = tube_mesh(line, np.full(10, 3.0), sides=8)
    assert vertices.shape == (80, 3) and faces.shape == (2 * 8 * 9, 3)
    assert np.allclose(np.linalg.norm(vertices[:, :2], axis=1), 3.0)          # radius kept
    assert faces.max() < len(vertices)


# ------------------------------------------------------------------ X-ray --
def test_xray_rules_and_camera():
    from parts.xray import region_of_interest, visible_in_xray, xray_camera
    assert visible_in_xray("vessel", contrast=False, policy_marks=True) is False   # no contrast: no vessels
    assert visible_in_xray("vessel", contrast=True, policy_marks=True) is None
    assert visible_in_xray("wire", contrast=False, policy_marks=False) is None
    assert visible_in_xray("centerlines", contrast=True, policy_marks=True) is False
    assert visible_in_xray("tracked", contrast=False, policy_marks=True) is True
    assert visible_in_xray("target", contrast=False, policy_marks=False) is False
    info = make_info()
    position, look_at, up, fov = xray_camera(info)
    forward = (look_at - position) / np.linalg.norm(look_at - position)
    assert np.allclose(project_2d(forward, info.image_rot_zx_deg), 0, atol=1e-9)   # along the beam
    assert np.allclose(project_2d(up, info.image_rot_zx_deg), [0, 1], atol=1e-9)
    center, half = region_of_interest(info)
    assert center[2] == pytest.approx(115.0) and half < 140                        # the task branches, not v7
    distance = np.linalg.norm(look_at - position) / 1e-3
    assert distance * np.tan(fov / 2) == pytest.approx(half)                       # the region fills the height


# ----------------------------------------------------------- environments --
def test_phantoms_and_settings_rebuild_the_configs(tmp_path):
    pytest.importorskip("guidewire_rl")
    from guidewire_rl.config import load_config
    from parts import parameters
    from parts.environments import (DEFAULT_PHANTOM, DEFAULT_SETTING, SETTING_TEXT, build_config, find_setting,
                                    friction_of, locate, phantoms, settings)
    P, S = phantoms(), settings()
    assert [p.key for p in P] == ["rebuilt_cut", "spheres_cut", "wilson", "rebuilt", "spheres"]
    assert all(p.kind == "sim" for p in P)
    by_key = {s.key: s for s in S}
    assert {"phantom_g13", "phantom_real_mass", "two_devices_round1"} <= set(by_key)
    folded = {"phantom_friction", "phantom_root", "phantom_all_vessels"}  # differ only in friction / vessel
    assert not folded & set(by_key) and folded <= set(by_key["phantom_g13"].also)
    assert find_setting("phantom_friction", S) is by_key["phantom_g13"]
    assert friction_of(ROOT / "configs" / "phantom_friction.yaml") == 0.1
    assert friction_of(ROOT / "configs" / "phantom_g13.yaml") == 0.001
    assert all(s.key in SETTING_TEXT for s in S)                # every listed setting has a readable name
    assert DEFAULT_PHANTOM in {p.key for p in P} and DEFAULT_SETTING in by_key
    path = tmp_path / "spheres_friction.yaml"                  # a config = a listed phantom x a setting
    path.write_text(f"extends: {ROOT / 'configs' / 'phantom_friction.yaml'}\n"
                    "env: {scene: {vessel_tree: {type: BiroPhantomArch, version: spheres_cut}}}\n")
    phantom, setting = locate(path, P, S)
    assert (phantom, setting) == ("spheres_cut", "phantom_g13")
    rebuilt, original = build_config(next(p for p in P if p.key == phantom), by_key[setting]), load_config(path)
    rebuilt = parameters.apply(rebuilt, {"friction": friction_of(path)})    # the friction is chosen apart
    for c in (rebuilt, original):
        c.pop("run", None)
    assert rebuilt == original
    # the lab's centerlines and wall are not in the workbench any more
    assert locate(ROOT / "configs" / "phantom_all_vessels.yaml", P, S) == (None, "phantom_g13")
    assert locate(ROOT / "configs" / "phantom_g13.yaml", P, S) == (None, "phantom_g13")
    assert locate(ROOT / "configs" / "two_devices_round1.yaml", P, S) == ("rebuilt_cut", "two_devices_round1")


def test_checkpoint_compatibility_looks_at_observation_and_action():
    from parts.environments import compatible
    a = {"env": {"observation": [{"name": "position"}], "action": "Linear"}}
    assert compatible(a, {"env": {"observation": [{"name": "position"}], "action": "Linear", "scene": 1}})
    assert not compatible(a, {"env": {"observation": [{"name": "target"}], "action": "Linear"}})


def test_shape_source_draws_a_vessel_without_a_simulator():
    from parts.sources import ShapeSource
    data = ROOT / "guidewire_rl" / "env" / "vessels" / "biro_phantom"
    source = ShapeSource(data / "rebuilt_cut.obj", data / "rebuilt_cut.json", (20.0, 5.0), description="rebuilt_cut")
    frame = source.reset()
    info = source.scene_info()
    assert not source.can_run and source.step() is frame and len(frame.wire_mm) == 1
    assert len(info.task_branches) == 6 and set(info.centerline_radii) == set(info.centerlines)
    for out in (status_html(info, frame, "—", "view only", "en"), force_html(frame, "en"),
                action_html(info, frame, lang="en")):
        assert isinstance(out, str)


# ------------------------------------------------------------- parameters --
def test_parameter_defaults_are_the_class_defaults():
    import inspect
    angled = pytest.importorskip("eve.intervention.instrument.angled")
    speedups = pytest.importorskip("simulator_modifications.speedups")
    from parts.parameters import find
    sim = inspect.signature(speedups.SofaBeamAdapter.__init__).parameters
    assert find("friction").default == sim["friction"].default
    assert find("dt").default == sim["dt_simulation"].default
    assert find("alarm").default == sim["alarm_distance"].default
    wire = angled.Angled()
    assert find("density").default == wire.mass_density and find("flex_density").default == wire.flex_mass_density


def test_parameters_read_apply_and_keep_the_force_label_in_step():
    pytest.importorskip("guidewire_rl")
    from guidewire_rl.config import load_config
    from parts import parameters as P
    config = load_config(ROOT / "configs" / "phantom_real_mass.yaml")
    values = P.read(config)
    assert values["friction"] == 0.1 and values["dt"] == 0.002 and values["alarm"] == 2.0
    assert values["density"] == 6.45e-9 and values["carm"] == [20, 5] and values["max_steps"] == 200
    assert P.shown(P.find("density"), values["density"]) == pytest.approx(6.45)
    assert P.stored(P.find("dt"), 6.0) == pytest.approx(0.006) and P.stored(P.find("max_steps"), 150.4) == 150
    assert P.changed(values, values) == []
    edited = dict(values, dt=0.006, friction=0.05)
    assert P.changed(values, edited) == ["friction", "dt"]
    out = P.apply(config, edited)
    assert out["env"]["scene"]["simulation"]["dt_simulation"] == 0.006
    assert out["env"]["labels"][0]["expected_substeps"] == 22            # 1/7.5 s / 6 ms, truncated
    assert config["env"]["scene"]["simulation"]["dt_simulation"] == 0.002   # the original is untouched
    g13 = load_config(ROOT / "configs" / "phantom_g13.yaml")
    base = P.read(g13)
    assert base["alarm"] == 0.5 and base["density"] == 2.1e-5           # not in the YAML: class defaults
    assert P.apply(g13, base)["env"]["labels"][0]["expected_substeps"] == 22
