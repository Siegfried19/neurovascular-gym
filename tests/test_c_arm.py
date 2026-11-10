"""C-arm angle selection: conventions, routes (the whole way from the aorta, and the definition before
2026-11-03) and scores, per route and per piece, and the vendored angle tables training reads (the full
angle grid takes minutes: the fresh-run check is marked slow, run it with --run-slow; the documented
result is in guidewire_rl/env/c_arm/README.md)."""
import numpy as np
import pytest

from guidewire_rl.env.c_arm import choose_angle as ca


@pytest.fixture(scope="module")
def phantom():
    return ca.load_phantom(ca.DATA["all_vessels"])


def test_data_is_the_vendored_copy():
    assert ca.DATA["all_vessels"].exists() and ca.DATA["original"].exists()
    assert ca.DATA["all_vessels"].parent.name == "biro_phantom"


def test_steve_convention_matches_eve():
    pytest.importorskip("eve")
    ca._check_conventions()


@pytest.mark.parametrize("lao,cran", [(0, 0), (30, 10), (-20, -15), (57.5, 40)])
def test_clinical_round_trip(lao, cran):
    theta = 0.6
    d = ca.clinical_to_beam(lao, cran, theta)
    assert np.isclose(np.linalg.norm(d), 1)
    assert np.allclose(ca.beam_to_clinical(d, theta), (lao, cran))
    rz, rx = ca.beam_to_steve(d)
    back = ca.steve_to_beam(rz, rx)           # same line; the sign does not matter for a parallel projection
    assert np.allclose(back, d) or np.allclose(back, -d)


def test_routes_run_along_the_training_branches(phantom):
    branches, _ = phantom
    assert len(branches) == 75
    assert ca.parent_of(branches, "bct")[0] == "aorta"
    assert ca.parent_of(branches, "rcca")[0] == "bct"
    assert ca.parent_of(branches, "lsa")[0] == "aorta"
    assert ca.chain_to(branches, "rcca") == ["aorta", "bct", "rcca"]
    assert ca.chain_to(branches, "lcca") == ["aorta", "lcca"]


def _piece(line, name):
    p, r, pieces = line
    return p[pieces == name]


def test_route_is_the_whole_way_from_the_aorta(phantom):
    """The route (since 2026-11-03): ROUTE_BEFORE_MM of aorta before the first ostium on the way, every
    vessel on the way up to where the next one leaves it, the whole target branch."""
    branches, _ = phantom
    (line,) = ca.route(branches, "rcca")
    p, r, pieces = line
    assert len(p) == len(r) == len(pieces)
    assert list(dict.fromkeys(pieces)) == ["aorta", "bct", "rcca"]
    np.testing.assert_allclose(p[-len(branches["rcca"][0]):], branches["rcca"][0])   # the whole target branch
    _, i_rcca = ca.parent_of(branches, "rcca")
    np.testing.assert_allclose(_piece(line, "bct"), branches["bct"][0][: i_rcca + 1])  # the whole trunk
    aorta = _piece(line, "aorta")
    _, i_bct = ca.parent_of(branches, "bct")
    np.testing.assert_allclose(aorta[-1], branches["aorta"][0][i_bct])                # up to the trunk's ostium
    assert ca.ROUTE_BEFORE_MM - 3 < ca.arclen(aorta)[-1] <= ca.ROUTE_BEFORE_MM        # 60 mm of it
    # bct, lcca, lsa: the aorta's 60 mm + the branch, the same as the definition before
    for t in ("bct", "lcca", "lsa"):
        (whole,), (own,) = ca.route(branches, t), ca.route(branches, t, kind="own_ostium")
        for a, b in zip(whole, own):
            np.testing.assert_array_equal(a, b)


def test_route_before_2026_11_03_is_still_reachable(phantom):
    """``own_ostium``: the last ROUTE_BEFORE_MM before the target's own ostium (for rcca all on the trunk,
    rounded to a centerline point) + the target branch."""
    branches, _ = phantom
    (line,) = ca.route(branches, "rcca", kind="own_ostium")
    p, _, pieces = line
    assert list(dict.fromkeys(pieces)) == ["bct", "rcca"]
    before = ca.arclen(p)[len(p) - len(branches["rcca"][0])]
    assert ca.ROUTE_BEFORE_MM <= before < ca.ROUTE_BEFORE_MM + 3
    with pytest.raises(ValueError, match="route kind"):
        ca.route(branches, "rcca", kind="from_trunk")
    with pytest.raises(ValueError, match="whole route only"):
        ca.route(branches, "rcca", kind="own_ostium", past_mm=30)


def test_overshoot_puts_the_aorta_past_the_first_ostium_on(phantom):
    """``past_mm`` (checked with 30: a hooked catheter's overshoot into the ascending aorta): a second
    polyline on the aorta from the trunk's ostium on, away from the inlet."""
    branches, _ = phantom
    main, past = ca.route(branches, "rsa", past_mm=30)
    np.testing.assert_array_equal(main[0], ca.route(branches, "rsa")[0][0])
    assert set(past[2]) == {"aorta_past"}
    s = ca.arclen(branches["aorta"][0])
    _, i_bct = ca.parent_of(branches, "bct")
    np.testing.assert_allclose(past[0][0], branches["aorta"][0][i_bct])
    assert 27 < ca.arclen(past[0])[-1] <= 30
    assert s[i_bct] > s[ca.parent_of(branches, "lcca")[1]]   # the inlet side is where the other ostia are


def test_piece_scores_make_up_the_route_score(phantom):
    branches, theta = phantom
    d = ca.clinical_to_beam(15, 7.5, theta)
    for t, past in (("rcca", 0.0), ("lsa", 0.0), ("rcca", 30.0)):
        kept, covered = ca.score_route(branches, t, d, past_mm=past)
        pieces = ca.score_pieces(branches, t, d, past_mm=past)
        assert list(pieces) == ca.chain_to(branches, t) + (["aorta_past"] if past else [])
        length = sum(v["length_mm"] for v in pieces.values())
        assert np.isclose(sum(v["length_kept"] * v["length_mm"] for v in pieces.values()) / length, kept)
        shares = [v["share_covered"] for v in pieces.values()]
        assert min(shares) - 1e-12 <= covered <= max(shares) + 1e-12


def test_scores_are_shares(phantom):
    branches, theta = phantom
    for t in ca.TARGETS:
        kept, covered = ca.score_route(branches, t, ca.clinical_to_beam(20, 5, theta))
        assert 0 < kept <= 1 and 0 <= covered <= 1


def test_select_prefers_uncovered_then_long_then_straight():
    lao = np.arange(-10, 10.1, ca.STEP_DEG)
    cran = np.arange(-5, 5.1, ca.STEP_DEG)
    kept = np.ones((len(cran), len(lao)))
    covered = np.zeros_like(kept)
    covered[:, lao <= 0] = 0.5            # the whole RAO half covered
    i, j = ca.select(kept, covered, lao, cran)
    assert covered[i, j] == 0
    assert lao[j] > 0 and cran[i] == 0    # closest to straight-on among the clear ones


# ------------------------------------------------------------------- tables ----
# The tables training reads for image_rot_zx: per_target (README.md, "In training").
VENDORED = {"rebuilt_cut": "rebuilt_cut.json", "all_vessels": "phantom_arch_all.json"}


def _table(which):
    import json
    return json.loads(ca.table_path(which).read_text())


def _same(a, b, path="table"):
    """Equal, floats to 1e-9 (relative)."""
    import math
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for key in a:
            _same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: {a} != {b}"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_tables_are_vendored_for_the_phantom_versions_that_train():
    assert sorted(p.name for p in ca.TABLE_DIR.glob("angles_*.json")) == [
        "angles_all_vessels.json", "angles_rebuilt_cut.json"]
    for which, file in VENDORED.items():
        table = _table(which)
        assert table["centerline"]["input"] == which and table["centerline"]["file"] == file
        assert table["centerline"]["sha256"] == ca.sha256(ca.DATA[which])     # made from today's file
        assert table["command"] == ca.command(which)
        assert list(table["angles"]) == ["all", *ca.TARGETS]
        assert (table["method"]["route"], table["method"]["route_past_mm"]) == ("whole", 0.0)
        theta = np.radians(table["centerline"]["rotation_z_deg"])
        for row in table["angles"].values():
            # stEVE's angle and the clinical one are the same beam
            d = ca.clinical_to_beam(row["lao_deg"], row["cranial_deg"], theta)
            np.testing.assert_allclose(ca.beam_to_steve(d), row["image_rot_zx"], atol=1e-9)
            assert row["clinical"] == ", ".join(ca.clinical_words(row["lao_deg"], row["cranial_deg"]))
            assert list(row["scores"]) == ca.TARGETS
            assert list(row["scores"]["rcca"]["pieces"]) == ["aorta", "bct", "rcca"]       # trunk vs RCCA
            assert list(row["scores"]["lsa"]["pieces"]) == ["aorta", "lsa"]


def test_rebuilt_cut_table_holds_the_documented_angles():
    """README.md, "Result, model-wall version, whole routes" (2026-11-03). bct, lcca and lsa (aorta +
    branch, routes unchanged) keep the angles of the 2026-10-30 log; the all-5 angle is no longer the
    [61.0, -21.0] configs/two_devices_round1.yaml fixes (that was the all-5 angle with the routes before)."""
    from guidewire_rl import WORKSPACE_ROOT
    from guidewire_rl.config import load_config
    angles = _table("rebuilt_cut")["angles"]
    expected = {"all": (20.0, 0.0, [57.7, 0.0]), "bct": (20.0, 0.0, [57.7, 0.0]),
                "rcca": (15.0, 7.5, [52.0, -12.2]), "rsa": (27.5, 10.0, [63.4, -22.8]),
                "lcca": (20.0, -10.0, [56.3, 18.2]), "lsa": (32.5, -25.0, [58.5, 54.0])}
    for key, (lao, cran, angle) in expected.items():
        assert (angles[key]["lao_deg"], angles[key]["cranial_deg"]) == (lao, cran)
        assert (np.round(angles[key]["image_rot_zx"], 1) + 0.0).tolist() == angle
    round1 = load_config(WORKSPACE_ROOT / "configs" / "two_devices_round1.yaml")
    assert round1["env"]["scene"]["fluoroscopy"]["image_rot_zx"] == [61.0, -21.0]


def test_all_vessels_table_holds_the_readme_angles():
    """README.md, "Result, all vessels, whole routes" (2026-11-03)."""
    angles = _table("all_vessels")["angles"]
    expected = {"all": [57.7, 0.0], "bct": [61.2, -32.5], "rcca": [50.2, -23.8], "rsa": [56.3, -18.2],
                "lcca": [55.2, 42.1], "lsa": [57.0, 44.6]}
    for key, angle in expected.items():
        assert (np.round(angles[key]["image_rot_zx"], 1) + 0.0).tolist() == angle


def test_continuations_carry_their_target_on():
    """The pieces past the end of a target branch get its angle: in the rebuilt phantom ``<target>_rest``,
    in the lab's the ``v<n>_rest`` its README names "past the end of <target>"."""
    import json
    rebuilt = _table("rebuilt_cut")["continuations"]
    assert rebuilt == {"rcca_rest": "rcca", "rsa_rest": "rsa", "lcca_rest": "lcca", "lsa_rest": "lsa"}
    lab = _table("all_vessels")["continuations"]
    assert lab == {"v10_rest": "rcca", "v8_rest": "rsa", "v7_rest": "lcca", "v6_rest": "lsa"}
    added = json.loads(ca.DATA["all_vessels"].read_text())["all_vessels"]["added"]
    for name, target in lab.items():
        assert f"past the end of {target})" in added[name]
    for which in VENDORED:                     # what --write-table would write today
        assert ca.continuations(ca.load_phantom(ca.DATA[which])[0]) == _table(which)["continuations"]


@pytest.mark.slow
def test_rebuilt_cut_table_matches_a_fresh_run():
    """The whole search again (about 2 min): the vendored table is what the method gives on today's
    rebuilt_cut.json. Run with --run-slow."""
    branches, theta = ca.load_phantom(ca.DATA["rebuilt_cut"])
    chosen, _ = ca.choose(branches, theta)
    import json
    fresh = json.loads(json.dumps(ca.make_table("rebuilt_cut", branches, theta, chosen)))
    vendored = _table("rebuilt_cut")
    fresh.pop("date"), vendored.pop("date")
    _same(fresh, vendored)
