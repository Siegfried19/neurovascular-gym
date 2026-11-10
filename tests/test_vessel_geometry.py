import numpy as np
import pytest

from guidewire_rl.env.observations.vessel_geometry import (FEATURE_GROUPS, G13_FEATURES,
                                                           VesselGeometry, assemble,
                                                           check_features)


def fake_groups():
    return {name: np.arange(len(dims), dtype=np.float32) + 10 * i
            for i, (name, dims) in enumerate(FEATURE_GROUPS.items())}


def test_g13_feature_order_and_size():
    term = VesselGeometry(intervention=None)
    assert term.size == 8
    assert term.feature_names == ["d0_normalized", "d1_normalized", "d2_normalized",
                                  "sin_delta_theta", "cos_delta_theta",
                                  "kappa_distal_normalized", "r_local_normalized",
                                  "direction_valid"]
    assert list(G13_FEATURES) == ["clearance", "heading", "bend", "radius", "valid"]


def test_subset_follows_listed_order():
    term = VesselGeometry(intervention=None, features=["radius", "clearance"])
    assert term.size == 4 and term.space.shape == (4,)
    groups = fake_groups()
    np.testing.assert_array_equal(assemble(groups, term.features), [30, 0, 1, 2])


@pytest.mark.parametrize("bad", [["clearence"], [], ["bend", "bend"]])
def test_bad_feature_lists_fail_loudly(bad):
    with pytest.raises(ValueError):
        check_features(bad)


def test_unknown_frame_rejected():
    with pytest.raises(ValueError):
        VesselGeometry(intervention=None, frame="tracking")


def test_sampling_spreads_points_along_a_long_segment():
    """G13/stEVE put every 1 mm point of one long segment at the first one."""
    from types import SimpleNamespace
    from guidewire_rl.env.observations.vessel_geometry import GeometryCore
    wire2 = np.asarray([[0, 0], [2.5, 0], [2.5, 3]], dtype=np.float32)
    wire3 = np.asarray([[0, 7, 0], [2.5, 7, 0], [2.5, 7, 3]], dtype=np.float32)
    fluoroscopy = SimpleNamespace(tracking2d=wire2, tracking3d=wire3)
    sampled2, sampled3 = GeometryCore.sample_tracking_pair(SimpleNamespace(fluoroscopy=fluoroscopy))
    np.testing.assert_array_equal(sampled2, [[0, 0], [1, 0], [2, 0]])
    np.testing.assert_array_equal(sampled3, [[0, 7, 0], [1, 7, 0], [2, 7, 0]])
