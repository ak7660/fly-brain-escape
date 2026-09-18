import numpy as np
import pytest

from flybrain import config
from flybrain.stimulus import (Threat, frame_labels, input_drive, loom_angle_deg, loom_rate_dps, rf_gain,
                               sample_episodes, threat_class)


def test_loom_angle_is_90_degrees_at_one_lv_before_collision_and_increasing():
    lv, tc = 0.04, 1.0
    assert loom_angle_deg(tc - lv, lv, tc) == pytest.approx(90.0)
    t = np.linspace(0, tc - 1e-3, 200)
    assert np.all(np.diff(loom_angle_deg(t, lv, tc)) > 0)


def test_loom_rate_matches_numerical_derivative():
    lv, tc, t, h = 0.04, 1.0, 0.8, 1e-6
    numeric = (loom_angle_deg(t + h, lv, tc) - loom_angle_deg(t - h, lv, tc)) / (2 * h)
    assert loom_rate_dps(t, lv, tc) == pytest.approx(numeric, rel=1e-5)


def test_receptive_fields_are_side_tuned():
    assert rf_gain(-90, 1) == pytest.approx(np.exp(-0.5 * (45 / 50) ** 2))
    assert rf_gain(-90, 2) == pytest.approx(np.exp(-0.5 * (135 / 50) ** 2))
    assert rf_gain(0, 1) == pytest.approx(rf_gain(0, 2))
    assert rf_gain(180, 1) == pytest.approx(rf_gain(-180, 1))  # azimuth wraps


def test_input_drive_kinds_sides_and_activity_window():
    kind = np.array([0, 0, 1])   # LPLC2 L, LPLC2 R, LC4 L
    side = np.array([1, 2, 1])
    th = [Threat(azimuth_deg=-90, lv=0.04, t_spawn=0.1, t_collision=0.7)]
    assert np.all(input_drive(0.05, th, kind, side) == 0)          # before spawn
    assert np.all(input_drive(0.70, th, kind, side) == 0)          # at collision: gone
    d = input_drive(0.65, th, kind, side)
    theta = loom_angle_deg(0.65, 0.04, 0.7)
    assert d[0] == pytest.approx(min(theta / 90, 1) * rf_gain(-90, 1))
    assert d[1] < 0.05 * d[0]
    assert d[2] == pytest.approx(min(loom_rate_dps(0.65, 0.04, 0.7) / 1000, 1) * rf_gain(-90, 1))


@pytest.mark.parametrize("az,cls", [(-90, 1), (-60, 1), (90, 2), (60, 2), (0, 3), (30, 3), (-44, 3)])
def test_threat_class(az, cls):
    assert threat_class(az) == cls


def test_frame_labels_switch_on_at_label_onset_angle():
    lv, spawn, tc = 0.04, 0.1, 0.7
    labels = frame_labels([Threat(90, lv, spawn, tc)], 60)
    t = np.arange(60) / config.FPS
    theta = np.where((t >= spawn) & (t < tc), loom_angle_deg(t, lv, tc), 0)
    expect = np.where(theta >= config.LABEL_ONSET_DEG, 2, 0)
    assert labels.tolist() == expect.tolist()
    assert frame_labels([], 10).tolist() == [0] * 10


def test_sample_episodes_shapes_masks_and_determinism():
    kind = np.array([0, 0, 1, 1]); side = np.array([1, 2, 1, 2])
    a = sample_episodes(200, kind, side, np.random.default_rng(0))
    b = sample_episodes(200, kind, side, np.random.default_rng(0))
    assert a["drive"].shape == (200, 72, 4) and a["drive"].dtype == np.float32
    assert a["labels"].shape == (200, 72) and a["mask"].dtype == bool
    assert np.array_equal(a["drive"], b["drive"])
    empty = a["episode_class"] == 0
    assert 0.1 < empty.mean() < 0.3
    assert np.all(a["labels"][empty] == 0) and np.all(a["mask"][empty])
    for i in np.flatnonzero(~empty):
        th = a["threats"][i][0]
        f_hit = int(round(th.t_collision * config.FPS))
        assert not a["mask"][i, f_hit:].any() and a["mask"][i, :f_hit].all()
        assert set(np.unique(a["labels"][i])) <= {0, a["episode_class"][i]}
