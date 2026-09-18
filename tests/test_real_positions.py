import numpy as np
import pytest

from flybrain import config
from flybrain.io import load_neurons
from flybrain.shells import fetch_shell, load_mesh

pytestmark = pytest.mark.slow
CACHE = config.CACHE_DIR / "centroids.npz"


@pytest.fixture(scope="module")
def cent():
    if not CACHE.exists():
        pytest.skip("run `uv run python -m flybrain.positions` first")
    return dict(np.load(CACHE))


@pytest.fixture(scope="module")
def neurons():
    return load_neurons()


def _post_x(neurons, cent, cell_type):
    at = dict(zip(cent["body"].tolist(), range(len(cent["body"]))))
    sel = neurons[neurons["type"] == cell_type]
    x = np.array([cent["post_mean"][at[b], 0] for b in sel["body"]])
    return x, sel["side"].to_numpy()


@pytest.mark.parametrize("cell_type", ["LPLC2", "LC4"])
def test_instance_side_matches_centroid_x(neurons, cent, cell_type):
    x, side = _post_x(neurons, cent, cell_type)
    # midpoint of the per-side medians (a pooled median is biased when L/R counts differ, e.g. LC4 71/55)
    mid = (np.nanmedian(x[side == 1]) + np.nanmedian(x[side == 2])) / 2
    left_ok = np.mean((x[side == 1] > mid) == config.LEFT_IS_POSITIVE_X)
    right_ok = np.mean((x[side == 2] < mid) == config.LEFT_IS_POSITIVE_X)
    assert left_ok >= 0.95 and right_ok >= 0.95, (left_ok, right_ok)
    xl, xr = x[side == 1], x[side == 2]
    gap = xl.min() - xr.max() if config.LEFT_IS_POSITIVE_X else xr.min() - xl.max()
    assert gap > 0, gap  # real data: the two sides do not overlap at all


def test_most_neurons_have_positions(cent):
    assert np.isfinite(cent["all_mean"][:, 0]).mean() > 0.98


def _inside_bbox(xyz_nm, verts):
    return np.all((xyz_nm >= verts.min(0)) & (xyz_nm <= verts.max(0)), axis=1).mean()


def test_shells_are_meshes_in_the_same_space_and_axis_order(neurons, cent):
    """Brain neurons sit in the brain-shell bbox and VNC neurons in the VNC one; any axis swap breaks this."""
    brain_v, brain_f = load_mesh(fetch_shell("brain"))
    vnc_v, _ = load_mesh(fetch_shell("vnc"))
    assert brain_v.shape == (32724, 3) and brain_f.shape == (66360, 3)
    assert np.array_equal(cent["body"], neurons["body"].to_numpy())
    xyz = cent["all_mean"] * config.VOXEL_NM  # voxels → nm
    ok = np.isfinite(xyz[:, 0])
    sc = neurons["superclass"].to_numpy()
    brain_cells = xyz[ok & np.isin(sc, ["ol_intrinsic", "cb_intrinsic"])]
    vnc_cells = xyz[ok & (sc == "vnc_intrinsic")]
    assert len(brain_cells) > 10_000 and len(vnc_cells) > 1_000
    in_brain = _inside_bbox(brain_cells, brain_v)
    vnc_in_vnc, vnc_in_brain = _inside_bbox(vnc_cells, vnc_v), _inside_bbox(vnc_cells, brain_v)
    assert in_brain >= 0.99 and vnc_in_vnc >= 0.99 and vnc_in_brain <= 0.01, (in_brain, vnc_in_vnc, vnc_in_brain)
