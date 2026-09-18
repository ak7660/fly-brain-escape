import json

import numpy as np
import pytest

from flybrain import config
from flybrain.bundle import read_bundle
from flybrain.export import DEFAULT_OUT, FIXTURE, arrays_from_bundle
from flybrain.sim_numpy import NumpySim

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bundle():
    if not (DEFAULT_OUT / "manifest.json").exists():
        pytest.skip("run `uv run python -m flybrain.export` first")
    return read_bundle(DEFAULT_OUT)


def test_manifest_counts_and_size(bundle):
    man, data = bundle
    assert man["counts"]["neurons"] == len(data["neurons.bin"]["bias"])
    assert data["csr.bin"]["row_ptr"][-1] == man["counts"]["edges"] == len(data["csr.bin"]["col"])
    assert sum(f["bytes"] for f in man["files"].values()) < 8_000_000
    for f in data.values():
        for arr in f.values():
            assert np.all(np.isfinite(arr))


def test_positions_sit_inside_the_shells(bundle):
    _, data = bundle
    pos = data["neurons.bin"]["pos"].reshape(-1, 3)
    verts = np.concatenate([data[f"shell_{n}.bin"]["verts"].reshape(-1, 3) for n in ("brain", "vnc")])
    lo, hi = verts.min(0) - 5, verts.max(0) + 5
    assert np.mean(np.all((pos >= lo) & (pos <= hi), axis=1)) > 0.99


def test_fixture_reproduces_from_bins(bundle):
    man, data = bundle
    fx = json.loads(FIXTURE.read_text())
    sim = NumpySim(arrays_from_bundle(data))
    r = data["neurons.bin"]["r0"].astype(np.float64)
    from flybrain.stimulus import Threat, input_drive
    threats = [Threat(**t) for t in fx["threats"]]
    a = arrays_from_bundle(data)
    for f in range(fx["frames"]):
        r, logits = sim.frame(r, input_drive(f / config.FPS, threats, a["input_kind"].astype(int), a["input_side"].astype(int)))
        assert np.allclose(logits, fx["logits"][f], rtol=1e-5, atol=1e-6)
