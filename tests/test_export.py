import json

import numpy as np
import torch

from flybrain.bundle import read_bundle
from flybrain.circuit import build_circuit
from flybrain.export import arrays_from_bundle, build_bundle, make_parity_fixture
from flybrain.io import load_edges, load_neurons
from flybrain.model import RateModel
from flybrain.sim_numpy import NumpySim


def test_bundle_roundtrip_simulates_identically_and_fixture_is_reproducible(tmp_path, synthetic_data_dir):
    neurons = load_neurons(synthetic_data_dir)
    c = build_circuit(neurons, load_edges(neurons["body"].to_numpy(), synthetic_data_dir))
    m = RateModel(c, gain=4.0, bias=0.05)
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 1e5, (len(c.body), 3))
    dust = rng.uniform(0, 1e5, (50, 3))
    tet = (np.array([[0, 0, 0], [1e4, 0, 0], [0, 1e4, 0], [0, 0, 1e4]], float), np.array([[0, 1, 2], [0, 1, 3]]))
    manifest = build_bundle(c, m, pos, dust, {"brain": tet, "vnc": tet}, tmp_path / "b", variant="test")

    man, data = read_bundle(tmp_path / "b")
    a = arrays_from_bundle(data)
    assert man["counts"]["neurons"] == len(c.body) and man["counts"]["edges"] == len(c.pre)
    ref = man["viz"]["role_ref"]
    assert len(ref) == 4 and all(v >= 1e-3 for v in ref)  # per-role display scale for change-from-rest
    assert a["row_ptr"][-1] == len(c.pre)
    ref = m.export_arrays()
    for k in ("row_ptr", "col", "val", "bias", "alpha", "input_idx", "dn_idx", "W_out", "b_out"):
        assert np.array_equal(a[k].reshape(ref[k].shape), ref[k]), k
    sim = NumpySim(a)
    assert np.allclose(data["neurons.bin"]["r0"], sim.steady_state(), atol=1e-6)
    assert json.loads((tmp_path / "b" / "neuron_info.json").read_text())["body"][0] == str(c.body[0])

    fx1 = make_parity_fixture(tmp_path / "b", tmp_path / "p1.json", frames=40)
    fx2 = make_parity_fixture(tmp_path / "b", tmp_path / "p2.json", frames=40)
    assert fx1 == fx2 and len(fx1["logits"]) == 40 and len(fx1["rates_full"]["0"]) == len(c.body)
