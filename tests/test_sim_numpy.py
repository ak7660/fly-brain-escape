import numpy as np
import torch

from flybrain.circuit import build_circuit
from flybrain.io import load_edges, load_neurons
from flybrain.model import RateModel
from flybrain.sim_numpy import NumpySim


def _model(d, seed=0, dtype=torch.float64):
    neurons = load_neurons(d)
    c = build_circuit(neurons, load_edges(neurons["body"].to_numpy(), d))
    m = RateModel(c, gain=3.0, bias=0.05, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (m.pre_scale, m.post_scale, m.type_bias, m.tau_raw, m.log_input_gain, m.W_out, m.b_out):
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=torch.float64))
    return c, m


def test_torch_matches_numpy_reference(synthetic_data_dir):
    c, m = _model(synthetic_data_dir)
    ni = int((c.role == 0).sum())
    drive = torch.rand(2, 7, ni, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    logits_t, r_t = m.simulate(drive)
    sim = NumpySim(m.export_arrays(dtype=np.float64))
    for b in range(2):
        out = sim.run(drive[b].numpy())
        assert np.allclose(out["logits"], logits_t[b].detach().numpy(), atol=1e-10)
        assert np.allclose(out["r"], r_t[b].detach().numpy(), atol=1e-10)
    r0 = torch.rand(len(c.body), dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 2
    logits_t, r_t = m.simulate(drive, r0)
    for b in range(2):
        out = sim.run(drive[b].numpy(), r0=r0.numpy())
        assert np.allclose(out["logits"], logits_t[b].detach().numpy(), atol=1e-10)
        assert np.allclose(out["r"], r_t[b].detach().numpy(), atol=1e-10)


def test_float32_model_matches_float64_sim_on_float32_export(synthetic_data_dir):
    c, m = _model(synthetic_data_dir, dtype=torch.float32)
    ni = int((c.role == 0).sum())
    drive = torch.rand(2, 7, ni, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits_t, r_t = m.simulate(drive)
    sim = NumpySim(m.export_arrays(dtype=np.float32))
    for b in range(2):
        out = sim.run(drive[b].numpy().astype(np.float64))
        assert np.allclose(out["logits"], logits_t[b].numpy(), atol=1e-5)
        assert np.allclose(out["r"], r_t[b].numpy(), atol=1e-5)


def test_steady_state_is_a_fixed_point_and_run_records(synthetic_data_dir):
    c, m = _model(synthetic_data_dir)
    sim = NumpySim(m.export_arrays(dtype=np.float64))
    r_ss = sim.steady_state(frames=400)
    r_next, _ = sim.frame(r_ss, np.zeros(len(sim.input_idx)))
    assert np.allclose(r_next, r_ss, atol=1e-8)
    out = sim.run(np.zeros((3, len(sim.input_idx))), r0=r_ss, record=True)
    assert out["rates"].shape == (3, sim.n)
