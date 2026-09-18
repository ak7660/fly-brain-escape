import math

import numpy as np
import torch

from flybrain import config
from flybrain.circuit import Circuit, build_circuit
from flybrain.io import load_edges, load_neurons
import pytest

from flybrain.model import RateModel, _FixedSpMV


def tiny_circuit():
    """input LPLC2 (0) → hidden (1) → DN (2); w0 0→1 = +0.5, 1→2 = −0.25."""
    s = lambda *v: np.array(v)
    return Circuit(body=s(1, 2, 3), type=s("LPLC2", "H", "DNx"), instance=s("a_L", "b", "c"),
                   superclass=s("visual_projection", "cb_intrinsic", "descending_neuron"),
                   sign=s(1, -1, 1).astype(np.int8), side=s(1, 0, 0).astype(np.uint8),
                   role=s(0, 1, 2).astype(np.uint8), layer=s(0, 1, 3).astype(np.uint8),
                   d_fwd=s(0, 1, 2), in_total=s(0, 10, 8).astype(np.float32),
                   pre=s(0, 1).astype(np.int32), post=s(1, 2).astype(np.int32),
                   weight=s(5, 2).astype(np.int32), w0=s(0.5, -0.25).astype(np.float32), meta={})


def test_hand_computed_frame():
    m = RateModel(tiny_circuit(), gain=2.0, bias=0.1, tau_ms=8.0, dtype=torch.float64)
    with torch.no_grad():
        m.W_out.zero_(); m.W_out[3, 0] = 1.0; m.b_out.zero_()
    logits, r = m.simulate(torch.ones(1, 1, 1, dtype=torch.float64))
    alpha = config.DT_SUB * 1000 / 8.0
    r_exp = [0.0, 0.0, 0.0]
    for _ in range(config.SUBSTEPS):
        i0 = 0.1 + 1.0
        i1 = 2.0 * 0.5 * r_exp[0] + 0.1
        i2 = 2.0 * -0.25 * r_exp[1] + 0.1
        r_exp = [(1 - alpha) * r + alpha * min(max(i, 0.0), config.R_MAX) for r, i in zip(r_exp, (i0, i1, i2))]
    assert np.allclose(r[0].detach().numpy(), r_exp, atol=1e-12)
    assert math.isclose(float(logits[0, 0, 3].detach()), r_exp[2], abs_tol=1e-12)


def test_tau_init_is_exact():
    m = RateModel(tiny_circuit(), tau_ms=20.0, dtype=torch.float64)
    assert np.allclose(m.neuron_alpha().detach().numpy(), config.DT_SUB * 1000 / 20.0)


def test_gradients_reach_every_type_level_parameter(synthetic_data_dir):
    neurons = load_neurons(synthetic_data_dir)
    c = build_circuit(neurons, load_edges(neurons["body"].to_numpy(), synthetic_data_dir))
    m = RateModel(c, gain=2.0, bias=0.05)
    drive = torch.rand(3, 5, int((c.role == 0).sum()))
    logits, _ = m.simulate(drive)
    torch.nn.functional.cross_entropy(logits.reshape(-1, 4), torch.randint(0, 4, (15,))).backward()
    for name in ("log_gain", "pre_scale", "post_scale", "type_bias", "tau_raw", "log_input_gain", "W_out", "b_out"):
        assert getattr(m, name).grad is not None and float(getattr(m, name).grad.abs().sum()) > 0, name


def test_export_keeps_signs_after_training_steps(synthetic_data_dir):
    neurons = load_neurons(synthetic_data_dir)
    c = build_circuit(neurons, load_edges(neurons["body"].to_numpy(), synthetic_data_dir))
    m = RateModel(c, gain=2.0, bias=0.05)
    opt = torch.optim.Adam(m.parameters(), lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        logits, _ = m.simulate(torch.rand(2, 4, int((c.role == 0).sum())))
        (-logits[..., 0].mean()).backward()
        opt.step()
    a = m.export_arrays()
    rows = np.repeat(np.arange(len(c.body)), np.diff(a["row_ptr"]))
    sign_by_pair = {(int(p), int(q)): int(s) for p, q, s in zip(c.post, c.pre, np.sign(c.w0))}
    assert all(np.sign(v) == sign_by_pair[(int(r), int(k))] for r, k, v in zip(rows, a["col"], a["val"]))
    assert a["row_ptr"][-1] == len(c.pre) and a["row_ptr"].dtype == np.uint32
    assert a["W_out"].shape == (4, int((c.role == 2).sum()))


def test_fixed_spmv_gradient_values():
    g = torch.Generator().manual_seed(0)
    dense = torch.randn(12, 12, generator=g, dtype=torch.float64) * (torch.rand(12, 12, generator=g) < 0.3)
    W, Wt = dense.to_sparse_csr(), dense.T.contiguous().to_sparse_csr()
    x = torch.randn(12, 3, generator=g, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: _FixedSpMV.apply(v, W, Wt), (x,))


def test_simulate_gradient_values():
    m = RateModel(tiny_circuit(), gain=2.0, bias=0.1, tau_ms=8.0, dtype=torch.float64)
    names = [n for n, _ in m.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_(True) for p in m.parameters())
    drive = torch.tensor([[[0.7], [1.3]]], dtype=torch.float64)

    def logits(*ps):
        return torch.func.functional_call(m, dict(zip(names, ps)), (drive,))[0]

    assert torch.autograd.gradcheck(logits, params)


def test_tau_outside_open_range_is_rejected():
    for tau in (config.TAU_MIN_MS, config.TAU_MAX_MS, 1.0, 500.0):
        with pytest.raises(ValueError, match="tau_ms"):
            RateModel(tiny_circuit(), tau_ms=tau)


def test_gains_are_bounded_so_float32_cannot_overflow():
    m = RateModel(tiny_circuit(), gain=2.0)
    with torch.no_grad():
        m.pre_scale.fill_(1000.0)
        m.post_scale.fill_(1000.0)
    assert torch.allclose(m.pre_gain(), torch.tensor(math.exp(6.0)))
    assert torch.allclose(m.post_gain(), torch.tensor(math.exp(6.0 + math.log(64.0))))
    with torch.no_grad():
        m.pre_scale.fill_(-1000.0)
        m.post_scale.fill_(-1000.0)
    assert torch.allclose(m.pre_gain(), torch.tensor(math.exp(-6.0)))
    assert torch.allclose(m.post_gain(), torch.tensor(math.exp(-6.0)))
    assert all(np.isfinite(v).all() for v in m.export_arrays().values())


def test_export_rejects_non_finite_parameters():
    m = RateModel(tiny_circuit())
    with torch.no_grad():
        m.b_out[0] = float("nan")
    with pytest.raises(ValueError, match="b_out"):
        m.export_arrays()


def test_simulate_casts_inputs_and_forward_is_simulate():
    m = RateModel(tiny_circuit(), gain=2.0, bias=0.1, dtype=torch.float64)
    drive = torch.ones(2, 3, 1, dtype=torch.float32)
    r0 = torch.full((3,), 0.5, dtype=torch.float32)
    logits, r = m.simulate(drive, r0)
    assert logits.dtype == torch.float64 and r.dtype == torch.float64
    logits2, r2 = m(drive, r0)
    assert torch.equal(logits, logits2) and torch.equal(r, r2)
