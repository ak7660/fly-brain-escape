import numpy as np

import flybrain.calibrate as cal
from flybrain.calibrate import evaluate_setting, run_sweep
from flybrain.circuit import build_circuit
from flybrain.io import load_edges, load_neurons


def _circuit(d):
    neurons = load_neurons(d)
    return build_circuit(neurons, load_edges(neurons["body"].to_numpy(), d))


def test_evaluate_setting_reports_every_gate_field(synthetic_data_dir):
    res = evaluate_setting(_circuit(synthetic_data_dir), gain=8.0, bias=0.02, rng=np.random.default_rng(0))
    assert set(res) >= {"stable", "frac_saturated", "max_drift", "noise_sd", "response", "responsive", "gf_peak",
                        "gf_front_preferred", "recruited", "tonic_active", "passed_v1", "passed"}
    assert set(res["response"]) == set(res["recruited"]) == {"L", "F", "R"}
    assert all(0.0 <= v <= 1.0 for v in res["recruited"].values())
    assert res["passed_v1"] == (res["stable"] and all(res["responsive"].values()) and res["gf_front_preferred"])
    # gate v2: v1 plus a non-silent baseline and substantial DN recruitment in every direction
    assert res["passed"] == (res["passed_v1"] and res["bias"] > 0 and
                             min(res["recruited"].values()) >= cal.MIN_DN_RECRUITMENT)


def test_silent_network_fails_gate_v2(synthetic_data_dir):
    res = evaluate_setting(_circuit(synthetic_data_dir), gain=8.0, bias=0.0, rng=np.random.default_rng(0))
    assert res["passed"] is False  # bias 0 → no tonic activity, rejected regardless of recruitment


def test_sweep_chooses_smallest_passing_gain_then_bias(synthetic_data_dir, monkeypatch):
    import flybrain.calibrate as cal
    passing = {(4.0, 0.05), (4.0, 0.02), (8.0, 0.0)}
    monkeypatch.setattr(cal, "evaluate_setting", lambda c, gain, bias, rng: {"passed": (gain, bias) in passing})
    out = run_sweep(None, gains=(2.0, 4.0, 8.0), biases=(0.0, 0.02, 0.05))
    assert out["chosen"] == {"gain": 4.0, "bias": 0.02} and len(out["settings"]) == 9
