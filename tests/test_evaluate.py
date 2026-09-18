import json

import numpy as np

from flybrain.circuit import build_circuit
from flybrain.evaluate import (evaluate_run, input_bypass, iters_to_threshold, pooled_gap_claim, summarize, make_test_sets)
from flybrain.io import load_edges, load_neurons
from flybrain.train import build, train


def _circuit(d):
    neurons = load_neurons(d)
    return build_circuit(neurons, load_edges(neurons["body"].to_numpy(), d))


def test_iters_to_threshold(tmp_path):
    p = tmp_path / "metrics.jsonl"
    rows = [{"event": "eval", "iter": i, "val_accuracy": a} for i, a in [(10, 0.2), (20, 0.95), (30, 0.85), (40, 0.97)]]
    p.write_text("\n".join(json.dumps(r) for r in rows + [{"event": "done"}]))
    assert iters_to_threshold(p, 0.9) == 20
    assert iters_to_threshold(p, 0.99) is None


def test_pooled_gap_claim_requires_two_pooled_sds():
    assert pooled_gap_claim([0.95, 0.96, 0.97], [0.80, 0.81, 0.82])["claim"] is True
    assert pooled_gap_claim([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])["claim"] is False
    assert pooled_gap_claim([0.9, 0.7, 0.8], [0.75, 0.85, 0.65])["claim"] is False


def test_evaluate_run_bands_lesion_and_bypass(tmp_path, synthetic_data_dir):
    c = _circuit(synthetic_data_dir)
    calib = {"gain": 4.0, "bias": 0.02}
    train("real", 0, tmp_path / "real_s0", circuit=c, calib=calib, iters=2, batch=4, n_frames=12, eval_every=1,
          val_n=8, log=lambda s: None)
    model, _, _ = build(c, "real", 0, calib)
    kind, side = model.input_kind.numpy(), model.input_side.numpy()
    sets = make_test_sets(kind, side, n_per_band=12, n_frames=12)
    assert set(sets) == {"fast", "trained", "slow"}
    res = evaluate_run(tmp_path / "real_s0", c, calib, sets, lesion=True)
    assert set(res["bands"]) == {"fast", "trained", "slow"} and 0 <= res["overall"]["accuracy"] <= 1
    assert "lesion" in res and set(res["lesion"]["bands"]) == {"fast", "trained", "slow"}
    assert res["variant"] == "real" and res["seed"] == 0
    bypass = input_bypass(kind, side, sets, n_train=64, n_frames=12, steps=20)
    assert set(bypass) == {"fast", "trained", "slow", "overall"}


def test_summarize_groups_by_variant():
    runs = [{"variant": v, "seed": s, "overall": {"accuracy": a, "false_alarm": 0.0, "latency_ms_median": 150.0},
             "iters_to_90": i} for v, s, a, i in [("real", 0, 1.0, 20), ("real", 1, 0.98, 30), ("shuffled", 0, 0.9, 60),
                                                  ("shuffled", 1, 0.92, 50)]]
    s = summarize(runs, bypass={"overall": {"accuracy": 0.99}})
    assert s["variants"]["real"]["accuracy"]["mean"] == 0.99 and s["variants"]["real"]["n"] == 2
    assert s["variants"]["shuffled"]["iters_to_90"]["mean"] == 55
    assert "real_vs_shuffled" in s["claims"] and s["input_bypass"]["overall"]["accuracy"] == 0.99
