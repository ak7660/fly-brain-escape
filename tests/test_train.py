import json

import numpy as np
import pytest
import torch

from flybrain import config
from flybrain.circuit import build_circuit
from flybrain.io import load_edges, load_neurons
from flybrain.train import decide, episode_loss, episode_metrics, train


def _circuit(d):
    neurons = load_neurons(d)
    return build_circuit(neurons, load_edges(neurons["body"].to_numpy(), d))


def _probs(seq):
    """seq of class indices with confidence 0.9 (0 = uncertain 'none')."""
    out = np.full((len(seq), 4), 0.1 / 3)
    for f, c in enumerate(seq):
        out[f, c] = 0.9
    return out


def test_decide_needs_consecutive_frames_of_one_action_class():
    assert decide(_probs([0, 1, 1, 1, 0])) == (1, 3)
    assert decide(_probs([1, 1, 2, 2, 1, 1])) == (0, None)  # never 3 in a row
    assert decide(_probs([0, 0, 0])) == (0, None)
    low = _probs([3, 3, 3]); low[:, 3] = 0.6
    assert decide(low) == (0, None)  # below DECISION_P
    assert decide(_probs([2, 2, 2, 2]), mask=np.array([1, 1, 0, 1], bool)) == (0, None)  # masked after collision


def test_episode_metrics_accuracy_false_alarms_and_latency():
    frames = 10
    logits = np.zeros((3, frames, 4)); logits[..., 0] = 5  # confident 'none'
    logits[0, 4:, 1] = 10                                     # episode 0: dodge_right from frame 4 → fires at 6
    logits[2, 1:, 3] = 10                                     # episode 2 (empty): false alarm
    eps = {"episode_class": np.array([1, 2, 0]), "mask": np.ones((3, frames), bool),
           "threats": [[type("T", (), {"t_collision": 9 / config.FPS})()], [type("T", (), {"t_collision": 9 / config.FPS})()], []]}
    m = episode_metrics(logits, eps)
    assert m["accuracy"] == 1 / 3
    assert m["false_alarm"] == 1.0
    assert m["latency_ms_median"] == pytest.approx((9 - 6) / config.FPS * 1000)
    assert np.array(m["confusion"]).tolist()[1][1] == 1 and np.array(m["confusion"]).tolist()[2][0] == 1


def test_episode_loss_weights_onset_frames_and_ignores_masked():
    logits = torch.zeros(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[0, 2, 2]]); mask = torch.tensor([[True, True, False]])
    loss = episode_loss(logits, labels, mask, onset_weight=3.0)
    assert torch.isclose(loss, torch.log(torch.tensor(4.0)))  # uniform logits → CE = log 4 regardless of weights
    loss.backward()
    assert torch.all(logits.grad[0, 2] == 0)


def test_train_smoke_writes_metrics_checkpoint_and_respects_readout_only(tmp_path, synthetic_data_dir):
    c = _circuit(synthetic_data_dir)
    calib = {"gain": 4.0, "bias": 0.02}
    kw = dict(circuit=c, calib=calib, iters=3, batch=4, n_frames=12, eval_every=2, val_n=8, log=lambda s: None)
    out = train("real", seed=0, out_dir=tmp_path / "real", **kw)
    lines = [json.loads(l) for l in (tmp_path / "real" / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["event"] == "done" and any(l["event"] == "eval" for l in lines)
    assert (tmp_path / "real" / "ckpt_best.pt").exists() and out["best_val_accuracy"] >= 0

    ro = train("readout_only", seed=0, out_dir=tmp_path / "ro", **kw)
    state = torch.load(tmp_path / "ro" / "ckpt_best.pt")
    assert torch.all(state["pre_scale"] == 0) and torch.all(state["post_scale"] == 0)
    assert json.loads((tmp_path / "ro" / "config.json").read_text())["variant"] == "readout_only"

    sh = train("shuffled", seed=0, out_dir=tmp_path / "sh", **kw)
    assert json.loads((tmp_path / "sh" / "config.json").read_text())["shuffle_seed"] == 1000
