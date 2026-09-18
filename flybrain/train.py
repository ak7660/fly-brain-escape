"""Train the connectome rate model on the looming-escape task.

Variants (same data, optimiser and budget):
  real          all type-level parameters + readout trained on the real wiring
  readout_only  frozen calibrated circuit; only the DN readout is trained
  shuffled      degree/layer-preserving shuffled wiring (flybrain.shuffle), trained like `real`

CLI: uv run python -m flybrain.train --variant real --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import config
from flybrain.circuit import Circuit
from flybrain.model import RateModel
from flybrain.shuffle import shuffled_circuit
from flybrain.sim_numpy import NumpySim
from flybrain.stimulus import sample_episodes

VARIANTS = ("real", "readout_only", "shuffled")
VAL_SEED = 10_000


def decide(probs, p=config.DECISION_P, frames=config.DECISION_FRAMES, mask=None):
    """First frame where the same action class (≠ none) has prob > p for `frames` consecutive frames."""
    run_cls, run = 0, 0
    for f in range(len(probs)):
        if mask is not None and not mask[f]:
            break
        c = int(np.argmax(probs[f]))
        if c != 0 and probs[f, c] > p:
            run = run + 1 if c == run_cls else 1
            run_cls = c
            if run >= frames:
                return c, f
        else:
            run_cls, run = 0, 0
    return 0, None


def episode_metrics(logits, episodes):
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    classes = np.asarray(episodes["episode_class"])
    confusion = np.zeros((4, 4), dtype=int)
    latencies, correct = [], 0
    for i, true in enumerate(classes):
        pred, frame = decide(probs[i], mask=episodes["mask"][i])
        confusion[true, pred] += 1
        if pred == true:
            correct += 1
            if true != 0:
                latencies.append((episodes["threats"][i][0].t_collision - frame / config.FPS) * 1000.0)
    empty = classes == 0
    false_alarm = float(confusion[0, 1:].sum() / max(empty.sum(), 1))
    return {"accuracy": correct / len(classes), "false_alarm": false_alarm,
            "latency_ms_median": float(np.median(latencies)) if latencies else None,
            "confusion": confusion.tolist(), "n": int(len(classes))}


def episode_loss(logits, labels, mask, onset_weight=3.0):
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
    w = torch.where(labels.reshape(-1) > 0, onset_weight, 1.0) * mask.reshape(-1).to(ce.dtype)
    return (ce * w).sum() / w.sum()


def build(circuit, variant, seed, calib):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    shuffle_seed = None
    if variant == "shuffled":
        shuffle_seed = 1000 + seed
        circuit = shuffled_circuit(circuit, seed=shuffle_seed)
    torch.manual_seed(seed)
    model = RateModel(circuit, gain=calib["gain"], bias=calib["bias"], readout_seed=seed)
    if variant == "readout_only":
        for name, p in model.named_parameters():
            p.requires_grad_(name in ("W_out", "b_out"))
    return model, circuit, shuffle_seed


def resting_state(model):
    return torch.as_tensor(NumpySim(model.export_arrays(dtype=np.float64)).steady_state(), dtype=model.dtype)


def _batch(episodes, idx):
    return (torch.as_tensor(episodes["drive"][idx]), torch.as_tensor(episodes["labels"][idx]),
            torch.as_tensor(episodes["mask"][idx]))


@torch.no_grad()
def predict(model, episodes, r0, chunk=64):
    out = []
    for s in range(0, len(episodes["drive"]), chunk):
        logits, _ = model.simulate(torch.as_tensor(episodes["drive"][s:s + chunk]), r0=r0)
        out.append(logits.numpy())
    return np.concatenate(out)


def train(variant, seed, out_dir, circuit=None, calib=None, iters=300, batch=32, n_frames=72, lr=1e-2,
          eval_every=10, val_n=256, threads=2, log=print):
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    circuit = circuit if circuit is not None else Circuit.load(config.CACHE_DIR / "circuit_v1.npz")
    calib = calib if calib is not None else json.loads((config.ROOT / "results" / "calibration.json").read_text())["chosen"]
    model, used, shuffle_seed = build(circuit, variant, seed, calib)
    kind, side = model.input_kind.numpy(), model.input_side.numpy()
    rng = np.random.default_rng(seed)
    val = sample_episodes(val_n, kind, side, np.random.default_rng(VAL_SEED), n_frames=n_frames)
    (out_dir / "config.json").write_text(json.dumps({
        "variant": variant, "seed": seed, "shuffle_seed": shuffle_seed, "calib": calib, "iters": iters, "batch": batch,
        "n_frames": n_frames, "lr": lr, "val_seed": VAL_SEED, "val_n": val_n}, indent=1))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")

    def emit(record):
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    r0 = resting_state(model)
    best, t0 = -1.0, time.time()
    for it in range(1, iters + 1):
        ep = sample_episodes(batch, kind, side, rng, n_frames=n_frames)
        drive, labels, mask = _batch(ep, slice(None))
        logits, _ = model.simulate(drive, r0=r0)
        loss = episode_loss(logits, labels, mask)
        if variant != "readout_only":
            loss = loss + 1e-3 * (model.pre_scale.pow(2).mean() + model.post_scale.pow(2).mean())
        loss = loss + 1e-4 * model.W_out.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % eval_every == 0 or it == iters:
            if variant != "readout_only":
                r0 = resting_state(model)
            m = episode_metrics(predict(model, val, r0), val)
            record = {"event": "eval", "iter": it, "loss": float(loss.detach()), "val_accuracy": m["accuracy"],
                      "val_false_alarm": m["false_alarm"], "val_latency_ms_median": m["latency_ms_median"],
                      "gain": float(torch.exp(model.log_gain.detach())), "s_per_iter": (time.time() - t0) / it}
            emit(record)
            log(f"[{variant} s{seed}] iter {it} loss {record['loss']:.4f} val_acc {m['accuracy']:.3f} "
                f"false_alarm {m['false_alarm']:.3f} latency {m['latency_ms_median']} ms")
            if m["accuracy"] > best:
                best = m["accuracy"]
                torch.save(model.state_dict(), out_dir / "ckpt_best.pt")
    emit({"event": "done", "best_val_accuracy": best, "seconds": time.time() - t0})
    return {"best_val_accuracy": best, "out_dir": str(out_dir)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    out = config.ROOT / "results" / "runs" / f"{args.variant}_s{args.seed}"
    res = train(args.variant, args.seed, out, iters=args.iters, threads=args.threads,
                log=lambda s: print(s, flush=True))
    print(f"DONE {res}", flush=True)


if __name__ == "__main__":
    main()
