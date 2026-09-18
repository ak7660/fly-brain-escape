"""Evaluate trained runs on held-out looming episodes; controls and pre-registered claims → results/summary.json.

Test bands (never used for training or validation):
  fast     l/v 15–20 ms (faster looms than trained)
  trained  l/v 20–60 ms (training range, new episodes)
  slow     l/v 60–80 ms (slower looms than trained)
Controls: readout_only and shuffled runs (see flybrain.train), lesion (inputs silenced) of each real run,
and input_bypass (logistic regression straight from the input cells, no connectome).
Claim rule (CLAUDE.md): "real beats X" only if mean(real) − mean(X) > 2 pooled SDs.
"""
import json
from pathlib import Path

import numpy as np
import torch

from flybrain import config
from flybrain.circuit import Circuit
from flybrain.stimulus import sample_episodes
from flybrain.train import build, episode_metrics, predict, resting_state

TEST_SEED = 20_000
BANDS = {"fast": (0.015, 0.02), "trained": (0.02, 0.06), "slow": (0.06, 0.08)}


def make_test_sets(kind, side, n_per_band=400, n_frames=72):
    return {band: sample_episodes(n_per_band, kind, side, np.random.default_rng(TEST_SEED + i), n_frames=n_frames,
                                  lv_ranges=(lv,))
            for i, (band, lv) in enumerate(BANDS.items())}


def _merge(sets):
    keys = ("drive", "labels", "mask", "episode_class")
    merged = {k: np.concatenate([s[k] for s in sets.values()]) for k in keys}
    merged["threats"] = [t for s in sets.values() for t in s["threats"]]
    return merged


def evaluate_model(model, sets, lesion=False):
    r0 = resting_state(model)
    bands = {}
    logits_all = []
    for band, eps in sets.items():
        drive = eps["drive"] * 0 if lesion else eps["drive"]
        logits = predict(model, {**eps, "drive": drive}, r0)
        bands[band] = episode_metrics(logits, eps)
        logits_all.append(logits)
    return {"bands": bands, "overall": episode_metrics(np.concatenate(logits_all), _merge(sets))}


def iters_to_threshold(metrics_path, threshold=0.9):
    for line in Path(metrics_path).read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "eval" and row["val_accuracy"] >= threshold:
            return row["iter"]
    return None


def evaluate_run(run_dir, circuit, calib, sets, lesion=False):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    model, _, _ = build(circuit, cfg["variant"], cfg["seed"], calib)
    model.load_state_dict(torch.load(run_dir / "ckpt_best.pt"))
    with torch.no_grad():
        res = {"variant": cfg["variant"], "seed": cfg["seed"], "run": run_dir.name, **evaluate_model(model, sets),
               "iters_to_90": iters_to_threshold(run_dir / "metrics.jsonl", 0.9),
               "gain": float(torch.exp(model.log_gain))}
        if lesion:
            res["lesion"] = evaluate_model(model, sets, lesion=True)
    return res


def input_bypass(kind, side, sets, n_train=2000, n_frames=72, steps=300, seed=0):
    """Upper-bound baseline: logistic regression on each input cell's peak drive → episode class."""
    torch.manual_seed(seed)
    train_eps = sample_episodes(n_train, kind, side, np.random.default_rng(30_000), n_frames=n_frames)
    feats = lambda eps: torch.as_tensor(eps["drive"].max(1))
    x, y = feats(train_eps), torch.as_tensor(train_eps["episode_class"])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    clf = torch.nn.Linear(x.shape[1], len(config.CLASSES))
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(clf((x - mu) / sd), y).backward()
        opt.step()
    out, correct, total = {}, 0, 0
    with torch.no_grad():
        for band, eps in sets.items():
            pred = clf((feats(eps) - mu) / sd).argmax(1).numpy()
            hits = int((pred == eps["episode_class"]).sum())
            out[band] = {"accuracy": hits / len(pred)}
            correct, total = correct + hits, total + len(pred)
    out["overall"] = {"accuracy": correct / total}
    return out


def _stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sd": None, "values": values}
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": values}


def pooled_gap_claim(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    pooled = float(np.sqrt((sd(a) ** 2 + sd(b) ** 2) / 2))
    gap = float(a.mean() - b.mean())
    return {"gap": gap, "pooled_sd": pooled, "claim": bool(gap > 2 * pooled and gap > 0)}


def summarize(runs, bypass=None, lesion=None):
    variants = {}
    for v in sorted({r["variant"] for r in runs}):
        rs = [r for r in runs if r["variant"] == v]
        variants[v] = {"n": len(rs),
                       "accuracy": _stats([r["overall"]["accuracy"] for r in rs]),
                       "false_alarm": _stats([r["overall"]["false_alarm"] for r in rs]),
                       "latency_ms_median": _stats([r["overall"]["latency_ms_median"] for r in rs]),
                       "iters_to_90": _stats([r["iters_to_90"] for r in rs])}
        if all("bands" in r for r in rs):
            variants[v]["bands"] = {b: _stats([r["bands"][b]["accuracy"] for r in rs]) for b in BANDS}
    claims = {}
    real = [r for r in runs if r["variant"] == "real"]
    for other in ("shuffled", "readout_only"):
        rest = [r for r in runs if r["variant"] == other]
        if real and rest:
            claims[f"real_vs_{other}"] = {
                "accuracy": pooled_gap_claim([r["overall"]["accuracy"] for r in real],
                                             [r["overall"]["accuracy"] for r in rest]),
                # fewer iterations is better → compare negated values; unreached counts as the full budget
                "learning_speed": pooled_gap_claim([-(r["iters_to_90"] or 10**4) for r in real],
                                                   [-(r["iters_to_90"] or 10**4) for r in rest]),
            }
    return {"variants": variants, "claims": claims, "input_bypass": bypass, "lesion": lesion}


def main():
    circuit = Circuit.load(config.CACHE_DIR / "circuit_v1.npz")
    calib = json.loads((config.ROOT / "results" / "calibration.json").read_text())["chosen"]
    probe, _, _ = build(circuit, "real", 0, calib)
    kind, side = probe.input_kind.numpy(), probe.input_side.numpy()
    sets = make_test_sets(kind, side)
    runs = []
    for run_dir in sorted((config.ROOT / "results" / "runs").glob("*_s*")):
        if not (run_dir / "ckpt_best.pt").exists():
            continue
        res = evaluate_run(run_dir, circuit, calib, sets, lesion=run_dir.name.startswith("real_"))
        runs.append(res)
        print(f"{run_dir.name:18s} acc {res['overall']['accuracy']:.3f} false_alarm {res['overall']['false_alarm']:.3f} "
              f"latency {res['overall']['latency_ms_median']} ms iters_to_90 {res['iters_to_90']}", flush=True)
    lesions = [r["lesion"]["overall"]["accuracy"] for r in runs if "lesion" in r]
    bypass = input_bypass(kind, side, sets)
    summary = summarize(runs, bypass=bypass, lesion={"accuracy": _stats(lesions)})
    summary["runs"] = runs
    summary["test_bands"] = {b: {"lv_s": list(lv), "n": len(sets[b]["episode_class"])} for b, lv in BANDS.items()}
    out = config.ROOT / "results" / "summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("claims", "input_bypass", "lesion")}, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
