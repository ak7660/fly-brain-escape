"""Calibration gate: find the smallest global gain (and tonic bias) at which looming signal reliably reaches the DNs.

Gate v1 (pre-registered): stable baseline, mean-DN response ≥ max(3·noise_sd, 1e-3) for L/F/R looms, and the Giant
Fiber preferring frontal looms. On the real circuit v1 passed even at gain 1, where only 5–10% of DNs respond at all
(mean response 0.0012 ≈ 0.03% of R_MAX), so its absolute floor was too weak to mean "signal reaches the DNs".
Gate v2 (revised after seeing that, documented in docs/research.md) adds the purpose of calibration explicitly:
a non-silent tonic baseline (bias > 0) and ≥ MIN_DN_RECRUITMENT of DNs changing by > RECRUIT_DELTA before collision
in every direction. Gain stays a trainable parameter; calibration only picks the training starting point.
"""
import json
import math

import numpy as np
import torch

from flybrain import config
from flybrain.circuit import Circuit
from flybrain.model import RateModel
from flybrain.sim_numpy import NumpySim
from flybrain.stimulus import Threat, input_drive

DIRECTIONS = {"L": -90.0, "F": 0.0, "R": 90.0}
N_FRAMES = 72
PROBE = dict(lv=0.04, t_spawn=0.1, t_collision=0.8)
MIN_DN_RECRUITMENT = 0.25
RECRUIT_DELTA = 0.01


def _drive(threats, sim_kind, sim_side, n_frames=N_FRAMES):
    t = np.arange(n_frames) / config.FPS
    return np.stack([input_drive(ti, threats, sim_kind, sim_side) for ti in t])


def evaluate_setting(circuit, gain, bias, rng, record=False):
    model = RateModel(circuit, gain=gain, bias=bias, dtype=torch.float64)
    arrays = model.export_arrays(dtype=np.float64)
    sim = NumpySim(arrays)
    kind, side = arrays["input_kind"].astype(int), arrays["input_side"].astype(int)
    dn, gf = sim.dn_idx, np.flatnonzero(circuit.type == "DNp01")

    warm = sim.run(np.zeros((config.WARMUP_FRAMES, len(kind))), record=True)
    r_ss = warm["r"]
    max_drift = float(np.abs(warm["rates"][-1] - warm["rates"][-2]).max())
    frac_sat = float((r_ss >= 0.99 * config.R_MAX).mean())
    stable = bool(max_drift < 1e-3 and frac_sat < 0.05)

    noise_means = [sim.run(0.03 * rng.standard_normal((N_FRAMES, len(kind))), r0=r_ss, record=True)["rates"][:, dn].mean(1)
                   for _ in range(10)]
    noise_sd = float(np.std(noise_means))
    base = float(r_ss[dn].mean())

    t = np.arange(N_FRAMES) / config.FPS
    t15 = PROBE["t_collision"] - PROBE["lv"] / math.tan(math.radians(config.LABEL_ONSET_DEG / 2))
    window = (t >= t15) & (t <= t15 + 0.15)
    before = t < PROBE["t_collision"]
    response, responsive, gf_peak, recruited, traces = {}, {}, {}, {}, {}
    for key, az in DIRECTIONS.items():
        out = sim.run(_drive([Threat(az, **PROBE)], kind, side), r0=r_ss, record=True)
        dn_mean = out["rates"][:, dn].mean(1)
        response[key] = float(np.abs(dn_mean[window] - base).max())
        responsive[key] = bool(response[key] >= max(3 * noise_sd, 1e-3))
        gf_peak[key] = float(out["rates"][before][:, gf].mean(1).max())
        recruited[key] = float((np.abs(out["rates"][before][:, dn] - r_ss[dn]).max(0) > RECRUIT_DELTA).mean())
        if record:
            traces[key] = {role: out["rates"][:, circuit.role == role].mean(1) for role in range(4)}
    gf_front = bool(gf_peak["F"] > max(gf_peak["L"], gf_peak["R"]))
    result = {"gain": gain, "bias": bias, "stable": stable, "frac_saturated": frac_sat, "max_drift": max_drift,
              "noise_sd": noise_sd, "response": response, "responsive": responsive, "gf_peak": gf_peak,
              "gf_front_preferred": gf_front, "recruited": recruited,
              "tonic_active": float((r_ss > RECRUIT_DELTA).mean())}
    result["passed_v1"] = bool(stable and all(responsive.values()) and gf_front)
    result["passed"] = bool(result["passed_v1"] and bias > 0 and min(recruited.values()) >= MIN_DN_RECRUITMENT)
    if record:
        result["traces"] = traces
    return result


def run_sweep(circuit, gains=(1, 2, 4, 8, 16, 32, 64), biases=(0.0, 0.02, 0.05, 0.1), seed=0):
    settings = []
    for gain in gains:
        for bias in biases:
            res = evaluate_setting(circuit, float(gain), float(bias), np.random.default_rng(seed))
            settings.append({"gain": float(gain), "bias": float(bias), **res})
            print(f"gain {gain:>5} bias {bias:<5} passed_v1={res.get('passed_v1')} passed={res['passed']}", flush=True)
    passing = [s for s in settings if s["passed"]]
    chosen = min(passing, key=lambda s: (s["gain"], s["bias"])) if passing else None
    return {"settings": settings, "chosen": None if chosen is None else {"gain": chosen["gain"], "bias": chosen["bias"]}}


def main():
    circuit = Circuit.load(config.CACHE_DIR / "circuit_v1.npz")
    out = run_sweep(circuit)
    path = config.ROOT / "results" / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("chosen:", out["chosen"], "→", path)
    if out["chosen"]:
        from flybrain import report
        detail = evaluate_setting(circuit, out["chosen"]["gain"], out["chosen"]["bias"], np.random.default_rng(0),
                                  record=True)
        report.p2_calibration(detail)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
