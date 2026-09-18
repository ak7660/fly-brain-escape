"""Export a trained or calibrated circuit as flybrain-bundle v1 for the browser, plus the JS parity fixture."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from flybrain import config
from flybrain.bundle import read_bundle, world_um, write_bundle
from flybrain.circuit import Circuit
from flybrain.model import RateModel
from flybrain.sim_numpy import NumpySim
from flybrain.stimulus import Threat, input_drive

DEFAULT_OUT = config.ROOT / "web" / "assets" / "circuit_v1"
FIXTURE = config.ROOT / "web" / "tests" / "fixtures" / "parity_v1.json"


def _exact(x):
    # full float64 precision (repr round-trips exactly): the JS parity test uses absolute tolerances, and trained
    # logits reach ~10, where 7 significant digits alone would already cost 5e-6
    return [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]


def arrays_from_bundle(data):
    io, csr, neu = data["io.bin"], data["csr.bin"], data["neurons.bin"]
    return {"row_ptr": csr["row_ptr"], "col": csr["col"], "val": csr["val"], "bias": neu["bias"],
            "alpha": neu["alpha"], "input_idx": io["input_idx"], "input_kind": io["input_kind"],
            "input_side": io["input_side"], "input_gain": io["input_gain"], "dn_idx": io["dn_idx"],
            "W_out": io["W_out"].reshape(len(config.CLASSES), -1), "b_out": io["b_out"]}


def role_display_scale(sim, r0, circuit, input_kind, input_side, frames=60):
    """p90 over neurons of the peak rise above rest during left/front/right probe looms, per role.

    The browser scene shows activity as change from rest divided by this scale, so each layer's typical strong
    response is visible whatever the trained model's overall activity level is. Floored at 1e-3.
    """
    peak = np.zeros(sim.n)
    for az in (-90.0, 0.0, 90.0):
        threat = [Threat(az, 0.04, 0.1, 0.7)]
        drive = np.stack([input_drive(f / config.FPS, threat, input_kind, input_side) for f in range(frames)])
        rates = sim.run(drive, r0=r0, record=True)["rates"]
        peak = np.maximum(peak, (rates - r0).max(0))
    return [max(float(np.percentile(peak[circuit.role == role], 90)), 1e-3) if (circuit.role == role).any() else 1e-3
            for role in range(4)]


def build_bundle(circuit, model, positions_nm, dust_nm, shells_nm, out_dir, variant, extra=None):
    arrays = model.export_arrays(dtype=np.float32)
    sim64 = NumpySim(arrays, dtype=np.float64)
    r0 = sim64.steady_state().astype(np.float32)
    dust_nm = np.asarray(dust_nm, dtype=np.float64)
    centre = (dust_nm.min(0) + dust_nm.max(0)) / 2
    pos = world_um(positions_nm, centre)
    dust = np.round(world_um(dust_nm, centre) / config.DUST_SCALE_UM)
    if np.abs(dust).max() > 32767:
        raise ValueError("dust coordinates overflow int16; raise DUST_SCALE_UM")
    vis = np.argsort(-np.abs(arrays["val"]), kind="stable")[:config.VIS_EDGES]
    type_id = model.type_id.numpy().astype(np.uint16)

    files = {
        "neurons.bin": [("pos", "f32", pos), ("bias", "f32", arrays["bias"]), ("alpha", "f32", arrays["alpha"]),
                        ("r0", "f32", r0)],
        "neuron_meta.bin": [("layer", "u8", circuit.layer), ("role", "u8", circuit.role), ("sign", "i8", circuit.sign),
                            ("side", "u8", circuit.side), ("type_id", "u16", type_id)],
        "csr.bin": [("row_ptr", "u32", arrays["row_ptr"]), ("col", "u32", arrays["col"]), ("val", "f32", arrays["val"])],
        "io.bin": [("input_idx", "u32", arrays["input_idx"]), ("input_kind", "u32", arrays["input_kind"]),
                   ("input_side", "u32", arrays["input_side"]), ("input_gain", "f32", arrays["input_gain"]),
                   ("dn_idx", "u32", arrays["dn_idx"]), ("W_out", "f32", arrays["W_out"]),
                   ("b_out", "f32", arrays["b_out"])],
        "vis_edges.bin": [("edge_idx", "u32", np.sort(vis))],
        "dust.bin": [("pos", "i16", dust)],
    }
    for name, (verts, faces) in shells_nm.items():
        files[f"shell_{name}.bin"] = [("n_verts", "u32", [len(verts)]), ("verts", "f32", world_um(verts, centre)),
                                      ("n_faces", "u32", [len(faces)]), ("faces", "u32", faces)]
    manifest = {
        "format": "flybrain-bundle", "version": 1, "variant": variant,
        "source": {"dataset": "MaleCNS v1.0 minconf-0.5 traced-only", "circuit": circuit.meta},
        "sim": {"fps": config.FPS, "substeps": config.SUBSTEPS, "dt_sub": config.DT_SUB, "r_max": config.R_MAX,
                "warmup_frames": config.WARMUP_FRAMES},
        "stimulus": {"input_kinds": list(config.INPUT_KINDS), "rf_centers_deg": list(config.RF_CENTERS_DEG),
                     "rf_sigma_deg": config.RF_SIGMA_DEG, "theta_sat_deg": config.THETA_SAT_DEG,
                     "thetadot_sat_dps": config.THETADOT_SAT_DPS, "label_onset_deg": config.LABEL_ONSET_DEG,
                     "front_half_width_deg": config.FRONT_HALF_WIDTH_DEG, "noise_sd": config.NOISE_SD,
                     "approach_s": config.APPROACH_S},
        "classes": list(config.CLASSES), "decision": {"p": config.DECISION_P, "frames": config.DECISION_FRAMES},
        "counts": {"neurons": len(circuit.body), "edges": len(circuit.pre), "inputs": int((circuit.role == 0).sum()),
                   "dn": int((circuit.role == 2).sum()), "mn": int((circuit.role == 3).sum()),
                   "types": len(model.type_names), "vis_edges": int(len(vis)), "dust": int(len(dust))},
        "units": {"world": "µm", "axes": "X=−x (fly left at −X), Y=−z (brain up), Z=−y",
                  "centre_nm": [float(v) for v in centre], "dust_scale_um": config.DUST_SCALE_UM},
        "nt_policy": {**config.NT_SIGN, "other": 0},
        "params": {"gain": float(torch.exp(model.log_gain.detach())), "trained": False},
        "metrics": None,
        "viz": {"role_ref": role_display_scale(sim64, r0.astype(np.float64), circuit,
                                               arrays["input_kind"].astype(int), arrays["input_side"].astype(int)),
                "role_ref_note": "p90 peak rise above rest per role (input, hidden, DN, motor) for L/F/R probe looms"},
        **(extra or {}),
    }
    manifest = write_bundle(out_dir, files, manifest)
    info = {"types": model.type_names, "instance": [str(s) for s in circuit.instance],
            "superclass": [str(s) for s in circuit.superclass], "body": [str(int(b)) for b in circuit.body]}
    (Path(out_dir) / "neuron_info.json").write_text(json.dumps(info))
    return manifest


def make_parity_fixture(bundle_dir, out_path, frames=180):
    manifest, data = read_bundle(bundle_dir)
    arrays = arrays_from_bundle(data)
    sim = NumpySim(arrays, dtype=np.float64)
    kind, side = arrays["input_kind"].astype(int), arrays["input_side"].astype(int)
    threats = [Threat(az, 0.04, f / config.FPS, f / config.FPS + 0.5) for az, f in ((-90.0, 5), (0.0, 60), (90.0, 110))]
    drive = np.stack([input_drive(f / config.FPS, threats, kind, side) for f in range(frames)])
    out = sim.run(drive, r0=data["neurons.bin"]["r0"], record=True)
    sample = np.linspace(0, sim.n - 1, 64).astype(int)
    fixture = {
        "bundle": Path(bundle_dir).name, "frames": frames, "tolerance": 1e-3,
        "threats": [asdict(t) for t in threats],
        "drive_first8": [_exact(row[:8]) for row in drive],
        "logits": [_exact(row) for row in out["logits"]],
        "sample_neurons": sample.tolist(),
        "rates_sampled": [_exact(row[sample]) for row in out["rates"]],
        "rates_full": {str(f): _exact(out["rates"][f]) for f in (0, frames // 2 - 1, frames - 1)},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(fixture))
    return json.loads(Path(out_path).read_text())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--variant", default="real")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    from flybrain.shells import fetch_shell, load_mesh
    circuit = Circuit.load(config.CACHE_DIR / "circuit_v1.npz")
    calib = json.loads((config.ROOT / "results" / "calibration.json").read_text())["chosen"]
    model = RateModel(circuit, gain=calib["gain"], bias=calib["bias"])
    extra = {"params": {**calib, "trained": False}}
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        # report the TRAINED gain, not the calibration starting point (calib is kept for provenance)
        extra = {"params": {"gain": float(torch.exp(model.log_gain.detach())), "bias": calib["bias"],
                            "calibration": calib, "trained": True, "checkpoint": str(args.checkpoint)}}
        summary_path = config.ROOT / "results" / "summary.json"
        run_name = Path(args.checkpoint).parent.name
        if summary_path.exists():
            runs = [r for r in json.loads(summary_path.read_text())["runs"] if r["run"] == run_name]
            if runs:
                r = runs[0]
                extra["metrics"] = {"run": run_name, "heldout_accuracy": r["overall"]["accuracy"],
                                    "false_alarm": r["overall"]["false_alarm"],
                                    "latency_ms_median": r["overall"]["latency_ms_median"],
                                    "bands": {b: r["bands"][b]["accuracy"] for b in r["bands"]},
                                    "iters_to_90": r["iters_to_90"]}
    cent = np.load(config.CACHE_DIR / "centroids.npz")
    at = dict(zip(cent["body"].tolist(), range(len(cent["body"]))))
    positions = np.array([cent["all_mean"][at[int(b)]] for b in circuit.body]) * config.VOXEL_NM
    if not np.isfinite(positions).all():
        raise ValueError("circuit neuron without a synapse centroid")
    dust = cent["all_mean"][np.isfinite(cent["all_mean"][:, 0])] * config.VOXEL_NM
    shells = {name: load_mesh(fetch_shell(name)) for name in ("brain", "vnc")}
    manifest = build_bundle(circuit, model, positions, dust, shells, args.out, args.variant, extra)
    make_parity_fixture(args.out, FIXTURE)
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"bundle {args.out}: {manifest['counts']} · {total / 1e6:.2f} MB · fixture {FIXTURE}")


if __name__ == "__main__":
    main()
