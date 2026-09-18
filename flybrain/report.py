"""Figures for docs/figures. P1: circuit composition, Giant Fiber inputs, anatomy.

Chart colors are the dataviz reference palette's dark steps, validated on the #05070a surface
(scripts/validate_palette.js: all checks pass). They are deliberately calmer than the neon
glow colors of the live 3D view. Each figure also writes a CSV with the plotted numbers
(the table view).
"""
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flybrain import config
from flybrain.circuit import Circuit

OUT = config.ROOT / "docs" / "figures"
SURFACE = "#05070a"
INK = {"primary": "#e6e8eb", "secondary": "#a3a9b3", "muted": "#6b7280", "grid": "#1c2129"}
ORANGE, BLUE, AQUA = "#d95926", "#3987e5", "#199e70"
SIGN = [(1, "excitatory (acetylcholine)", ORANGE), (-1, "inhibitory (GABA, glutamate, histamine)", BLUE),
        (0, "no fast output (modulatory / unclear)", AQUA)]
LAYERS = ["input\nLPLC2 + LC4", "hidden\n1 hop", "hidden\n2–3 hops", "descending\nneurons", "motor\nneurons"]


def _style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK["grid"])
    ax.tick_params(colors=INK["secondary"], length=0, labelsize=9)
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK["secondary"], **kw)
    return leg


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def circuit_layers(c, out):
    counts = {s: np.array([int(((c.layer == i) & (c.sign == s)).sum()) for i in range(5)]) for s, _, _ in SIGN}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    _style(ax, fig)
    bottom = np.zeros(5)
    for s, label, color in SIGN:
        # 2px surface-colored edge = the gap between stacked segments
        ax.bar(range(5), counts[s], bottom=bottom, width=0.55, color=color, label=label,
               edgecolor=SURFACE, linewidth=2)
        bottom += counts[s]
    for i, total in enumerate(bottom):
        ax.text(i, total + 40, f"{int(total):,}", ha="center", va="bottom", fontsize=9, color=INK["primary"])
    ax.set_xticks(range(5), LAYERS)
    ax.set_yticks([])
    ax.set_title(f"Looming-escape circuit: {len(c.body):,} neurons, {len(c.pre):,} connections",
                 loc="left", fontsize=11, pad=14, color=INK["primary"])
    _legend(ax, loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(out / "p1_circuit_layers.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    _write_csv(out / "p1_circuit_layers.csv", ["layer", "excitatory", "inhibitory", "no_output", "total"],
               [[LAYERS[i].replace("\n", " "), counts[1][i], counts[-1][i], counts[0][i], int(bottom[i])]
                for i in range(5)])


def gf_inputs(c, out, top_n=15):
    gf = np.flatnonzero(c.type == "DNp01")
    into = np.isin(c.post, gf)
    by_type = Counter()
    for p, w in zip(c.pre[into], c.weight[into]):
        by_type[str(c.type[p])] += int(w)
    top = by_type.most_common(top_n)
    names, values = [t for t, _ in top][::-1], [w for _, w in top][::-1]
    colors = [AQUA if t in config.SOURCE_TYPES else ORANGE for t in names]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    _style(ax, fig)
    ax.barh(names, values, height=0.6, color=colors)
    for y, v in enumerate(values):
        ax.text(v + max(values) * 0.01, y, f"{v:,}", va="center", fontsize=8, color=INK["secondary"])
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK["primary"])
    ax.set_title("Strongest inputs to the Giant Fiber (DNp01)\nsynapses from each cell type, within the circuit",
                 loc="left", fontsize=10.5, pad=10, color=INK["primary"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=AQUA), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["looming detector (model input)", "other circuit neuron"], frameon=False, fontsize=8,
              labelcolor=INK["secondary"], loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "p1_gf_inputs.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    _write_csv(out / "p1_gf_inputs.csv", ["presynaptic_type", "synapses_onto_DNp01"], top)


def anatomy(c, out):
    cent = np.load(config.CACHE_DIR / "centroids.npz")
    at = dict(zip(cent["body"].tolist(), range(len(cent["body"]))))
    xyz = np.array([cent["all_mean"][at[b]] for b in c.body])
    dust = cent["all_mean"][np.isfinite(cent["all_mean"][:, 0])]
    groups = [("output: descending + motor neurons", c.role >= 2, BLUE, 3),
              ("hidden interneurons", c.role == 1, ORANGE, 2.5),
              ("input: LPLC2 + LC4 looming detectors", c.role == 0, AQUA, 4)]
    fig, ax = plt.subplots(figsize=(5.2, 7.6))
    _style(ax, fig)
    ax.axis("off")
    # front view: x (left/right) vs z (anterior brain at top → VNC at bottom)
    ax.scatter(dust[::6, 0], dust[::6, 2], s=0.15, c=INK["muted"], alpha=0.25, linewidths=0, rasterized=True)
    for label, mask, color, size in groups:
        ax.scatter(xyz[mask, 0], xyz[mask, 2], s=size, c=color, linewidths=0, label=f"{label} ({int(mask.sum()):,})")
    ax.invert_yaxis()
    ax.invert_xaxis()  # fly's left (larger x) on the viewer's right, as seen from the front
    ax.set_aspect("equal")
    ax.set_title("Where the circuit sits in the male fly CNS", loc="left", fontsize=11, color=INK["primary"])
    _legend(ax, loc="lower center", bbox_to_anchor=(0.5, -0.06), markerscale=4)
    fig.tight_layout()
    fig.savefig(out / "p1_anatomy.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def p1(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    c = Circuit.load(config.CACHE_DIR / "circuit_v1.npz")
    circuit_layers(c, out)
    gf_inputs(c, out)
    anatomy(c, out)
    print(f"wrote P1 figures to {out}")


if __name__ == "__main__":
    p1()


ROLE_LINES = [(0, "input (LPLC2+LC4)", AQUA), (1, "hidden", ORANGE), (2, "descending", BLUE), (3, "motor", "#d55181")]
DIRECTION_TITLES = {"L": "threat from the left", "F": "threat from the front", "R": "threat from the right"}


def p2_calibration(detail, out=OUT, t_collision=0.8):
    """Change from resting rate over time: one row per neuron role (own y-scale), one column per threat direction."""
    out.mkdir(parents=True, exist_ok=True)
    traces = detail["traces"]
    n_frames = len(traces["L"][0])
    t = np.arange(n_frames) / config.FPS
    fig, axes = plt.subplots(len(ROLE_LINES), 3, figsize=(10, 7.2), sharex=True, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    rows = []
    for i, (role, label, color) in enumerate(ROLE_LINES):
        for j, key in enumerate(("L", "F", "R")):
            ax = axes[i, j]
            _style(ax, fig)
            y = np.asarray(traces[key][role])
            dy = y - y[0]  # the probe starts from the resting (steady) state
            ax.fill_between(t, 0, dy, color=color, alpha=0.10, linewidth=0)
            ax.plot(t, dy, color=color, linewidth=2, solid_capstyle="round")
            ax.axvline(t_collision, color=INK["muted"], linewidth=1, linestyle=(0, (3, 3)))
            ax.grid(axis="y", color=INK["grid"], linewidth=1)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(DIRECTION_TITLES[key], loc="left", fontsize=10, color=INK["primary"])
                if j == 2:
                    ax.text(t_collision, 1.0, " collision", transform=ax.get_xaxis_transform(), color=INK["muted"],
                            fontsize=8, va="top")
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK["primary"], rotation=0, ha="right", va="center", labelpad=8)
            if i == len(ROLE_LINES) - 1:
                ax.set_xlabel("time (s)", fontsize=9)
            rows += [[f"{ti:.4f}", key, label, f"{yi:.6g}", f"{di:.6g}"] for ti, yi, di in zip(t, y, dy)]
    fig.suptitle(f"Calibration: a looming threat's signal reaches the descending neurons\n"
                 f"mean change in firing rate from rest per neuron group · gain {detail['gain']:g}, bias {detail['bias']:g} "
                 f"· each row has its own scale", x=0.01, ha="left", fontsize=11, color=INK["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "p2_calibration.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    _write_csv(out / "p2_calibration.csv", ["time_s", "direction", "role", "mean_rate", "change_from_rest"], rows)


VARIANT_STYLE = [("real", "real wiring", ORANGE), ("shuffled", "shuffled wiring", BLUE),
                 ("readout_only", "readout only (frozen circuit)", AQUA)]


def p3_results(summary_path=config.ROOT / "results" / "summary.json", runs_dir=config.ROOT / "results" / "runs", out=OUT):
    """Learning curves (every seed) and held-out accuracy/latency (every seed as a dot, mean as a bar tick)."""
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(summary_path).read_text())
    rows = []

    # ---- 1. learning curves
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    _style(ax, fig)
    for key, label, color in VARIANT_STYLE:
        curves = []
        for run in sorted(Path(runs_dir).glob(f"{key}_s*")):
            evals = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
            evals = [e for e in evals if e["event"] == "eval"]
            it = np.array([e["iter"] for e in evals]); acc = np.array([e["val_accuracy"] for e in evals])
            ax.plot(it, acc, color=color, linewidth=1, alpha=0.35)
            curves.append(acc)
            rows += [[key, run.name, int(i), float(a)] for i, a in zip(it, acc)]
        mean = np.mean(curves, axis=0)
        iters90 = summary["variants"][key]["iters_to_90"]["mean"]
        ax.plot(it, mean, color=color, linewidth=2, solid_capstyle="round",
                label=f"{label} · 90% after ~{iters90:.0f} iterations")
    ax.axhline(0.9, color=INK["muted"], linewidth=1, linestyle=(0, (3, 3)))
    ax.text(3, 0.915, "90%", color=INK["muted"], fontsize=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK["secondary"], loc="lower right", handlelength=1.6)
    ax.set_xlim(0, 300); ax.set_ylim(0, 1.02)
    ax.set_xlabel("training iteration", fontsize=9); ax.set_ylabel("validation accuracy", fontsize=9)
    ax.grid(axis="y", color=INK["grid"], linewidth=1); ax.set_axisbelow(True)
    ax.set_title("Real fly wiring learns the escape task ~2.6× faster than shuffled wiring\n"
                 "validation accuracy per training run (thin: each of 3 seeds · thick: mean)",
                 loc="left", fontsize=10.5, color=INK["primary"])
    fig.tight_layout()
    fig.savefig(out / "p3_learning_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    _write_csv(out / "p3_learning_curves.csv", ["variant", "run", "iter", "val_accuracy"], rows)

    # ---- 2. held-out accuracy by loom speed + decision latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    bands = [("fast", "fast looms\n(untrained speeds)"), ("trained", "training speeds\n(new episodes)"),
             ("slow", "slow looms\n(untrained speeds)")]
    table = []
    for ax in (ax1, ax2):
        _style(ax, fig)
        ax.grid(axis="y", color=INK["grid"], linewidth=1); ax.set_axisbelow(True)
    offsets = {"real": -0.22, "shuffled": 0.0, "readout_only": 0.22}
    for key, label, color in VARIANT_STYLE:
        runs = [r for r in summary["runs"] if r["variant"] == key]
        for b, (band, _) in enumerate(bands):
            vals = [r["bands"][band]["accuracy"] for r in runs]
            x = b + offsets[key]
            ax1.scatter([x] * len(vals), vals, s=36, color=color, edgecolors=SURFACE, linewidths=2, zorder=3,
                        label=label if b == 0 else None)
            ax1.plot([x - 0.1, x + 0.1], [np.mean(vals)] * 2, color=color, linewidth=1, alpha=0.6, zorder=2)
            table += [[key, band, r["run"], r["bands"][band]["accuracy"]] for r in runs]
        lat = [r["overall"]["latency_ms_median"] for r in runs]
        x = [k for k, *_ in VARIANT_STYLE].index(key)
        ax2.scatter([x] * len(lat), lat, s=36, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax2.plot([x - 0.22, x + 0.22], [np.mean(lat)] * 2, color=color, linewidth=1, alpha=0.6, zorder=2)
    chance = summary["lesion"]["accuracy"]["mean"]
    ax1.axhline(chance, color=INK["muted"], linewidth=1, linestyle=(0, (3, 3)))
    ax1.text(2.45, chance + 0.02, f"inputs silenced (lesion): {chance:.0%}", color=INK["muted"], fontsize=8, ha="right")
    ax1.set_xticks(range(3), [b[1] for b in bands], fontsize=8.5)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("held-out accuracy", fontsize=9)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK["secondary"], loc="lower right", markerscale=0.9)
    ax1.set_title("Held-out episodes (400 per band) · dot = one seed, line = mean", loc="left", fontsize=10,
                  color=INK["primary"])
    ax2.set_xticks(range(3), ["real", "shuffled", "readout\nonly"], fontsize=8.5)
    ax2.set_ylim(0, 340); ax2.set_ylabel("median decision time\nbefore contact (ms)", fontsize=9)
    ax2.set_title("Earlier is better", loc="left", fontsize=10, color=INK["primary"])
    fig.tight_layout()
    fig.savefig(out / "p3_heldout.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    _write_csv(out / "p3_heldout.csv", ["variant", "band", "run", "accuracy"], table)
