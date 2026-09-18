"""3D neuron positions = centroids of each neuron's synapse partner rows (8 nm voxels), streamed from syn-partners.

syn-partners has one row per pre→post pair, so a presynaptic site with k postsynaptic partners
contributes k times to its body's pre centroid (pre centroids are weighted by partner count).
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from flybrain import config


def _lookup(bodies, ids):
    i = np.minimum(np.searchsorted(bodies, ids), len(bodies) - 1)
    return i, bodies[i] == ids


def compute_centroids(bodies, data_dir: Path = config.DATA_DIR, log=print, log_every=100):
    """Per-body pre/post/all synapse centroids in 8 nm voxels; `bodies` must be sorted and unique.

    Bodies without any traced-only synapse rows get NaN centroids (in MaleCNS v1.0: 203 of 164,606
    neurons, mostly cb_sensory, ENS and vnc_sensory), so exports must drop them or use a fallback.
    """
    bodies = np.asarray(bodies)
    if len(bodies) == 0 or np.any(np.diff(bodies) <= 0):
        raise ValueError("bodies must be non-empty and strictly increasing")
    n = len(bodies)
    sums = {"pre": np.zeros((n, 3)), "post": np.zeros((n, 3))}
    counts = {"pre": np.zeros(n, np.int64), "post": np.zeros(n, np.int64)}
    with pa.memory_map(str(Path(data_dir) / config.SYN_PARTNERS_TRACED)) as src:
        reader = ipc.open_file(src)
        nb = reader.num_record_batches
        for b in range(nb):
            batch = reader.get_batch(b)
            for side in ("pre", "post"):
                i, ok = _lookup(bodies, batch.column(f"body_{side}").to_numpy())
                ii = i[ok]
                counts[side] += np.bincount(ii, minlength=n)
                for k, ax in enumerate("xyz"):
                    xs = batch.column(f"{ax}_{side}").to_numpy()[ok].astype(np.float64)
                    sums[side][:, k] += np.bincount(ii, weights=xs, minlength=n)
            if (b + 1) % log_every == 0 or b + 1 == nb:
                log(f"batch {b + 1}/{nb}")

    def mean(s, c):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (s / c[:, None]).astype(np.float32)

    return {"body": np.asarray(bodies, np.int64),
            "pre_mean": mean(sums["pre"], counts["pre"]), "post_mean": mean(sums["post"], counts["post"]),
            "all_mean": mean(sums["pre"] + sums["post"], counts["pre"] + counts["post"]),
            "n_pre": counts["pre"], "n_post": counts["post"]}


def main():
    from flybrain.io import load_neurons
    bodies = load_neurons()["body"].to_numpy()
    c = compute_centroids(bodies, log=lambda s: print(s, flush=True))
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.CACHE_DIR / "centroids.npz", **c)
    have = int(np.isfinite(c["all_mean"][:, 0]).sum())
    print(f"centroids for {have}/{len(bodies)} neurons → {config.CACHE_DIR / 'centroids.npz'}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
