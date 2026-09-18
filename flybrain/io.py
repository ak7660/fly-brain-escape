"""Streaming readers for the MaleCNS feathers (column projection only; see CLAUDE.md)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from flybrain import config


def _side(instance, soma_side, root_side) -> int:
    if isinstance(instance, str):
        if instance.endswith("_L"):
            return 1
        if instance.endswith("_R"):
            return 2
    for s in (soma_side, root_side):
        if s in ("L", "R"):
            return 1 if s == "L" else 2
    return 0


def load_neurons(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    ann = feather.read_table(
        Path(data_dir) / config.ANNOTATIONS,
        columns=["bodyId", "type", "instance", "superclass", "class", "subclass",
                 "somaSide", "rootSide", "status"],
    ).to_pandas()
    ann = ann[ann["superclass"].notna() & (ann["status"] == "Traced")]
    nt = feather.read_table(Path(data_dir) / config.NEUROTRANSMITTERS,
                            columns=["body", "consensus_nt"]).to_pandas().drop_duplicates("body")
    df = ann.rename(columns={"bodyId": "body", "class": "cls"}).merge(
        nt.rename(columns={"consensus_nt": "nt"}), on="body", how="left")
    df["type"] = df["type"].fillna(df["instance"]).fillna(df["body"].astype(str))
    df["side"] = np.array([_side(i, s, r) for i, s, r in
                           zip(df["instance"], df["somaSide"], df["rootSide"])], dtype=np.uint8)
    df["sign"] = df["nt"].map(config.NT_SIGN).fillna(0).astype(np.int8)
    df = df.sort_values("body", kind="stable").reset_index(drop=True)
    return df[["body", "type", "instance", "superclass", "cls", "subclass", "side", "nt", "sign"]]


def _lookup(bodies: np.ndarray, ids: np.ndarray):
    """Index of each id in sorted `bodies`, plus a mask of ids that are present."""
    i = np.searchsorted(bodies, ids)
    i = np.minimum(i, len(bodies) - 1)
    return i, bodies[i] == ids


def load_edges(bodies: np.ndarray, data_dir: Path = config.DATA_DIR,
               min_weight: int = config.MIN_WEIGHT) -> dict:
    """Stream the traced-only weights: neuron→neuron edges with weight >= min_weight, plus
    `in_total` = all traced input synapses of each body (any weight, any presynaptic body)."""
    n = len(bodies)
    in_total = np.zeros(n, dtype=np.float64)
    pres, posts, ws = [], [], []
    with pa.memory_map(str(Path(data_dir) / config.WEIGHTS_TRACED)) as src:
        reader = ipc.open_file(src)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            pre = batch.column("body_pre").to_numpy()
            post = batch.column("body_post").to_numpy()
            w = batch.column("weight").to_numpy()
            qi, qok = _lookup(bodies, post)
            in_total += np.bincount(qi[qok], weights=w[qok], minlength=n)
            strong = np.flatnonzero(qok & (w >= min_weight))  # threshold before the pre lookup
            pi, pok = _lookup(bodies, pre[strong])
            keep = strong[pok]
            pres.append(pi[pok].astype(np.int32))
            posts.append(qi[keep].astype(np.int32))
            ws.append(w[keep].astype(np.int32))
    return {"pre": np.concatenate(pres), "post": np.concatenate(posts),
            "weight": np.concatenate(ws), "in_total": in_total}
