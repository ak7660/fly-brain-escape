"""Looming-escape circuit: neurons on short LPLC2/LC4 → descending-neuron paths, plus motor neurons."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from flybrain import config

INF = 10**6


def hop_distances(pre, post, n, seeds, max_hops, reverse=False):
    """Unweighted multi-source BFS over edges pre→post (reverse=True walks post→pre)."""
    a = sp.csr_matrix((np.ones(len(pre), np.float32), (pre, post)), shape=(n, n))
    step = a if reverse else a.T.tocsr()
    dist = np.full(n, INF, dtype=np.int64)
    frontier = np.zeros(n, dtype=np.float32)
    dist[seeds] = 0
    frontier[seeds] = 1
    for h in range(1, max_hops + 1):
        new = ((step @ frontier) > 0) & (dist == INF)
        if not new.any():
            break
        dist[new] = h
        frontier = new.astype(np.float32)
    return dist


def _weights(pre, post, w, n):
    return sp.csr_matrix((np.asarray(w, np.float64), (pre, post)), shape=(n, n))


def forward_flow(pre, post, w, n, src_mask, steps):
    """Source activity pushed forward `steps` times, split by each neuron's output shares (summed)."""
    W = _weights(pre, post, w, n)
    P = sp.diags(1 / np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1)) @ W  # share of pre's output
    x, f = src_mask.astype(np.float64), np.zeros(n)
    for _ in range(steps):
        x = P.T @ x
        f += x
    return f


def backward_flow(pre, post, w, n, tgt_mask, steps):
    """Target relevance pulled backward `steps` times, split by each neuron's input shares (summed)."""
    W = _weights(pre, post, w, n)
    Q = W @ sp.diags(1 / np.maximum(np.asarray(W.sum(axis=0)).ravel(), 1))  # share of post's input
    y, b = tgt_mask.astype(np.float64), np.zeros(n)
    for _ in range(steps):
        y = Q @ y
        b += y
    return b


def select_dns(f, candidates, must, cap):
    """`must` DNs (Giant Fiber) take slots first; the rest go to the highest-flow candidates."""
    keep = must & candidates
    rest = np.flatnonzero(candidates & ~keep)
    room = max(cap - int(keep.sum()), 0)
    keep = keep.copy()
    keep[rest[np.argsort(-f[rest], kind="stable")[:room]]] = True
    return keep


def _relay_paths(pre, post, w, d_f, targets, allowed):
    """One shortest source→target chain per target, walked backwards along d_f − 1, heaviest edge
    first (reusing already-chosen chains when possible).
    Returns (relay node mask, protected edge mask, mask of targets with no chain)."""
    n = len(d_f)
    order = np.argsort(post, kind="stable")
    starts = np.searchsorted(post[order], np.arange(n + 1))
    on_path = d_f == 0
    protected = np.zeros(len(pre), dtype=bool)
    failed = np.zeros(n, dtype=bool)
    targets = np.asarray(targets)
    for t in targets[np.argsort(d_f[targets], kind="stable")]:
        chain, v = [], t
        while not on_path[v]:
            ei = order[starts[v]:starts[v + 1]]
            ei = ei[(d_f[pre[ei]] == d_f[v] - 1) & allowed[pre[ei]]]
            if len(ei) == 0:
                chain = None
                break
            done = ei[on_path[pre[ei]]]
            pick = done if len(done) else ei
            e = pick[np.argmax(w[pick])]
            chain.append(e)
            v = pre[e]
        if chain is None:
            failed[t] = True
            continue
        on_path[t] = True
        on_path[pre[chain]] = True
        protected[chain] = True
    targets_mask = np.zeros(n, dtype=bool)
    targets_mask[targets] = True
    return on_path & (d_f > 0) & ~targets_mask, protected, failed


@dataclass
class Circuit:
    body: np.ndarray
    type: np.ndarray
    instance: np.ndarray
    superclass: np.ndarray
    sign: np.ndarray
    side: np.ndarray
    role: np.ndarray
    layer: np.ndarray
    d_fwd: np.ndarray
    in_total: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    w0: np.ndarray
    meta: dict = field(default_factory=dict)

    def save(self, path):
        arrays = {k: v for k, v in asdict(self).items() if k != "meta"}
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files if k != "meta"}
        return cls(meta=json.loads(str(z["meta"])), **kw)


def build_circuit(neurons, edges, source_types=config.SOURCE_TYPES, max_hops=config.MAX_HOPS,
                  node_cap=config.NODE_CAP, mn_cap=config.MN_CAP, edge_cap=config.EDGE_CAP,
                  min_weight=config.MIN_WEIGHT, dn_cap=config.DN_CAP) -> Circuit:
    n = len(neurons)
    ok = edges["weight"] >= min_weight
    pre_all = edges["pre"][ok].astype(np.int64)
    post_all = edges["post"][ok].astype(np.int64)
    w_all = edges["weight"][ok].astype(np.int64)
    types = neurons["type"].to_numpy().astype(str)
    sc = neurons["superclass"].to_numpy().astype(str)
    sign_all = neurons["sign"].to_numpy().astype(np.int8)
    is_src = np.isin(types, source_types)
    is_dn = sc == "descending_neuron"
    is_mn = np.isin(sc, config.MOTOR_SUPERCLASSES)
    is_gf = types == "DNp01"
    signal = sign_all[pre_all] != 0  # edges from sign-0 neurons carry nothing in the model

    # Selection graph: signal-carrying edges, no motor neurons (they are added at the end)
    sel_idx = np.flatnonzero(signal & ~is_mn[pre_all] & ~is_mn[post_all])

    def graph(nodes):
        i = sel_idx[nodes[pre_all[sel_idx]] & nodes[post_all[sel_idx]]]
        return i, pre_all[i], post_all[i], w_all[i]

    # 1) DN layer: the dn_cap reachable DNs with the most forward flow (Giant Fiber reserved)
    gi, pre, post, w = graph(np.ones(n, dtype=bool))
    d_f = hop_distances(pre, post, n, np.flatnonzero(is_src), max_hops)
    dn_cand = is_dn & (d_f <= max_hops)
    dn_flow = forward_flow(pre, post, w, n, is_src, max_hops)
    dn_keep = select_dns(dn_flow, dn_cand, is_gf, dn_cap)
    # 2) dropped DNs leave the graph; a kept DN reachable only through them is dropped too
    while True:
        allowed = ~is_mn & (~is_dn | dn_keep)
        gi, pre, post, w = graph(allowed)
        d_f = hop_distances(pre, post, n, np.flatnonzero(is_src), max_hops)
        lost = dn_keep & (d_f > max_hops)
        if not lost.any():
            break
        dn_keep &= ~lost

    # 3) neurons on a short source → kept-DN path, with one guaranteed relay chain per DN
    d_b = hop_distances(pre, post, n, np.flatnonzero(dn_keep), max_hops, reverse=True)
    keep = is_src | (((d_f + d_b) <= max_hops) & allowed)
    n_path = int(keep.sum())
    relay, prot_sel, no_chain = _relay_paths(pre, post, w, d_f, np.flatnonzero(dn_keep), allowed)
    dn_keep &= ~no_chain
    keep &= ~no_chain
    protected = np.zeros(len(pre_all), dtype=bool)
    protected[gi[prot_sel]] = True

    if n_path > node_cap:
        must = is_src | dn_keep | relay
        score = (forward_flow(pre, post, w, n, is_src, max_hops)
                 * backward_flow(pre, post, w, n, dn_keep, max_hops))
        cand = np.flatnonzero(keep & ~must)
        room = max(node_cap - int(must.sum()), 0)
        keep = must.copy()
        keep[cand[np.argsort(-score[cand], kind="stable")[:room]]] = True

    # 4) motor neurons, from the original (signal-carrying) edges of kept DNs
    from_dn = signal & dn_keep[pre_all] & is_mn[post_all]
    mn_in = np.bincount(post_all[from_dn], weights=w_all[from_dn], minlength=n)
    mn_cand = np.flatnonzero(mn_in > 0)
    keep[mn_cand[np.argsort(-mn_in[mn_cand], kind="stable")[:mn_cap]]] = True
    keep[(types == "TTMn") & (mn_in > 0)] = True

    # 5) induced edges minus sign-0 edges, then hidden→hidden pruning to the edge cap
    e = keep[pre_all] & keep[post_all]
    n_zero_sign = int((e & ~signal).sum())
    e &= signal
    epre, epost, ew, eprot = pre_all[e], post_all[e], w_all[e], protected[e]
    n_edges_induced = len(ew)
    hidden = ~is_src & ~is_dn & ~is_mn
    thr = min_weight
    while len(ew) > edge_cap and thr < ew.max():
        thr += 1
        drop = hidden[epre] & hidden[epost] & (ew < thr) & ~eprot
        epre, epost, ew, eprot = epre[~drop], epost[~drop], ew[~drop], eprot[~drop]

    # 6) remove hidden neurons cut off from the inputs or from every DN, and MNs without DN input
    n_removed = 0
    while True:
        live_e = keep[epre] & keep[epost]
        epre, epost, ew = epre[live_e], epost[live_e], ew[live_e]
        d_fin = hop_distances(epre, epost, n, np.flatnonzero(is_src & keep), n)
        d_bin = hop_distances(epre, epost, n, np.flatnonzero(dn_keep & keep), n, reverse=True)
        mn_fed = np.zeros(n, dtype=bool)
        mn_fed[epost[dn_keep[epre]]] = True
        bad = keep & ((hidden & ((d_fin >= INF) | (d_bin >= INF))) | (is_mn & ~mn_fed))
        if not bad.any():
            break
        keep &= ~bad
        n_removed += int(bad.sum())

    idx = np.flatnonzero(keep)
    local = np.full(n, -1, dtype=np.int64)
    local[idx] = np.arange(len(idx))
    cpre, cpost, cw = local[epre], local[epost], ew
    d_fwd = d_fin[idx]
    role = np.select([is_src[idx], is_dn[idx], is_mn[idx]], [0, 2, 3], default=1).astype(np.uint8)
    layer = np.select([role == 0, role == 2, role == 3, d_fwd <= 1], [0, 3, 4, 1],
                      default=2).astype(np.uint8)

    sign = sign_all[idx]
    in_total = edges["in_total"][idx].astype(np.float32)
    w0 = (sign[cpre] * cw / np.maximum(in_total[cpost], 1)).astype(np.float32)
    col = lambda name: neurons[name].fillna("").to_numpy()[idx].astype(str)
    return Circuit(
        body=neurons["body"].to_numpy()[idx].astype(np.int64), type=types[idx], instance=col("instance"),
        superclass=sc[idx], sign=sign, side=neurons["side"].to_numpy()[idx].astype(np.uint8),
        role=role, layer=layer, d_fwd=d_fwd, in_total=in_total,
        pre=cpre.astype(np.int32), post=cpost.astype(np.int32), weight=cw.astype(np.int32), w0=w0,
        meta={"source_types": list(source_types), "max_hops": max_hops, "min_weight": min_weight,
              "node_cap": node_cap, "mn_cap": mn_cap, "edge_cap": edge_cap, "dn_cap": dn_cap,
              "n_dn_candidates": int(dn_cand.sum()), "dn_cap_hit": int(dn_cand.sum()) > dn_cap,
              "n_dn_kept": int(dn_keep.sum()), "n_dn_no_chain": int(no_chain.sum()),
              "n_path_nodes": n_path, "node_cap_hit": n_path > node_cap, "n_relay_nodes": int(relay.sum()),
              "n_zero_sign_edges_dropped": n_zero_sign, "n_edges_induced": int(n_edges_induced),
              "edge_cap_hit": n_edges_induced > edge_cap, "hidden_hidden_min_weight": int(thr),
              "n_removed_unconnected": n_removed},
    )


def main():
    import time
    from flybrain.io import load_edges, load_neurons
    t = time.time()
    neurons = load_neurons()
    edges = load_edges(neurons["body"].to_numpy())
    print(f"loaded {len(neurons)} neurons, {len(edges['pre'])} edges (w>={config.MIN_WEIGHT}) in {time.time()-t:.0f}s")
    c = build_circuit(neurons, edges)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CACHE_DIR / "circuit_v1.npz"
    c.save(out)
    names = ["input", "hidden d=1", "hidden d>=2", "DN", "MN"]
    print(f"circuit: {len(c.body)} neurons, {len(c.pre)} edges → {out}")
    for i, name in enumerate(names):
        print(f"  layer {i} {name:12s} {int((c.layer == i).sum())}")
    print(f"  signs +{int((c.sign > 0).sum())} -{int((c.sign < 0).sum())} 0:{int((c.sign == 0).sum())}; meta {c.meta}")


if __name__ == "__main__":
    main()
