"""Shuffled-wiring control: directed double-edge swaps preserving in/out degree, per-neuron output
weights and the (pre-layer, post-layer) block structure."""
from dataclasses import replace

import numpy as np


def shuffle_edges(pre, post, weight, swaps_per_edge=10, seed=0, layer=None):
    """Swap targets of random edge pairs (a→b, c→d) → (a→d, c→b), rejecting self-loops, duplicates
    and (when `layer` is given) swaps between targets in different layers."""
    rng = np.random.default_rng(seed)
    pre_l, post_l = pre.tolist(), post.tolist()
    lay = None if layer is None else np.asarray(layer).tolist()
    m = len(pre_l)
    existing = set(zip(pre_l, post_l))
    for i, j in rng.integers(0, m, size=(swaps_per_edge * m, 2)).tolist():
        a, b, c, d = pre_l[i], post_l[i], pre_l[j], post_l[j]
        if i == j or a == d or c == b or (lay is not None and lay[b] != lay[d]):
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing -= {(a, b), (c, d)}
        existing |= {(a, d), (c, b)}
        post_l[i], post_l[j] = d, b
    return (np.array(pre_l, dtype=pre.dtype), np.array(post_l, dtype=post.dtype), weight.copy())


def shuffled_circuit(c, seed=0, swaps_per_edge=10):
    """Layer-preserving shuffle of `c`. Each neuron's out-of-circuit input stays constant:
    in_total' = in_total − real in-circuit input + shuffled in-circuit input."""
    pre, post, w = shuffle_edges(c.pre, c.post, c.weight, swaps_per_edge=swaps_per_edge, seed=seed,
                                 layer=c.layer)
    n = len(c.body)
    real_in = np.bincount(c.post, weights=c.weight, minlength=n)
    shuf_in = np.bincount(post, weights=w, minlength=n)
    in_total = (c.in_total.astype(np.float64) - real_in + shuf_in).astype(np.float32)
    w0 = (c.sign[pre] * w / np.maximum(in_total[post], 1)).astype(np.float32)
    return replace(c, pre=pre, post=post, weight=w, w0=w0, in_total=in_total,
                   meta={**c.meta, "shuffled_seed": seed})
