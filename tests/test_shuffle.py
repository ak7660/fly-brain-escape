import numpy as np

from flybrain.circuit import build_circuit
from flybrain.io import load_edges, load_neurons
from flybrain.shuffle import shuffle_edges, shuffled_circuit


def _random_graph(n=200, m=2000, seed=1):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < m:
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs.add((int(a), int(b)))
    pre, post = map(np.array, zip(*sorted(pairs)))
    return pre, post, rng.integers(5, 50, m)


def test_shuffle_preserves_degrees_weights_and_simplicity():
    pre, post, w = _random_graph()
    sp_, sq, sw = shuffle_edges(pre, post, w, seed=3)
    n = 200
    assert np.array_equal(np.bincount(sp_, minlength=n), np.bincount(pre, minlength=n))
    assert np.array_equal(np.bincount(sq, minlength=n), np.bincount(post, minlength=n))
    assert np.array_equal(np.bincount(sp_, weights=sw, minlength=n), np.bincount(pre, weights=w, minlength=n))
    assert not np.any(sp_ == sq)
    assert len(set(zip(sp_.tolist(), sq.tolist()))) == len(pre)
    assert np.array_equal(sp_, pre)  # presynaptic side never moves
    assert np.mean(sq != post) > 0.5  # most targets were rewired


def test_shuffle_is_deterministic():
    pre, post, w = _random_graph()
    a = shuffle_edges(pre, post, w, seed=7)
    b = shuffle_edges(pre, post, w, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def _blocks(pre, post, layer):
    return np.bincount(layer[pre] * 5 + layer[post], minlength=25)


def test_shuffle_preserves_layer_blocks():
    pre, post, w = _random_graph()
    rng = np.random.default_rng(0)
    layer = rng.integers(0, 5, 200)
    # no edges into layer 0 in the original graph
    ok = layer[post] != 0
    pre, post, w = pre[ok], post[ok], w[ok]
    sp_, sq, _ = shuffle_edges(pre, post, w, seed=3, layer=layer)
    assert np.array_equal(_blocks(sp_, sq, layer), _blocks(pre, post, layer))
    assert not np.any(layer[sq] == 0)
    assert np.mean(sq != post) > 0.5
    assert len(set(zip(sp_.tolist(), sq.tolist()))) == len(pre) and not np.any(sp_ == sq)


def test_shuffled_circuit_keeps_out_of_circuit_input(synthetic_data_dir):
    neurons = load_neurons(synthetic_data_dir)
    c = build_circuit(neurons, load_edges(neurons["body"].to_numpy(), synthetic_data_dir))
    s = shuffled_circuit(c, seed=0)
    n = len(c.body)
    expect = (c.in_total - np.bincount(c.post, weights=c.weight, minlength=n)
              + np.bincount(s.post, weights=s.weight, minlength=n))
    assert np.allclose(s.in_total, expect)
    assert np.allclose(s.w0, c.sign[s.pre] * s.weight / np.maximum(s.in_total[s.post], 1))
    assert np.array_equal(_blocks(s.pre, s.post, c.layer), _blocks(c.pre, c.post, c.layer))
    same = shuffled_circuit(c, seed=0, swaps_per_edge=0)
    assert np.array_equal(same.w0, c.w0) and np.array_equal(same.in_total, c.in_total)


def test_shuffled_circuit_recomputes_w0(synthetic_data_dir):
    neurons = load_neurons(synthetic_data_dir)
    c = build_circuit(neurons, load_edges(neurons["body"].to_numpy(), synthetic_data_dir))
    s = shuffled_circuit(c, seed=0)
    expect = c.sign[s.pre] * s.weight / np.maximum(s.in_total[s.post], 1)  # adjusted totals
    assert np.allclose(s.w0, expect) and s.meta["shuffled_seed"] == 0
    assert np.array_equal(s.body, c.body) and np.array_equal(s.layer, c.layer)
