import numpy as np
import pytest

from flybrain import config
from flybrain.circuit import Circuit, INF, hop_distances
from flybrain.io import load_neurons

pytestmark = pytest.mark.slow
CACHE = config.CACHE_DIR / "circuit_v1.npz"


@pytest.fixture(scope="module")
def neurons():
    return load_neurons()


@pytest.fixture(scope="module")
def circuit():
    if not CACHE.exists():
        pytest.skip("run `uv run python -m flybrain.circuit` first")
    return Circuit.load(CACHE)


def test_known_cell_counts(neurons):
    t = neurons["type"]
    assert (t == "LPLC2").sum() == 185
    assert (t == "LC4").sum() == 126
    assert (t == "DNp01").sum() == 2


def test_lplc2_to_giant_fiber_synapses(neurons):
    import pyarrow.compute as pc
    import pyarrow.feather as feather
    tbl = feather.read_table(config.DATA_DIR / config.WEIGHTS_TRACED,
                             columns=["type_pre", "type_post", "weight"])
    m = pc.and_(pc.equal(tbl["type_pre"], "LPLC2"), pc.equal(tbl["type_post"], "DNp01"))
    # 4,862 in all three local weights files (traced-only, significant-only, full) and in the
    # traced-only syn-partners rows; the earlier note of 4,836 did not reproduce.
    assert pc.sum(pc.filter(tbl["weight"], m)).as_py() == 4862


def test_circuit_shape(circuit):
    assert set(circuit.type[circuit.role == 0]) == {"LPLC2", "LC4"}
    assert (circuit.role == 0).sum() == 185 + 126
    assert (circuit.type == "DNp01").sum() == 2
    m = circuit.meta
    assert (m["node_cap"], m["mn_cap"], m["edge_cap"], m["dn_cap"], m["min_weight"], m["max_hops"]) == (
        config.NODE_CAP, config.MN_CAP, config.EDGE_CAP, config.DN_CAP, config.MIN_WEIGHT, config.MAX_HOPS)
    # all three caps bind on the real data; the reachability pass can only shrink the node count
    assert m["node_cap_hit"] and m["edge_cap_hit"] and m["dn_cap_hit"]
    assert (circuit.role <= 2).sum() <= config.NODE_CAP
    assert len(circuit.pre) <= config.EDGE_CAP and m["hidden_hidden_min_weight"] > config.MIN_WEIGHT
    # MN_CAP by DN input, plus any TTMn (jump muscle) forced in beyond it
    assert (circuit.role == 3).sum() <= config.MN_CAP + (circuit.type == "TTMn").sum()
    assert np.all(np.isfinite(circuit.w0)) and np.all(circuit.w0 != 0)
    assert np.all(np.sign(circuit.w0) == circuit.sign[circuit.pre])


def test_dn_layer_is_capped_and_keeps_giant_fiber(circuit):
    names = ["input", "hidden d=1", "hidden d>=2", "DN", "MN"]
    print("layers", {k: int((circuit.layer == i).sum()) for i, k in enumerate(names)},
          "edges", len(circuit.pre), "meta", circuit.meta)
    assert (circuit.role == 2).sum() == config.DN_CAP == circuit.meta["n_dn_kept"]
    assert circuit.meta["n_dn_candidates"] > config.DN_CAP
    assert (circuit.type[circuit.role == 2] == "DNp01").sum() == 2


def test_final_edges_connect_every_neuron(circuit):
    c = circuit
    n = len(c.body)
    d_f = hop_distances(c.pre, c.post, n, np.flatnonzero(c.role == 0), n)
    d_b = hop_distances(c.pre, c.post, n, np.flatnonzero(c.role == 2), n, reverse=True)
    assert np.all(d_f[c.role != 0] < INF)
    assert np.all(d_b[c.role == 1] < INF)
    assert set(np.flatnonzero(c.role == 3)) <= set(c.post[c.role[c.pre] == 2].tolist())
    assert np.array_equal(c.d_fwd, d_f)
    hidden = c.role == 1
    assert np.array_equal(c.layer[hidden], np.where(d_f[hidden] == 1, 1, 2))


def test_shuffled_control_preserves_layer_blocks(circuit):
    import time
    from flybrain.shuffle import shuffled_circuit
    t = time.time()
    s = shuffled_circuit(circuit, seed=0)
    print(f"shuffle took {time.time() - t:.1f}s; rewired {np.mean(s.post != circuit.post):.2f}")
    blocks = lambda c: np.bincount(circuit.layer[c.pre] * 5 + circuit.layer[c.post], minlength=25)
    assert np.array_equal(blocks(s), blocks(circuit))
    assert np.mean(s.post != circuit.post) > 0.5


def test_every_dn_reachable_within_max_hops(circuit):
    n = len(circuit.body)
    d = hop_distances(circuit.pre, circuit.post, n, np.flatnonzero(circuit.role == 0), config.MAX_HOPS)
    assert np.all(d[circuit.role == 2] <= config.MAX_HOPS)
