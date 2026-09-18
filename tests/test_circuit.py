import numpy as np

from flybrain.circuit import INF, Circuit, build_circuit, hop_distances
from flybrain.io import load_edges, load_neurons


def _data(d):
    neurons = load_neurons(d)
    return neurons, load_edges(neurons["body"].to_numpy(), d)


def test_hop_distances_forward_and_reverse():
    pre, post = np.array([0, 1, 2]), np.array([1, 2, 3])  # 0→1→2→3
    assert hop_distances(pre, post, 5, [0], 3).tolist() == [0, 1, 2, 3, INF]
    assert hop_distances(pre, post, 5, [3], 2, reverse=True).tolist() == [INF, 2, 1, 0, INF]


def test_build_circuit_selects_path_nodes(synthetic_data_dir):
    c = build_circuit(*_data(synthetic_data_dir))
    assert sorted(c.body.tolist()) == [101, 102, 103, 201, 202, 301, 302, 401, 402]
    by = dict(zip(c.body.tolist(), range(len(c.body))))
    assert c.layer[by[101]] == 0 and c.role[by[101]] == 0
    assert c.layer[by[201]] == 1 and c.layer[by[202]] == 1
    assert c.layer[by[301]] == 3 and c.role[by[302]] == 2
    assert c.layer[by[401]] == 4 and c.role[by[402]] == 3
    edges = {(int(c.body[a]), int(c.body[b])) for a, b in zip(c.pre, c.post)}
    assert edges == {(101, 201), (102, 201), (103, 202), (201, 301), (202, 302),
                     (101, 301), (301, 401), (302, 402)}


def test_w0_is_signed_input_fraction(synthetic_data_dir):
    c = build_circuit(*_data(synthetic_data_dir))
    by = dict(zip(c.body.tolist(), range(len(c.body))))
    w0 = {(int(c.body[a]), int(c.body[b])): float(v) for a, b, v in zip(c.pre, c.post, c.w0)}
    assert np.isclose(w0[(201, 301)], 12 / 42)
    assert np.isclose(w0[(202, 302)], -5 / 13)
    assert c.in_total[by[301]] == 42


def test_node_cap_keeps_sources_and_dns(synthetic_data_dir):
    c = build_circuit(*_data(synthetic_data_dir), node_cap=6)
    kept = set(c.body.tolist())
    assert {101, 102, 103, 301, 302} <= kept
    assert len(kept & {201, 202}) == 1  # one slot left for hidden neurons


def test_edge_cap_prunes_hidden_to_hidden_only(synthetic_data_dir):
    c = build_circuit(*_data(synthetic_data_dir), edge_cap=7)
    # no hidden→hidden edges exist here, so nothing can be pruned: the cap is best-effort
    assert len(c.pre) == 8


def _dn_hops(c):
    return hop_distances(c.pre, c.post, len(c.body), np.flatnonzero(c.role == 0), 3)[c.role == 2]


def test_node_cap_never_strands_a_dn(synthetic_data_dir):
    # room for 0 hidden neurons: DN 302 would lose its only relay (202) without relay protection
    c = build_circuit(*_data(synthetic_data_dir), node_cap=5)
    assert np.all(_dn_hops(c) <= 3)
    assert 202 in c.body.tolist()


def test_edge_cap_never_cuts_a_dn_relay_edge():
    import pandas as pd
    # S→A→B→D (A→B weak) and S→X→Y→E: pruning weak hidden→hidden edges would strand DN D
    names = ["S", "A", "B", "D", "X", "Y", "E"]
    neurons = pd.DataFrame({
        "body": np.arange(1, 8), "type": ["LPLC2", "A", "B", "D", "X", "Y", "E"],
        "instance": names, "superclass": ["visual_projection", "h", "h", "descending_neuron", "h", "h",
                                          "descending_neuron"],
        "side": np.zeros(7, np.uint8), "sign": np.ones(7, np.int8)})
    at = {k: i for i, k in enumerate(names)}
    pairs = [("S", "A", 50), ("A", "B", 6), ("B", "D", 50), ("S", "X", 50), ("X", "Y", 40), ("Y", "E", 50)]
    edges = {"pre": np.array([at[a] for a, _, _ in pairs], np.int32),
             "post": np.array([at[b] for _, b, _ in pairs], np.int32),
             "weight": np.array([w for _, _, w in pairs], np.int32), "in_total": np.full(7, 100.0)}
    c = build_circuit(neurons, edges, edge_cap=5)
    assert np.all(_dn_hops(c) <= 3)


def test_select_dns_top_flow_with_reserved_giant_fiber():
    from flybrain.circuit import select_dns
    f = np.array([5.0, 9.0, 1.0, 7.0])
    cand = np.array([True, True, True, False])  # DN 3 is not reachable
    none = np.zeros(4, bool)
    assert select_dns(f, cand, none, 1).tolist() == [False, True, False, False]
    assert select_dns(f, cand, none, 2).tolist() == [True, True, False, False]
    gf = np.array([False, False, True, False])
    assert select_dns(f, cand, gf, 1).tolist() == [False, False, True, False]  # GF takes the slot
    assert select_dns(f, cand, gf, 2).tolist() == [False, True, True, False]


def test_forward_flow_follows_output_shares():
    from flybrain.circuit import forward_flow
    pre, post, w = np.array([0, 0, 1]), np.array([1, 2, 3]), np.array([30, 10, 5])
    f = forward_flow(pre, post, w, 4, np.array([True, False, False, False]), 3)
    assert np.allclose(f, [0, 0.75, 0.25, 0.75])


def test_dn_cap_keeps_giant_fiber_and_drops_its_exclusive_partners(synthetic_data_dir):
    c = build_circuit(*_data(synthetic_data_dir), dn_cap=1)
    assert sorted(c.body.tolist()) == [101, 102, 103, 201, 301, 401]
    assert c.meta["n_dn_candidates"] == 2 and c.meta["dn_cap"] == 1 and c.meta["dn_cap_hit"]
    assert np.all(_dn_hops(c) <= 3)


def _toy(names, types, superclasses, pairs, signs=None):
    import pandas as pd
    n = len(names)
    neurons = pd.DataFrame({"body": np.arange(1, n + 1), "type": types, "instance": names,
                            "superclass": superclasses, "side": np.zeros(n, np.uint8),
                            "sign": np.ones(n, np.int8) if signs is None else np.array(signs, np.int8)})
    at = {k: i for i, k in enumerate(names)}
    edges = {"pre": np.array([at[a] for a, _, _ in pairs], np.int32),
             "post": np.array([at[b] for _, b, _ in pairs], np.int32),
             "weight": np.array([w for _, _, w in pairs], np.int32), "in_total": np.full(n, 100.0)}
    return neurons, edges


def test_dn_cap_prefers_giant_fiber_over_higher_flow_dn():
    # S→GF weakly, S→H→X strongly: X has more flow but the GF is always kept; H only serves X
    neurons, edges = _toy(["S", "GF", "H", "X"], ["LPLC2", "DNp01", "H", "DNx"],
                          ["visual_projection", "descending_neuron", "h", "descending_neuron"],
                          [("S", "GF", 5), ("S", "H", 50), ("H", "X", 50), ("S", "X", 50)])
    c = build_circuit(neurons, edges, dn_cap=1)
    assert sorted(c.type.tolist()) == ["DNp01", "LPLC2"]
    c2 = build_circuit(neurons, edges, dn_cap=2)
    assert sorted(c2.type.tolist()) == ["DNp01", "DNx", "H", "LPLC2"]


def test_dropped_dn_is_not_kept_as_hidden():
    # S→X(DN)→Y(DN), S→Y weak: Y collects more flow (0.09 + 0.91 vs 0.91), so Y wins the single
    # slot; X lies on the S→X→Y path but must not sneak in as a hidden neuron
    neurons, edges = _toy(["S", "X", "Y"], ["LPLC2", "DNx", "DNy"],
                          ["visual_projection", "descending_neuron", "descending_neuron"],
                          [("S", "X", 50), ("X", "Y", 50), ("S", "Y", 5)])
    c = build_circuit(neurons, edges, dn_cap=1)
    assert sorted(c.type.tolist()) == ["DNy", "LPLC2"] and np.all(c.role != 1)


def _assert_connected(c):
    """Final-edge-set invariants: every non-input neuron is reachable from an input, every hidden
    neuron reaches a DN, every MN receives from a DN, and layers match d_fwd."""
    n = len(c.body)
    d_f = hop_distances(c.pre, c.post, n, np.flatnonzero(c.role == 0), n)
    d_b = hop_distances(c.pre, c.post, n, np.flatnonzero(c.role == 2), n, reverse=True)
    assert np.all(d_f[c.role != 0] < INF)
    assert np.all(d_b[c.role == 1] < INF)
    mn_from_dn = np.zeros(n, bool)
    mn_from_dn[c.post[c.role[c.pre] == 2]] = True
    assert np.all(mn_from_dn[c.role == 3])
    assert np.array_equal(c.d_fwd, d_f)
    hidden = c.role == 1
    assert np.array_equal(c.layer[hidden], np.where(d_f[hidden] == 1, 1, 2))
    assert np.all(c.w0 != 0)


def test_synthetic_circuit_is_connected(synthetic_data_dir):
    for kw in [{}, {"node_cap": 5}, {"node_cap": 6}, {"dn_cap": 1}]:
        _assert_connected(build_circuit(*_data(synthetic_data_dir), **kw))


def test_edge_pruning_removes_neurons_left_without_a_path():
    # S→C→D is D's relay; S→A→B→D is not. Pruning weak A→B leaves B without inputs and A without
    # a route to a DN: both must leave the circuit.
    neurons, edges = _toy(["S", "A", "B", "C", "D"], ["LPLC2", "A", "B", "C", "DNx"],
                          ["visual_projection", "h", "h", "h", "descending_neuron"],
                          [("S", "C", 50), ("C", "D", 60), ("S", "A", 50), ("A", "B", 6), ("B", "D", 50)])
    c = build_circuit(neurons, edges, edge_cap=4)
    assert sorted(c.instance.tolist()) == ["C", "D", "S"]
    assert c.meta["n_removed_unconnected"] == 2
    _assert_connected(c)


def test_motor_neurons_do_not_relay_during_selection():
    neurons, edges = _toy(["S", "M", "D"], ["LPLC2", "MNx", "DNx"],
                          ["visual_projection", "vnc_motor", "descending_neuron"],
                          [("S", "M", 50), ("M", "D", 50)])
    c = build_circuit(neurons, edges)
    assert c.instance.tolist() == ["S"]


def test_relay_paths_reports_targets_without_a_chain():
    from flybrain.circuit import _relay_paths
    pre, post, w = np.array([0, 1]), np.array([1, 2]), np.array([9, 9])
    d_f = np.array([0, 1, 2])
    allowed = np.array([True, False, True])  # node 1 may not relay
    relay, protected, failed = _relay_paths(pre, post, w, d_f, np.array([2]), allowed)
    assert failed.tolist() == [False, False, True] and not protected.any() and not relay.any()


def test_zero_sign_edges_are_dropped_and_counted():
    # Z is a DN with no transmitter sign: its edge Z→D carries nothing and must not enter the circuit
    neurons, edges = _toy(["S", "Z", "D"], ["LPLC2", "DNz", "DNx"],
                          ["visual_projection", "descending_neuron", "descending_neuron"],
                          [("S", "Z", 50), ("S", "D", 50), ("Z", "D", 50)], signs=[1, 0, 1])
    c = build_circuit(neurons, edges)
    pairs = {(c.instance[a], c.instance[b]) for a, b in zip(c.pre, c.post)}
    assert pairs == {("S", "Z"), ("S", "D")}
    assert c.meta["n_zero_sign_edges_dropped"] == 1


def test_dn_cap_default_lives_in_config():
    import inspect
    from flybrain import config
    assert config.DN_CAP == 300
    assert inspect.signature(build_circuit).parameters["dn_cap"].default == config.DN_CAP


def test_save_load_roundtrip(tmp_path, synthetic_data_dir):
    c = build_circuit(*_data(synthetic_data_dir))
    c.save(tmp_path / "c.npz")
    d = Circuit.load(tmp_path / "c.npz")
    assert d.body.tolist() == c.body.tolist() and d.type.tolist() == c.type.tolist()
    assert np.array_equal(d.w0, c.w0) and d.meta == c.meta
