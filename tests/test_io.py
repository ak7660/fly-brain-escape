import numpy as np

from flybrain.io import load_neurons


def test_load_neurons_filters_signs_and_sides(synthetic_data_dir):
    df = load_neurons(synthetic_data_dir)
    assert df["body"].tolist() == sorted(df["body"].tolist())
    assert 601 not in df["body"].values and 701 not in df["body"].values
    assert len(df) == 14
    row = df.set_index("body")
    assert row.loc[101, "sign"] == 1 and row.loc[202, "sign"] == -1 and row.loc[203, "sign"] == -1
    assert row.loc[104, "sign"] == 0  # DA → 0
    assert row.loc[104, "type"] == "104"  # missing type falls back to body id
    assert row.loc[101, "side"] == 1 and row.loc[102, "side"] == 2  # from instance suffix
    assert row.loc[201, "side"] == 1  # from somaSide
    assert row.loc[203, "side"] == 0
    assert df["sign"].dtype == np.int8 and df["side"].dtype == np.uint8


from flybrain.io import load_edges


def test_load_edges_threshold_and_input_totals(synthetic_data_dir):
    bodies = load_neurons(synthetic_data_dir)["body"].to_numpy()
    e = load_edges(bodies, synthetic_data_dir, min_weight=5)
    pairs = {(int(bodies[p]), int(bodies[q]), int(w)) for p, q, w in zip(e["pre"], e["post"], e["weight"])}
    assert (201, 302, 3) not in pairs          # below threshold
    assert all(601 not in (a, b) for a, b, _ in pairs)  # non-neuron endpoint dropped
    assert (101, 301, 20) in pairs and len(pairs) == 13
    tot = dict(zip(bodies.tolist(), e["in_total"].tolist()))
    assert tot[301] == 12 + 20 + 4 + 6        # includes non-neuron 601 and weight-agnostic
    assert tot[302] == 5 + 3 + 5
    assert tot[101] == 0
    assert e["pre"].dtype == np.int32 and e["weight"].dtype == np.int32
