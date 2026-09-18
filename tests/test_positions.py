import numpy as np
import pytest

from flybrain.io import load_neurons
from flybrain.positions import compute_centroids


def test_centroids_average_pre_and_post_sites(synthetic_data_dir):
    bodies = load_neurons(synthetic_data_dir)["body"].to_numpy()
    lines = []
    c = compute_centroids(bodies, synthetic_data_dir, log=lines.append, log_every=1)
    at = dict(zip(c["body"].tolist(), range(len(c["body"]))))
    assert np.allclose(c["pre_mean"][at[101]], [20, 0, 0]) and c["n_pre"][at[101]] == 2
    assert np.allclose(c["post_mean"][at[201]], [30, 0, 0]) and c["n_post"][at[201]] == 2
    assert np.allclose(c["post_mean"][at[301]], [70, 0, 6]) and c["n_post"][at[301]] == 2
    assert np.allclose(c["all_mean"][at[201]], [(20 + 40 + 50) / 3, 2, 0])  # 2 post + 1 pre site
    assert np.all(np.isnan(c["pre_mean"][at[102]]))
    assert lines and lines[-1].startswith("batch 2/2")


@pytest.mark.parametrize("bodies", [[], [3, 2, 1], [1, 1, 2]])
def test_centroids_require_sorted_unique_bodies(synthetic_data_dir, bodies):
    with pytest.raises(ValueError):
        compute_centroids(np.array(bodies, np.int64), synthetic_data_dir)
