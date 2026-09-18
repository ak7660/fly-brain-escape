"""A 15-body synthetic MaleCNS with the real file names and column names.

Graph (weights), sources LPLC2 101/102 and LC4 103, DNs 301 (DNp01) and 302:
  101→201 10   102→201 6   103→202 8   201→301 12   202→302 5   201→302 3 (below threshold)
  101→301 20   301→401 9   302→402 7   601→301 4 (601 is not a neuron)
  203→301 6 (203 unreachable from sources)
  101→501 5    501→502 5   502→503 5   503→302 5   (4-hop detour, must be excluded)
"""
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flybrain import config

NEURONS = [
    # body, type, instance, superclass, class, subclass, somaSide, status
    (101, "LPLC2", "LPLC2_L", "visual_projection", None, None, None, "Traced"),
    (102, "LPLC2", "LPLC2_R", "visual_projection", None, None, None, "Traced"),
    (103, "LC4", "LC4_L", "visual_projection", None, None, None, "Traced"),
    (201, "H1", "H1_L", "cb_intrinsic", None, None, "L", "Traced"),
    (202, "H2", "H2_R", "cb_intrinsic", None, None, "R", "Traced"),
    (203, "H3", "H3", "cb_intrinsic", None, None, None, "Traced"),
    (104, None, None, "cb_intrinsic", None, None, None, "Traced"),  # no type, no NT
    (301, "DNp01", "DNp01(GF)_R", "descending_neuron", None, "xl", None, "Traced"),
    (302, "DNx", "DNx_L", "descending_neuron", None, "fl", None, "Traced"),
    (401, "TTMn", "TTMn_R", "vnc_motor", None, "ml", None, "Traced"),
    (402, "MNx", "MNx_L", "vnc_motor", None, "fl", None, "Traced"),
    (501, "D1", "D1", "cb_intrinsic", None, None, None, "Traced"),
    (502, "D2", "D2", "cb_intrinsic", None, None, None, "Traced"),
    (503, "D3", "D3", "cb_intrinsic", None, None, None, "Traced"),
    (601, None, None, None, None, None, None, "Orphan"),  # not a neuron
    (701, "X", "X", "cb_intrinsic", None, None, None, "Assign"),  # neuron-like but not traced
]
# consensus_nt uses full lowercase names in the real file (verified 2026-09-17)
NT = {101: "acetylcholine", 102: "acetylcholine", 103: "acetylcholine", 201: "acetylcholine",
      202: "gaba", 203: "glutamate", 104: "dopamine", 301: "acetylcholine", 302: "acetylcholine",
      401: "glutamate", 402: "glutamate", 501: "acetylcholine", 502: "acetylcholine",
      503: "acetylcholine"}
EDGES = [
    (101, 201, 10), (102, 201, 6), (103, 202, 8), (201, 301, 12), (202, 302, 5), (201, 302, 3),
    (101, 301, 20), (301, 401, 9), (302, 402, 7), (601, 301, 4), (203, 301, 6),
    (101, 501, 5), (501, 502, 5), (502, 503, 5), (503, 302, 5),
]
# synapse sites: (x_pre, y_pre, z_pre, body_pre, x_post, y_post, z_post, body_post)
SYNAPSES = [
    (10, 0, 0, 101, 20, 0, 0, 201),
    (30, 0, 0, 101, 40, 0, 0, 201),
    (50, 6, 0, 201, 60, 0, 9, 301),
    (70, 0, 0, 999, 80, 0, 3, 301),  # 999 is not a neuron: pre ignored, post counted
]


def _write(tmp, name, table, chunk=None):
    feather.write_feather(table, tmp / name, compression="uncompressed", chunksize=chunk)


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("malecns")
    cols = list(zip(*NEURONS))
    _write(tmp, config.ANNOTATIONS, pa.table({
        "bodyId": pa.array(cols[0], pa.int64()), "type": cols[1], "instance": cols[2],
        "superclass": cols[3], "class": cols[4], "subclass": cols[5], "somaSide": cols[6],
        "rootSide": [None] * len(NEURONS), "status": cols[7],
    }))
    _write(tmp, config.NEUROTRANSMITTERS, pa.table({
        "body": pa.array(list(NT), pa.int64()), "consensus_nt": list(NT.values()),
    }))
    e = list(zip(*EDGES))
    _write(tmp, config.WEIGHTS_TRACED, pa.table({
        "body_pre": pa.array(e[0], pa.int64()), "body_post": pa.array(e[1], pa.int64()),
        "weight": pa.array(e[2], pa.int64()),
        "type_pre": ["t"] * len(EDGES), "type_post": ["t"] * len(EDGES),
    }), chunk=4)  # several record batches, like the real file
    s = list(zip(*SYNAPSES))
    _write(tmp, config.SYN_PARTNERS_TRACED, pa.table({
        "x_pre": pa.array(s[0], pa.int32()), "y_pre": pa.array(s[1], pa.int32()),
        "z_pre": pa.array(s[2], pa.int32()), "body_pre": pa.array(s[3], pa.int64()),
        "x_post": pa.array(s[4], pa.int32()), "y_post": pa.array(s[5], pa.int32()),
        "z_post": pa.array(s[6], pa.int32()), "body_post": pa.array(s[7], pa.int64()),
    }), chunk=2)
    return tmp
