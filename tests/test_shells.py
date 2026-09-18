import numpy as np
import pytest
import trimesh

from flybrain.shells import load_mesh, read_mesh_bin, write_mesh_bin


def test_ply_load_and_bin_roundtrip(tmp_path):
    tet = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          faces=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], process=False)
    ply = tmp_path / "tet.ply"
    tet.export(ply)
    v, f = load_mesh(ply)
    assert v.shape == (4, 3) and f.shape == (4, 3)
    assert v.dtype == np.float32 and f.dtype == np.uint32
    write_mesh_bin(tmp_path / "tet.bin", v, f)
    assert (tmp_path / "tet.bin").stat().st_size == 4 + 4 * 12 + 4 + 4 * 12
    v2, f2 = read_mesh_bin(tmp_path / "tet.bin")
    assert np.array_equal(v, v2) and np.array_equal(f, f2)


def test_read_mesh_bin_rejects_truncated_and_bad_indices(tmp_path):
    v = np.zeros((4, 3), np.float32)
    f = np.array([[0, 1, 2], [0, 1, 3]], np.uint32)
    good = tmp_path / "good.bin"
    write_mesh_bin(good, v, f)
    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(good.read_bytes()[:-5])
    with pytest.raises(ValueError, match="expected .* bytes"):
        read_mesh_bin(truncated)
    bad = tmp_path / "bad.bin"
    write_mesh_bin(bad, v, np.array([[0, 1, 4]], np.uint32))
    with pytest.raises(ValueError, match="face index"):
        read_mesh_bin(bad)


@pytest.mark.parametrize("verts, faces", [
    (np.zeros((4, 2)), [[0, 1, 2]]),
    (np.zeros((4, 3)), [[0, 1, 2, 3]]),
    (np.zeros((4, 3)), [[0, -1, 2]]),
])
def test_write_mesh_bin_rejects_malformed_input(tmp_path, verts, faces):
    with pytest.raises(ValueError):
        write_mesh_bin(tmp_path / "x.bin", verts, np.asarray(faces))
