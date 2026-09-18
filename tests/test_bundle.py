import json

import numpy as np
import pytest

from flybrain import config
from flybrain.bundle import read_bin, read_bundle, world_um, write_bin, write_bundle


def test_world_transform_is_a_rotation_with_expected_axes():
    m = np.array(config.WORLD_AXES)
    assert round(np.linalg.det(m)) == 1
    p = world_um(np.array([[1000.0, 2000.0, 3000.0]]), np.zeros(3))
    assert np.allclose(p, [[-1.0, -3.0, -2.0]]) and p.dtype == np.float32


def test_bin_roundtrip_and_alignment(tmp_path):
    fields = [("row_ptr", "u32", np.arange(5)), ("val", "f32", np.linspace(0, 1, 7)), ("flag", "u8", [1, 0, 1, 1])]
    layout = write_bin(tmp_path / "a.bin", fields)
    assert layout == [["row_ptr", "u32", 5], ["val", "f32", 7], ["flag", "u8", 4]]
    got = read_bin(tmp_path / "a.bin", layout)
    assert got["row_ptr"].dtype == np.uint32 and np.allclose(got["val"], np.linspace(0, 1, 7))
    with pytest.raises(ValueError, match="aligned"):
        write_bin(tmp_path / "b.bin", [("flag", "u8", [1, 2, 3]), ("val", "f32", [1.0])])


def test_bundle_roundtrip_and_tamper_detection(tmp_path):
    files = {"x.bin": [("a", "f32", np.ones(3)), ("b", "u32", [7, 8])], "y.bin": [("c", "i16", [-1, 2])]}
    manifest = write_bundle(tmp_path / "bundle", files, {"format": "flybrain-bundle", "version": 1})
    assert json.loads((tmp_path / "bundle" / "manifest.json").read_text())["files"]["x.bin"]["bytes"] == 20
    m2, data = read_bundle(tmp_path / "bundle")
    assert m2 == manifest and data["y.bin"]["c"].tolist() == [-1, 2]
    (tmp_path / "bundle" / "y.bin").write_bytes(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="sha256"):
        read_bundle(tmp_path / "bundle")


@pytest.mark.parametrize("code,value", [
    ("u32", [-1]), ("u32", [1.7]), ("i16", [40000]), ("u8", [256]), ("i8", [-129]), ("u16", [np.nan]),
    ("f32", [np.nan]), ("f32", [np.inf]), ("f32", [1e40]), ("f32", [-1e40]),
])
def test_write_bin_rejects_values_that_would_wrap_or_truncate(tmp_path, code, value):
    with pytest.raises(ValueError, match="bad_field"):
        write_bin(tmp_path / "c.bin", [("ok", "u8", [1, 2, 3, 4]), ("bad_field", code, value)])
    assert not (tmp_path / "c.bin").exists()


def test_write_bin_accepts_integral_floats_and_range_edges(tmp_path):
    layout = write_bin(tmp_path / "d.bin", [("a", "u32", np.array([0.0, 4294967295.0])), ("b", "i16", [-32768, 32767]),
                                            ("c", "f32", [3.4e38, -3.4e38])])
    got = read_bin(tmp_path / "d.bin", layout)
    assert got["a"].tolist() == [0, 4294967295] and got["b"].tolist() == [-32768, 32767]


def test_manifest_accepts_numpy_values(tmp_path):
    manifest = write_bundle(tmp_path / "bundle", {"x.bin": [("a", "u8", [1])]},
                            {"n": np.int64(3), "r_max": np.float32(4.0), "centre": np.array([1.5, 2.0])})
    assert manifest["n"] == 3 and manifest["r_max"] == 4.0 and manifest["centre"] == [1.5, 2.0]
    assert not list((tmp_path / "bundle").glob("*.tmp"))
