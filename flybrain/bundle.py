"""flybrain-bundle v1: little-endian field-concatenated .bin files described by manifest.json (read by web/data.js)."""
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain import config

DTYPES = {"u8": "<u1", "i8": "<i1", "u16": "<u2", "i16": "<i2", "u32": "<u4", "f32": "<f4"}


def _checked(name, code, array):
    """Cast to the field's little-endian dtype, refusing values the cast would wrap, truncate or overflow."""
    src = np.asarray(array)
    dt = np.dtype(DTYPES[code])
    if src.dtype.kind not in "biuf":
        raise ValueError(f"field {name!r}: unsupported source dtype {src.dtype}")
    if dt.kind == "f":
        x = src.astype(np.float64)
        if not np.isfinite(x).all() or (x.size and np.abs(x).max() > np.finfo(np.float32).max):
            raise ValueError(f"field {name!r}: f32 values must be finite and within float32 range")
    else:
        info = np.iinfo(dt)
        if src.dtype.kind == "f":
            if not np.isfinite(src).all() or not np.array_equal(src, np.round(src)):
                raise ValueError(f"field {name!r}: {code} values must be integral")
        if src.size and (src.min() < info.min or src.max() > info.max):
            raise ValueError(f"field {name!r}: {code} values must lie in [{info.min}, {info.max}]")
    return np.ascontiguousarray(src, dtype=dt).ravel()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def write_bin(path, fields):
    """Validate every field (range, alignment) first, so a rejected field never leaves a partial file."""
    layout, arrays, offset = [], [], 0
    for name, code, array in fields:
        a = _checked(name, code, array)
        size = a.dtype.itemsize
        if offset % size:
            raise ValueError(f"field {name!r} at byte {offset} is not aligned to its {size}-byte items")
        arrays.append(a)
        layout.append([name, code, int(a.size)])
        offset += a.nbytes
    with open(path, "wb") as fh:
        for a in arrays:
            fh.write(a.tobytes())
    return layout


def read_bin(path, layout):
    buf = Path(path).read_bytes()
    out, offset = {}, 0
    for name, code, count in layout:
        dt = np.dtype(DTYPES[code])
        out[name] = np.frombuffer(buf, dtype=dt, count=count, offset=offset).copy()
        offset += dt.itemsize * count
    if offset != len(buf):
        raise ValueError(f"{path}: layout covers {offset} bytes but file has {len(buf)}")
    return out


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world_um(xyz_nm, centre_nm):
    m = np.asarray(config.WORLD_AXES, dtype=np.float64)
    return ((np.asarray(xyz_nm, dtype=np.float64) - np.asarray(centre_nm)) @ m.T / 1000.0).astype(np.float32)


def write_bundle(out_dir, files, manifest):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for fname, fields in files.items():
        layout = write_bin(out_dir / fname, fields)
        entries[fname] = {"bytes": (out_dir / fname).stat().st_size, "sha256": sha256_file(out_dir / fname),
                          "layout": layout}
    manifest = {**manifest, "files": entries}
    tmp = out_dir / "manifest.json.tmp"  # written after the bins, then renamed: readers never see a partial manifest
    tmp.write_text(json.dumps(manifest, indent=1, default=_json_default))
    tmp.replace(out_dir / "manifest.json")
    return json.loads((out_dir / "manifest.json").read_text())


def read_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    data = {}
    for fname, entry in manifest["files"].items():
        path = bundle_dir / fname
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{fname}: size/sha256 mismatch with manifest")
        data[fname] = read_bin(path, entry["layout"])
    return manifest, data
