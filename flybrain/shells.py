"""Brain and VNC surface meshes (nm) from the MaleCNS bucket, plus a tiny binary mesh format for the web."""
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import trimesh

from flybrain import config

_BASE = "https://storage.googleapis.com/flyem-male-cns/rois/pointcloud-shells/"
SHELL_URLS = {"brain": _BASE + "JRCFIB2022M_brain.ply", "vnc": _BASE + "JRCFIB2022M_vnc.ply"}


def fetch_shell(name, cache_dir: Path = config.CACHE_DIR / "shells") -> Path:
    url = SHELL_URLS[name]
    path = Path(cache_dir) / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.rename(path)
    return path


def load_mesh(path):
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)


def write_mesh_bin(path, verts, faces):
    verts, faces = np.asarray(verts), np.asarray(faces)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must have shape (V, 3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if faces.size and faces.min() < 0:
        raise ValueError("faces contain negative vertex indices")
    with open(path, "wb") as fh:
        np.array([len(verts)], "<u4").tofile(fh)
        np.asarray(verts, "<f4").tofile(fh)
        np.array([len(faces)], "<u4").tofile(fh)
        np.asarray(faces, "<u4").tofile(fh)


def read_mesh_bin(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size < 8:
        raise ValueError(f"{path}: expected at least 8 bytes, got {buf.size}")
    nv = int(np.frombuffer(buf[:4], "<u4")[0])
    v_end = 4 + nv * 12
    if buf.size < v_end + 4:
        raise ValueError(f"{path}: expected at least {v_end + 4} bytes for {nv} vertices, got {buf.size}")
    verts = np.frombuffer(buf[4:v_end], "<f4").reshape(nv, 3)
    nf = int(np.frombuffer(buf[v_end:v_end + 4], "<u4")[0])
    if buf.size != 8 + 12 * nv + 12 * nf:
        raise ValueError(f"{path}: expected {8 + 12 * nv + 12 * nf} bytes for {nv} vertices "
                         f"and {nf} faces, got {buf.size}")
    faces = np.frombuffer(buf[v_end + 4:], "<u4").reshape(nf, 3)
    if nf and int(faces.max()) >= nv:
        raise ValueError(f"{path}: face index {int(faces.max())} out of range for {nv} vertices")
    return verts.astype(np.float32), faces.astype(np.uint32)
