"""MagicaVoxel (.vox) loading utilities for the BTP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import struct

import numpy as np

from magicavoxel_merge.vox import load_vox as _load_vox


@dataclass(slots=True)
class VoxModel:
    name: str
    size: tuple[int, int, int]
    voxels: np.ndarray  # bool/int occupancy grid with palette indices
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(slots=True)
class VoxScene:
    models: List[VoxModel]
    palette_rgba: np.ndarray  # shape (256, 4), uint8
    nodes: list["VoxNode"]
    root_node_ids: list[int]

    @property
    def model_names(self) -> Sequence[str]:
        return [m.name for m in self.models]


def load_scene(path: str | Path) -> VoxScene:
    """Load a MagicaVoxel scene and return a simplified structure."""

    path = Path(path)
    vox = _load_vox(str(path))
    models: List[VoxModel] = []

    nodes, root_node_ids, sg_world_translations, sg_world_rotations = _parse_scenegraph(path)

    for idx, model in enumerate(vox.models):
        name = vox.model_names[idx] if idx < len(vox.model_names) else f"model_{idx}"
        # Prefer scene-graph accumulated transforms when available.
        # Fall back to upstream loader's best-effort per-model translations.
        translation = (0.0, 0.0, 0.0)
        if idx < len(sg_world_translations):
            translation = sg_world_translations[idx]
        elif getattr(vox, "model_translations", None) and idx < len(vox.model_translations):
            t = vox.model_translations[idx]
            translation = (float(t[0]), float(t[1]), float(t[2]))

        rotation = (0.0, 0.0, 0.0, 1.0)
        if idx < len(sg_world_rotations):
            rotation = sg_world_rotations[idx]

        voxels = np.asarray(model.voxels, dtype=np.int32)
        models.append(
            VoxModel(
                name=name,
                size=tuple(int(s) for s in model.size),
                voxels=voxels,
                translation=translation,
                rotation=rotation,
            )
        )

    palette = np.asarray(vox.palette_rgba, dtype=np.uint8)
    return VoxScene(models=models, palette_rgba=palette, nodes=nodes, root_node_ids=root_node_ids)


@dataclass(slots=True)
class VoxNode:
    kind: str  # trn, grp, shp
    name: str
    # Local transform (only meaningful for trn nodes)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    children: list[int] = None  # type: ignore[assignment]
    model_ids: list[int] = None  # only for shp

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []
        if self.model_ids is None:
            self.model_ids = []


def _decode_rotation_byte(r: int) -> np.ndarray:
    r = int(r) & 0xFF
    i0 = r & 0x3
    i1 = (r >> 2) & 0x3
    if i0 == i1 or i0 > 2 or i1 > 2:
        return np.eye(3, dtype=np.float32)
    i2 = 3 - i0 - i1
    if i2 < 0 or i2 > 2:
        return np.eye(3, dtype=np.float32)

    s0 = -1.0 if ((r >> 4) & 0x1) else 1.0
    s1 = -1.0 if ((r >> 5) & 0x1) else 1.0
    s2 = -1.0 if ((r >> 6) & 0x1) else 1.0

    m = np.zeros((3, 3), dtype=np.float32)
    m[0, int(i0)] = float(s0)
    m[1, int(i1)] = float(s1)
    m[2, int(i2)] = float(s2)
    return m


def _quat_from_mat3(m: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(m, dtype=np.float32)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = (tr + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = float(m[2, 1] - m[1, 2]) / s
        y = float(m[0, 2] - m[2, 0]) / s
        z = float(m[1, 0] - m[0, 1]) / s
    elif float(m[0, 0]) > float(m[1, 1]) and float(m[0, 0]) > float(m[2, 2]):
        s = (1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2])) ** 0.5 * 2.0
        w = float(m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = float(m[0, 1] + m[1, 0]) / s
        z = float(m[0, 2] + m[2, 0]) / s
    elif float(m[1, 1]) > float(m[2, 2]):
        s = (1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2])) ** 0.5 * 2.0
        w = float(m[0, 2] - m[2, 0]) / s
        x = float(m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = float(m[1, 2] + m[2, 1]) / s
    else:
        s = (1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1])) ** 0.5 * 2.0
        w = float(m[1, 0] - m[0, 1]) / s
        x = float(m[0, 2] + m[2, 0]) / s
        y = float(m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (float(x / n), float(y / n), float(z / n), float(w / n))


def _quat_mul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Quaternion multiplication (NOT normalized).

    Normalizing here breaks vector rotation (q*(v,0)*q^-1) by destroying magnitude.
    """

    ax, ay, az, aw = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx, by, bz, bw = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return (float(x), float(y), float(z), float(w))


def _quat_norm(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (float(x / n), float(y / n), float(z / n), float(w / n))


def _quat_conj(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-float(q[0]), -float(q[1]), -float(q[2]), float(q[3]))


def _quat_rotate_vec(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    # v' = q * (v,0) * conj(q)
    vx, vy, vz = (float(v[0]), float(v[1]), float(v[2]))
    qv = (vx, vy, vz, 0.0)
    t = _quat_mul(_quat_mul(q, qv), _quat_conj(q))
    return (float(t[0]), float(t[1]), float(t[2]))


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected end of file")
    return b


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_chunk_header(f) -> tuple[bytes, int, int]:
    chunk_id = _read_exact(f, 4)
    content_size = _read_u32(f)
    children_size = _read_u32(f)
    return chunk_id, int(content_size), int(children_size)


def _read_i32_from(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<i", buf, off)[0], off + 4


def _read_u32_from(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buf, off)[0], off + 4


def _read_str_from(buf: bytes, off: int) -> tuple[str, int]:
    ln, off = _read_u32_from(buf, off)
    s = buf[off : off + ln].decode("utf-8", errors="replace")
    return s, off + ln


def _read_dict_from(buf: bytes, off: int) -> tuple[dict[str, str], int]:
    n, off = _read_u32_from(buf, off)
    d: dict[str, str] = {}
    for _ in range(int(n)):
        k, off = _read_str_from(buf, off)
        v, off = _read_str_from(buf, off)
        d[k] = v
    return d, off


def _parse_scenegraph(
    path: Path,
) -> tuple[
    list[VoxNode],
    list[int],
    list[tuple[float, float, float]],
    list[tuple[float, float, float, float]],
]:
    """Parse VOX scene graph and also compute per-model world transforms.

    Returns:
    - nodes: list of VoxNode indexed by node_id (sparse ids allowed; missing ids are filled with dummy grp)
    - root_node_ids: ids of root nodes
    - model_world_translations: list indexed by model id
    - model_world_rotations: list indexed by model id
    """

    try:
        with path.open("rb") as f:
            if _read_exact(f, 4) != b"VOX ":
                return ([], [], [], [])
            _ = _read_u32(f)  # version
            chunk_id, content_size, children_size = _read_chunk_header(f)
            if chunk_id != b"MAIN":
                return ([], [], [], [])
            if content_size:
                f.seek(content_size, 1)
            end_of_main = f.tell() + children_size

            trn_nodes: dict[int, tuple[int, tuple[float, float, float], tuple[float, float, float, float]]] = {}
            grp_nodes: dict[int, list[int]] = {}
            shp_nodes: dict[int, list[int]] = {}
            node_names: dict[int, str] = {}
            child_nodes: set[int] = set()
            max_model_id = -1
            all_node_ids: set[int] = set()

            while f.tell() < end_of_main:
                cid, csize, chsize = _read_chunk_header(f)
                content = _read_exact(f, csize)

                if cid == b"nTRN":
                    off = 0
                    node_id, off = _read_i32_from(content, off)
                    node_dict, off = _read_dict_from(content, off)
                    n = node_dict.get("_name")
                    if n:
                        node_names[int(node_id)] = str(n)
                    child_id, off = _read_i32_from(content, off)
                    child_nodes.add(int(child_id))
                    _, off = _read_i32_from(content, off)
                    _, off = _read_i32_from(content, off)
                    num_frames, off = _read_i32_from(content, off)

                    t = (0.0, 0.0, 0.0)
                    r = (0.0, 0.0, 0.0, 1.0)
                    if num_frames > 0:
                        frame_dict, off = _read_dict_from(content, off)
                        tt = frame_dict.get("_t")
                        if tt:
                            parts = tt.replace(",", " ").split()
                            if len(parts) >= 3:
                                try:
                                    t = (float(int(parts[0])), float(int(parts[1])), float(int(parts[2])))
                                except ValueError:
                                    pass
                        rr = frame_dict.get("_r")
                        if rr:
                            try:
                                rb = int(str(rr).strip())
                                r = _quat_from_mat3(_decode_rotation_byte(rb))
                            except ValueError:
                                pass

                    trn_nodes[int(node_id)] = (int(child_id), t, r)
                    all_node_ids.add(int(node_id))
                    all_node_ids.add(int(child_id))

                elif cid == b"nGRP":
                    off = 0
                    node_id, off = _read_i32_from(content, off)
                    node_dict, off = _read_dict_from(content, off)
                    n = node_dict.get("_name")
                    if n:
                        node_names[int(node_id)] = str(n)
                    nchild, off = _read_i32_from(content, off)
                    children: list[int] = []
                    for _ in range(int(nchild)):
                        ch, off = _read_i32_from(content, off)
                        children.append(int(ch))
                        child_nodes.add(int(ch))
                    grp_nodes[int(node_id)] = children
                    all_node_ids.add(int(node_id))
                    for ch in children:
                        all_node_ids.add(int(ch))

                elif cid == b"nSHP":
                    off = 0
                    node_id, off = _read_i32_from(content, off)
                    node_dict, off = _read_dict_from(content, off)
                    n = node_dict.get("_name")
                    if n:
                        node_names[int(node_id)] = str(n)
                    nmodels, off = _read_i32_from(content, off)
                    mids: list[int] = []
                    for _ in range(int(nmodels)):
                        mid, off = _read_i32_from(content, off)
                        _, off = _read_dict_from(content, off)
                        mids.append(int(mid))
                        if int(mid) > max_model_id:
                            max_model_id = int(mid)
                    shp_nodes[int(node_id)] = mids
                    all_node_ids.add(int(node_id))

                if chsize:
                    f.seek(chsize, 1)

            if max_model_id < 0:
                return ([], [], [], [])

            translations: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * (max_model_id + 1)
            rotations: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 1.0)] * (max_model_id + 1)

            def walk(node_id: int, wt: tuple[float, float, float], wr: tuple[float, float, float, float]) -> None:
                if node_id in trn_nodes:
                    child_id, lt, lr = trn_nodes[node_id]
                    # world = parent * local
                    rt = _quat_rotate_vec(wr, lt)
                    nwt = (wt[0] + rt[0], wt[1] + rt[1], wt[2] + rt[2])
                    nwr = _quat_norm(_quat_mul(wr, lr))
                    walk(child_id, nwt, nwr)
                    return

                if node_id in grp_nodes:
                    for ch in grp_nodes[node_id]:
                        walk(ch, wt, wr)
                    return

                if node_id in shp_nodes:
                    for mid in shp_nodes[node_id]:
                        if 0 <= mid < len(translations):
                            translations[mid] = wt
                            rotations[mid] = wr
                    return

            roots: set[int] = set()
            roots.update(trn_nodes.keys())
            roots.update(grp_nodes.keys())
            roots.update(shp_nodes.keys())
            roots = {rid for rid in roots if rid not in child_nodes}
            if not roots:
                # Fallback: try walking node 0 if present
                roots = {0} if (0 in trn_nodes or 0 in grp_nodes or 0 in shp_nodes) else set()

            for rid in sorted(roots):
                walk(int(rid), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

            # Build VoxNode list indexed by id.
            max_node_id = max(all_node_ids) if all_node_ids else -1
            nodes: list[VoxNode] = [VoxNode(kind="grp", name=f"node_{i}") for i in range(max_node_id + 1)]
            for nid in range(max_node_id + 1):
                nodes[nid].children = []
                nodes[nid].model_ids = []

            for nid, (child_id, lt, lr) in trn_nodes.items():
                name = node_names.get(int(nid)) or f"trn_{nid}"
                nodes[nid] = VoxNode(kind="trn", name=str(name), translation=lt, rotation=lr, children=[int(child_id)])

            for nid, children in grp_nodes.items():
                name = node_names.get(int(nid)) or f"grp_{nid}"
                nodes[nid] = VoxNode(kind="grp", name=str(name), children=[int(c) for c in children])

            for nid, mids in shp_nodes.items():
                name = node_names.get(int(nid)) or f"shp_{nid}"
                nodes[nid] = VoxNode(kind="shp", name=str(name), model_ids=[int(mid) for mid in mids])

            return nodes, [int(r) for r in sorted(roots)], translations, rotations
    except Exception:
        return ([], [], [], [])
