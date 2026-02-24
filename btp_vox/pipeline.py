"""High-level pipeline tying together VOX IO, mesher, atlas, and exporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from . import atlas as atlas_mod
from . import uv_export, glb_writer
from .btp_mesher import MesherResult, build_quads
from .voxio import VoxScene, load_scene


@dataclass(slots=True)
class AtlasOptions:
    pad: int = 2
    inset: float = 1.0
    texel_scale: int = 1
    layout: str = "by-model"  # or "global"
    square: bool = False
    pot: bool = False
    style: str = "baked"  # baked or solid


@dataclass(slots=True)
class PipelineOptions:
    axis: str = "y_up"
    scale: float = 1.0
    center: bool = False
    center_bounds: bool = False
    weld: bool = False
    flip_v: bool = False
    bake_translation: bool = False
    atlas: AtlasOptions = field(default_factory=AtlasOptions)


def convert(
    input_vox: str | Path,
    output_glb: str | Path,
    *,
    texture_out: str | None = None,
    uv_json_out: str | None = None,
    options: PipelineOptions | None = None,
) -> None:
    """Run the Block-Topology Preservation pipeline end-to-end."""

    opts = options or PipelineOptions()
    scene = load_scene(input_vox)
    mesher_result = build_quads(scene)

    atlas_result = atlas_mod.build_atlas(
        scene,
        mesher_result,
        pad=int(opts.atlas.pad),
        inset=float(opts.atlas.inset),
        texel_scale=int(opts.atlas.texel_scale),
        square=bool(opts.atlas.square),
        pot=bool(opts.atlas.pot),
        style=str(opts.atlas.style),
    )

    texture_uri = None
    texture_png: bytes | None = atlas_result.texture_png
    if texture_out:
        texture_path = Path(texture_out)
        if texture_path.parent:
            texture_path.parent.mkdir(parents=True, exist_ok=True)
        texture_path.write_bytes(atlas_result.texture_png)
        texture_uri = texture_path.name
        texture_png = None

    meshes = _assemble_meshes(
        scene,
        mesher_result,
        atlas_result,
        scale=float(opts.scale),
        flip_v=bool(opts.flip_v),
    )

    meshes = _to_y_up_left_handed(meshes)
    if bool(opts.bake_translation):
        meshes = _bake_translation_into_vertices(meshes)
    glb_writer.write_meshes(
        output_glb,
        meshes=meshes,
        texture_png=texture_png,
        texture_path=texture_uri,
        name_prefix=Path(output_glb).stem,
    )

    if uv_json_out:
        name_counts: dict[str, int] = {}
        rects: dict[str, tuple[float, float, float, float]] = {}
        for midx, rect in atlas_result.model_uv_rects.items():
            name = scene.models[midx].name
            if name in name_counts:
                name_counts[name] += 1
                key = f"{name}_{name_counts[name]}"
            else:
                name_counts[name] = 0
                key = name
            rects[key] = rect
        uv_export.write_uv_json(uv_json_out, width=atlas_result.width, height=atlas_result.height, model_rects=rects)


def _assemble_meshes(
    scene: VoxScene,
    mesher_result: MesherResult,
    atlas_result: atlas_mod.AtlasBuildResult,
    *,
    scale: float,
    flip_v: bool,
) -> List[dict]:
    meshes: List[dict] = []

    for midx, quads in enumerate(mesher_result.quads_per_model):
        tx, ty, tz = scene.models[midx].translation
        rx, ry, rz, rw = scene.models[midx].rotation
        positions: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        texcoords: list[tuple[float, float]] = []
        indices: list[int] = []

        for qidx, quad in enumerate(quads):
            uv_rect = atlas_result.quad_uvs[(midx, qidx)]

            p_arr = np.asarray(quad.verts, dtype=np.float32)
            p0, p1, p2, p3 = p_arr
            edge_u = p1 - p0
            edge_v = p3 - p0
            face_normal = np.cross(edge_u, edge_v)
            tri_order = (0, 1, 2, 0, 2, 3)
            if float(np.dot(face_normal, np.asarray(quad.normal, dtype=np.float32))) < 0.0:
                tri_order = (0, 2, 1, 0, 3, 2)

            # Determine which quad edge corresponds to (size_u, size_v).
            # This avoids per-face rotated/mirrored atlas sampling.
            w_vox = float(quad.size_u) * float(scale)
            h_vox = float(quad.size_v) * float(scale)
            e1 = edge_u
            e3 = edge_v
            len1 = float(np.linalg.norm(e1))
            len3 = float(np.linalg.norm(e3))
            d_keep = abs(len1 - w_vox) + abs(len3 - h_vox)
            d_swap = abs(len1 - h_vox) + abs(len3 - w_vox)
            if d_swap < d_keep:
                u_axis = e3
                v_axis = e1
            else:
                u_axis = e1
                v_axis = e3

            uu = float(np.dot(u_axis, u_axis)) or 1.0
            vv = float(np.dot(v_axis, v_axis)) or 1.0
            u_span = uv_rect.u1 - uv_rect.u0
            v_span = uv_rect.v1 - uv_rect.v0

            def map_uv(idx: int) -> tuple[float, float]:
                rel = p_arr[idx] - p0
                a = float(np.dot(rel, u_axis) / uu)
                b = float(np.dot(rel, v_axis) / vv)
                a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
                b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
                u = uv_rect.u0 + u_span * a
                v = uv_rect.v0 + v_span * b
                if flip_v:
                    v = 1.0 - v
                return (float(u), float(v))

            base = len(positions)

            for vx in quad.verts:
                positions.append((vx[0] * scale, vx[1] * scale, vx[2] * scale))
                normals.append(quad.normal)

            texcoords.extend(map_uv(i) for i in range(4))
            indices.extend(base + idx for idx in tri_order)

        if not positions:
            continue

        meshes.append(
            {
                "name": scene.models[midx].name,
                "positions": np.asarray(positions, dtype=np.float32),
                "normals": np.asarray(normals, dtype=np.float32),
                "texcoords": np.asarray(texcoords, dtype=np.float32),
                "indices": np.asarray(indices, dtype=np.uint32),
                "translation": (float(tx) * scale, float(ty) * scale, float(tz) * scale),
                "rotation": (float(rx), float(ry), float(rz), float(rw)),
            }
        )

    return meshes


def _quat_to_mat3(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _mat3_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
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


def _to_y_up_left_handed(meshes: List[dict]) -> List[dict]:
    """Convert from MagicaVoxel coords to Y-up, then mirror Z for left-handed output.

    Axis map used by the main project: (x,y,z) -> (x,z,-y)
    Left-handed conversion: mirror Z and flip triangle winding.
    """

    # Basis change MV->Y-up.
    b = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    bt = b.T
    h = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)

    out: List[dict] = []
    for m in meshes:
        mm = dict(m)

        pos = np.asarray(mm["positions"], dtype=np.float32)
        nrm = np.asarray(mm["normals"], dtype=np.float32)
        idx = np.asarray(mm["indices"], dtype=np.uint32)

        # Axis map
        pos = pos @ bt
        nrm = nrm @ bt

        tr = mm.get("translation")
        if tr is not None:
            t = np.asarray(tr, dtype=np.float32)
            t = t @ bt
            mm["translation"] = (float(t[0]), float(t[1]), float(t[2]))

        rot = mm.get("rotation")
        if rot is not None:
            rmat = _quat_to_mat3(tuple(rot))
            rmat = b @ rmat @ bt
            rmat = h @ rmat @ h
            mm["rotation"] = _mat3_to_quat(rmat)

        # Mirror Z for left-handed
        pos[:, 2] *= -1.0
        nrm[:, 2] *= -1.0
        if tr is not None:
            tx, ty, tz = mm["translation"]
            mm["translation"] = (float(tx), float(ty), -float(tz))

        if idx.size % 3 != 0:
            raise ValueError("indices length must be multiple of 3")
        idx = idx.reshape((-1, 3))[:, [0, 2, 1]].reshape((-1,)).astype(np.uint32)

        mm["positions"] = pos
        mm["normals"] = nrm
        mm["indices"] = idx
        out.append(mm)

    return out


def _bake_translation_into_vertices(meshes: List[dict]) -> List[dict]:
    out: List[dict] = []
    for m in meshes:
        mm = dict(m)
        tr = mm.get("translation")
        if tr is None:
            out.append(mm)
            continue

        t = np.asarray(tr, dtype=np.float32)
        pos = np.asarray(mm["positions"], dtype=np.float32)
        if pos.size:
            pos = pos.copy()
            pos[:, 0] += float(t[0])
            pos[:, 1] += float(t[1])
            pos[:, 2] += float(t[2])
        mm["positions"] = pos
        mm["translation"] = (0.0, 0.0, 0.0)
        out.append(mm)
    return out
