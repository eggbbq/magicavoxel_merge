"""High-level pipeline tying together VOX IO, mesher, atlas, and exporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import json
import sys

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
    tight_blocks: bool = False
    style: str = "baked"  # baked or solid


@dataclass(slots=True)
class PipelineOptions:
    axis: str = "y_up"
    scale: float = 1.0
    pivot: str = "corner"
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
    debug_transforms_out: str | None = None,
    print_nodes: bool = False,
    output_format: str = "glb",
    options: PipelineOptions | None = None,
) -> None:
    """Run the Block-Topology Preservation pipeline end-to-end."""

    opts = options or PipelineOptions()
    scene = load_scene(input_vox)

    if bool(print_nodes):
        _print_scene_nodes(scene)
    mesher_result = build_quads(scene)

    atlas_result = atlas_mod.build_atlas(
        scene,
        mesher_result,
        pad=int(opts.atlas.pad),
        inset=float(opts.atlas.inset),
        texel_scale=int(opts.atlas.texel_scale),
        square=bool(opts.atlas.square),
        pot=bool(opts.atlas.pot),
        layout=str(opts.atlas.layout),
        tight_blocks=bool(opts.atlas.tight_blocks),
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

    meshes, model_to_mesh = _build_model_meshes(
        scene,
        mesher_result,
        atlas_result,
        scale=float(opts.scale),
        flip_v=bool(opts.flip_v),
        pivot=str(opts.pivot),
    )
    nodes, root_node_ids = _build_scene_nodes_two_level(scene, model_to_mesh, scale=float(opts.scale))

    if debug_transforms_out:
        _write_transform_debug(debug_transforms_out, scene=scene, meshes=meshes, stage="pre_axis")

    meshes = _to_y_up_left_handed(meshes)
    nodes = _to_y_up_left_handed_nodes(nodes)
    if bool(opts.bake_translation):
        meshes = _bake_translation_into_vertices(meshes)

    if debug_transforms_out:
        _write_transform_debug(debug_transforms_out, scene=scene, meshes=meshes, stage="post_axis")

    if str(output_format) == "gltf":
        glb_writer.write_scene_gltf(
            output_glb,
            meshes=meshes,
            nodes=nodes,
            root_node_ids=root_node_ids,
            texture_png=texture_png,
            texture_path=texture_uri,
            name_prefix=Path(output_glb).stem,
        )
    else:
        glb_writer.write_scene(
            output_glb,
            meshes=meshes,
            nodes=nodes,
            root_node_ids=root_node_ids,
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


def _print_scene_nodes(scene: VoxScene) -> None:
    sys.stderr.write(f"[btp_vox] scene nodes: count={len(scene.nodes)} roots={scene.root_node_ids}\n")
    # Flat listing.
    for nid, nd in enumerate(scene.nodes):
        sys.stderr.write(
            f"[btp_vox] node id={nid} kind={nd.kind} name={nd.name} "
            f"t={tuple(float(x) for x in nd.translation)} "
            f"r={tuple(float(x) for x in nd.rotation)} "
            f"children={list(nd.children)} models={list(nd.model_ids)}\n"
        )

    # Tree traversal from roots.
    sys.stderr.write("[btp_vox] scene tree:\n")
    visited: set[int] = set()

    def walk(node_id: int, depth: int) -> None:
        if node_id in visited:
            sys.stderr.write(f"[btp_vox] {'  ' * depth}- id={node_id} (cycle)\n")
            return
        visited.add(int(node_id))
        if node_id < 0 or node_id >= len(scene.nodes):
            sys.stderr.write(f"[btp_vox] {'  ' * depth}- id={node_id} (out_of_range)\n")
            return
        nd = scene.nodes[int(node_id)]
        sys.stderr.write(
            f"[btp_vox] {'  ' * depth}- id={node_id} kind={nd.kind} name={nd.name} "
            f"children={list(nd.children)} models={list(nd.model_ids)}\n"
        )
        for ch in nd.children:
            walk(int(ch), depth + 1)

    for rid in scene.root_node_ids:
        walk(int(rid), 0)


def _write_transform_debug(path: str | Path, *, scene: VoxScene, meshes: list[dict], stage: str) -> None:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)

    prev: dict = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    model_rows: list[dict] = []
    for midx, m in enumerate(scene.models):
        model_rows.append(
            {
                "model_id": int(midx),
                "name": m.name,
                "scene_translation": [float(m.translation[0]), float(m.translation[1]), float(m.translation[2])],
                "scene_rotation": [float(m.rotation[0]), float(m.rotation[1]), float(m.rotation[2]), float(m.rotation[3])],
            }
        )

    mesh_rows: list[dict] = []
    for i, mm in enumerate(meshes):
        mesh_rows.append(
            {
                "mesh_index": int(i),
                "name": str(mm.get("name")),
                "model_id": int(mm.get("model_id", -1)),
                "translation": [float(x) for x in (mm.get("translation") or (0.0, 0.0, 0.0))],
                "rotation": [float(x) for x in (mm.get("rotation") or (0.0, 0.0, 0.0, 1.0))],
                "pivot_p": [float(x) for x in (mm.get("pivot_p") or (0.0, 0.0, 0.0))],
                "pivot_rp": [float(x) for x in (mm.get("pivot_rp") or (0.0, 0.0, 0.0))],
            }
        )

    prev[str(stage)] = {"models": model_rows, "meshes": mesh_rows}
    p.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")


def _assemble_meshes(
    scene: VoxScene,
    mesher_result: MesherResult,
    atlas_result: atlas_mod.AtlasBuildResult,
    *,
    scale: float,
    flip_v: bool,
    pivot: str = "corner",
) -> List[dict]:
    # IMPORTANT: We assemble meshes per scene-graph shape instance (nSHP), not per model.
    # A single model_id can be referenced by multiple nSHP nodes (instancing). If we only
    # export one mesh per model_id, those additional instances disappear.
    meshes: List[dict] = []

    if pivot not in ("corner", "bottom_center", "center"):
        raise ValueError("pivot must be one of: corner, bottom_center, center")

    model_geom_cache: dict[int, dict] = {}

    def build_model_geom(midx: int) -> dict | None:
        if midx in model_geom_cache:
            return model_geom_cache[midx]

        quads = mesher_result.quads_per_model[midx]
        positions: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        texcoords: list[tuple[float, float]] = []
        indices: list[int] = []

        for qidx, quad in enumerate(quads):
            uv_rect = atlas_result.quad_uvs[(midx, qidx)]

            p_arr = np.asarray(quad.verts, dtype=np.float32)
            p0, p1, _p2, p3 = p_arr
            edge_u = p1 - p0
            edge_v = p3 - p0
            face_normal = np.cross(edge_u, edge_v)
            tri_order = (0, 1, 2, 0, 2, 3)
            if float(np.dot(face_normal, np.asarray(quad.normal, dtype=np.float32))) < 0.0:
                tri_order = (0, 2, 1, 0, 3, 2)

            # Determine which quad edge corresponds to (size_u, size_v).
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
            vox = np.asarray(scene.models[midx].voxels)
            if vox.size:
                try:
                    vcnt = int(vox.shape[0])
                except Exception:
                    vcnt = int(vox.size)
                sys.stderr.write(
                    f"[btp_vox] warning: model has voxels but produced no quads; skipped export. "
                    f"model_id={midx} name={scene.models[midx].name} voxel_count={vcnt} size={scene.models[midx].size}\n"
                )
            model_geom_cache[midx] = {}
            return None

        pos_arr = np.asarray(positions, dtype=np.float32)
        sx, sy, sz = scene.models[midx].size
        if pivot == "corner":
            p = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
        elif pivot == "bottom_center":
            p = np.asarray((float(sx) * 0.5 * scale, float(sy) * 0.5 * scale, 0.0), dtype=np.float32)
        else:
            p = np.asarray((float(sx) * 0.5 * scale, float(sy) * 0.5 * scale, float(sz) * 0.5 * scale), dtype=np.float32)
        if pos_arr.size:
            pos_arr = pos_arr.copy()
            pos_arr[:, 0] -= float(p[0])
            pos_arr[:, 1] -= float(p[1])
            pos_arr[:, 2] -= float(p[2])

        geom = {
            "positions": pos_arr,
            "normals": np.asarray(normals, dtype=np.float32),
            "texcoords": np.asarray(texcoords, dtype=np.float32),
            "indices": np.asarray(indices, dtype=np.uint32),
            "pivot_p": (float(p[0]), float(p[1]), float(p[2])),
        }
        model_geom_cache[midx] = geom
        return geom

    def merge_geoms(geoms: list[tuple[int, dict]]) -> dict | None:
        if not geoms:
            return None
        pos_list: list[np.ndarray] = []
        nrm_list: list[np.ndarray] = []
        uv_list: list[np.ndarray] = []
        idx_list: list[np.ndarray] = []
        base = 0
        for _midx, g in geoms:
            pos = np.asarray(g["positions"], dtype=np.float32)
            nrm = np.asarray(g["normals"], dtype=np.float32)
            uv = np.asarray(g["texcoords"], dtype=np.float32)
            idx = np.asarray(g["indices"], dtype=np.uint32)
            pos_list.append(pos)
            nrm_list.append(nrm)
            uv_list.append(uv)
            idx_list.append(idx + np.uint32(base))
            base += int(pos.shape[0])
        if base == 0:
            return None
        return {
            "positions": np.vstack(pos_list),
            "normals": np.vstack(nrm_list),
            "texcoords": np.vstack(uv_list),
            "indices": np.concatenate(idx_list),
        }

    def walk(node_id: int, wt: tuple[float, float, float], wr: tuple[float, float, float, float]) -> None:
        if node_id < 0 or node_id >= len(scene.nodes):
            return
        nd = scene.nodes[int(node_id)]
        if nd.kind == "trn":
            lt = nd.translation
            lr = nd.rotation
            # MagicaVoxel nTRN translations are authored in parent space; do NOT rotate by parent rotation.
            nwt = (wt[0] + lt[0], wt[1] + lt[1], wt[2] + lt[2])
            nwr = _quat_mul(wr, lr)
            nwr = _quat_norm(nwr)
            for ch in nd.children:
                walk(int(ch), nwt, nwr)
            return

        if nd.kind == "grp":
            for ch in nd.children:
                walk(int(ch), wt, wr)
            return

        if nd.kind == "shp":
            geoms: list[tuple[int, dict]] = []
            for mid in nd.model_ids:
                if 0 <= int(mid) < len(scene.models):
                    g = build_model_geom(int(mid))
                    if g is not None:
                        geoms.append((int(mid), g))
            merged = merge_geoms(geoms)
            if merged is None:
                return

            t = np.asarray((float(wt[0]) * scale, float(wt[1]) * scale, float(wt[2]) * scale), dtype=np.float32)
            r = (float(wr[0]), float(wr[1]), float(wr[2]), float(wr[3]))

            # Name instances uniquely to avoid editor collapsing duplicates.
            if len(nd.model_ids) == 1 and 0 <= int(nd.model_ids[0]) < len(scene.models):
                base_name = scene.models[int(nd.model_ids[0])].name
            else:
                base_name = nd.name
            name = f"{base_name}__{node_id}"

            meshes.append(
                {
                    "name": name,
                    "model_id": int(nd.model_ids[0]) if nd.model_ids else -1,
                    "positions": merged["positions"],
                    "normals": merged["normals"],
                    "texcoords": merged["texcoords"],
                    "indices": merged["indices"],
                    "translation": (float(t[0]), float(t[1]), float(t[2])),
                    "rotation": r,
                    "pivot_p": (0.0, 0.0, 0.0),
                    "pivot_rp": (0.0, 0.0, 0.0),
                }
            )
            return

    for rid in scene.root_node_ids:
        walk(int(rid), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    return meshes


def _build_model_meshes(
    scene: VoxScene,
    mesher_result: MesherResult,
    atlas_result: atlas_mod.AtlasBuildResult,
    *,
    scale: float,
    flip_v: bool,
    pivot: str = "corner",
) -> tuple[list[dict], dict[int, int]]:
    if pivot not in ("corner", "bottom_center", "center"):
        raise ValueError("pivot must be one of: corner, bottom_center, center")

    meshes: list[dict] = []
    model_to_mesh: dict[int, int] = {}

    for midx, quads in enumerate(mesher_result.quads_per_model):
        positions: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        texcoords: list[tuple[float, float]] = []
        indices: list[int] = []

        for qidx, quad in enumerate(quads):
            uv_rect = atlas_result.quad_uvs[(midx, qidx)]

            p_arr = np.asarray(quad.verts, dtype=np.float32)
            p0, p1, _p2, p3 = p_arr
            edge_u = p1 - p0
            edge_v = p3 - p0
            face_normal = np.cross(edge_u, edge_v)
            tri_order = (0, 1, 2, 0, 2, 3)
            if float(np.dot(face_normal, np.asarray(quad.normal, dtype=np.float32))) < 0.0:
                tri_order = (0, 2, 1, 0, 3, 2)

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
            vox = np.asarray(scene.models[midx].voxels)
            if vox.size:
                try:
                    vcnt = int(vox.shape[0])
                except Exception:
                    vcnt = int(vox.size)
                sys.stderr.write(
                    f"[btp_vox] warning: model has voxels but produced no quads; skipped export. "
                    f"model_id={midx} name={scene.models[midx].name} voxel_count={vcnt} size={scene.models[midx].size}\n"
                )
            continue

        pos_arr = np.asarray(positions, dtype=np.float32)
        sx, sy, sz = scene.models[midx].size
        if pivot == "corner":
            p = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
        elif pivot == "bottom_center":
            p = np.asarray((float(sx) * 0.5 * scale, float(sy) * 0.5 * scale, 0.0), dtype=np.float32)
        else:
            p = np.asarray((float(sx) * 0.5 * scale, float(sy) * 0.5 * scale, float(sz) * 0.5 * scale), dtype=np.float32)

        if pos_arr.size:
            pos_arr = pos_arr.copy()
            pos_arr[:, 0] -= float(p[0])
            pos_arr[:, 1] -= float(p[1])
            pos_arr[:, 2] -= float(p[2])

        mesh_index = len(meshes)
        meshes.append(
            {
                "name": scene.models[midx].name,
                "model_id": int(midx),
                "positions": pos_arr,
                "normals": np.asarray(normals, dtype=np.float32),
                "texcoords": np.asarray(texcoords, dtype=np.float32),
                "indices": np.asarray(indices, dtype=np.uint32),
            }
        )
        model_to_mesh[int(midx)] = int(mesh_index)

    return meshes, model_to_mesh


def _build_scene_nodes(scene: VoxScene, model_to_mesh: dict[int, int]) -> tuple[list[dict], list[int]]:
    nodes: list[dict] = []
    # Pre-create nodes matching VOX node indices.
    for nd in scene.nodes:
        out: dict = {"name": nd.name, "children": list(nd.children)}
        if nd.kind == "trn":
            out["translation"] = tuple(float(x) for x in nd.translation)
            out["rotation"] = tuple(float(x) for x in nd.rotation)
        nodes.append(out)

    # Attach meshes to shape nodes. Avoid extra nodes unless a shp references multiple models.
    for nid, nd in enumerate(scene.nodes):
        if nd.kind != "shp":
            continue
        model_ids = [int(x) for x in nd.model_ids]
        mesh_ids = [model_to_mesh[mid] for mid in model_ids if mid in model_to_mesh]
        if not mesh_ids:
            continue
        if len(mesh_ids) == 1:
            nodes[nid]["mesh"] = int(mesh_ids[0])
        else:
            # Minimal extra nodes: only for multi-model shapes.
            extra_children: list[int] = []
            for midx, mesh_id in zip(model_ids, mesh_ids):
                child_id = len(nodes)
                extra_children.append(child_id)
                nodes.append({"name": f"{scene.models[midx].name}__inst", "mesh": int(mesh_id), "children": []})
            nodes[nid]["children"] = list(nodes[nid].get("children") or []) + extra_children

    return nodes, [int(r) for r in scene.root_node_ids]


def _build_scene_nodes_two_level(
    scene: VoxScene,
    model_to_mesh: dict[int, int],
    *,
    scale: float = 1.0,
) -> tuple[list[dict], list[int]]:
    # Flatten VOX scene graph into: shp parent nodes -> mesh child nodes.
    # This enforces a shallow hierarchy while preserving per-instance world transforms.

    world_t: dict[int, tuple[float, float, float]] = {}
    world_r: dict[int, tuple[float, float, float, float]] = {}

    def walk(nid: int, parent_t: tuple[float, float, float], parent_r: tuple[float, float, float, float]) -> None:
        nd = scene.nodes[nid]
        t = parent_t
        r = parent_r
        if nd.kind == "trn":
            lt = tuple(float(x) for x in nd.translation)
            lr = tuple(float(x) for x in nd.rotation)
            # MagicaVoxel nTRN translation is in parent space; do not rotate by parent.
            t = (t[0] + lt[0], t[1] + lt[1], t[2] + lt[2])
            r = _quat_norm(_quat_mul(r, lr))

        world_t[nid] = t
        world_r[nid] = r

        for ch in nd.children:
            walk(int(ch), t, r)

    for rid in scene.root_node_ids:
        walk(int(rid), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    nodes: list[dict] = []
    parent_ids: list[int] = []
    for nid, nd in enumerate(scene.nodes):
        if nd.kind != "shp":
            continue

        t0 = world_t.get(nid, (0.0, 0.0, 0.0))
        t = (t0[0] * float(scale), t0[1] * float(scale), t0[2] * float(scale))
        r = world_r.get(nid, (0.0, 0.0, 0.0, 1.0))
        model_ids = [int(x) for x in nd.model_ids]
        if not model_ids:
            continue

        parent_name = str(nd.name)

        parent_id = len(nodes)
        nodes.append(
            {
                "name": parent_name,
                "children": [],
                "translation": t,
                "rotation": r,
            }
        )

        child_ids: list[int] = []
        for mid in model_ids:
            mesh_id = model_to_mesh.get(mid)
            if mesh_id is None:
                continue

            child_id = len(nodes)
            nodes.append(
                {
                    "name": str(scene.models[mid].name),
                    "children": [],
                    "mesh": int(mesh_id),
                }
            )
            child_ids.append(child_id)

        if not child_ids:
            # No mesh exported for this shp; drop parent.
            nodes.pop()
            continue

        nodes[parent_id]["children"] = child_ids
        parent_ids.append(parent_id)

    return nodes, parent_ids


def _to_y_up_left_handed_nodes(nodes: list[dict]) -> list[dict]:
    axis_m = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    b = axis_m.T
    bt = b.T
    h = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)

    out: list[dict] = []
    for nd in nodes:
        mm = dict(nd)

        tr = mm.get("translation")
        if tr is not None:
            t = np.asarray(tr, dtype=np.float32) @ axis_m
            t[2] *= -1.0
            mm["translation"] = (float(t[0]), float(t[1]), float(t[2]))

        rot = mm.get("rotation")
        if rot is not None:
            rmat = _quat_to_mat3(tuple(rot))
            rmat = b @ rmat @ bt
            rmat = h @ rmat @ h
            mm["rotation"] = _mat3_to_quat(rmat)

        out.append(mm)

    return out


def _quat_mul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx, by, bz, bw = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_norm(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (float(x / n), float(y / n), float(z / n), float(w / n))


def _quat_rotate_vec(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    # Rotate vector v by quaternion q.
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    vx, vy, vz = (float(v[0]), float(v[1]), float(v[2]))
    # q * (v,0)
    ix = w * vx + y * vz - z * vy
    iy = w * vy + z * vx - x * vz
    iz = w * vz + x * vy - y * vx
    iw = -x * vx - y * vy - z * vz
    # (q * v) * conj(q)
    rx = ix * w + iw * -x + iy * -z - iz * -y
    ry = iy * w + iw * -y + iz * -x - ix * -z
    rz = iz * w + iw * -z + ix * -y - iy * -x
    return (float(rx), float(ry), float(rz))


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
    """Match magicavoxel_merge's coordinate + normal conversion.

    1) Axis map MV->Y-up: (x,y,z) -> (x,z,-y)
    2) Left-handed output: mirror Z and flip triangle winding.
    """

    # Row-vector transform for axis map: v' = v @ m
    axis_m = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Column-vector basis change for rotations: v' = b @ v
    b = axis_m.T
    bt = b.T

    # Handedness flip (mirror Z): v' = h @ v (column vectors)
    h = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)

    out: List[dict] = []
    for mesh in meshes:
        mm = dict(mesh)

        pos = np.asarray(mm["positions"], dtype=np.float32)
        nrm = np.asarray(mm["normals"], dtype=np.float32)
        idx = np.asarray(mm["indices"], dtype=np.uint32)

        # Axis map (MV->Y-up): (x,y,z)->(x,z,-y)
        pos = pos @ axis_m
        nrm = nrm @ axis_m

        tr = mm.get("translation")
        if tr is not None:
            t = np.asarray(tr, dtype=np.float32)
            t = t @ axis_m
            mm["translation"] = (float(t[0]), float(t[1]), float(t[2]))

        rot = mm.get("rotation")
        if rot is not None:
            rmat = _quat_to_mat3(tuple(rot))
            # Apply axis basis change, then handedness conjugation.
            rmat = b @ rmat @ bt
            rmat = h @ rmat @ h
            mm["rotation"] = _mat3_to_quat(rmat)

        # Mirror Z for left-handed output
        pos[:, 2] *= -1.0
        nrm[:, 2] *= -1.0
        if tr is not None:
            tx, ty, tz = mm["translation"]
            mm["translation"] = (float(tx), float(ty), -float(tz))

        # Flip winding to match the handedness reflection.
        if idx.size % 3 != 0:
            raise ValueError("indices length must be multiple of 3")
        tris = idx.reshape((-1, 3))
        idx = tris[:, [0, 2, 1]].reshape((-1,)).astype(np.uint32)

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
