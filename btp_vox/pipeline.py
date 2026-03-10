"""High-level pipeline tying together VOX IO, mesher, atlas, and exporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import math
import json
import os
import sys
import time

import numpy as np

from . import atlas as atlas_mod
from . import uv_export, glb_writer
from .btp_mesher import MesherResult, Quad, build_quads
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
    export_uv2: bool = False
    uv2_mode: str = "copy"
    export_vertex_color: bool = False
    no_merge_nodes: bool = False
    character_apart: bool = False
    character_flat: bool = False
    cull: str = ""
    plat_cutout: bool = False
    plat_cutoff: float = 0.5
    plat_suffix: str = "-cutout"
    texture_alpha: str = "auto"
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

    timings_enabled = bool(os.environ.get("BTP_VOX_TIMINGS"))
    t0 = time.perf_counter()
    last_t = t0

    def mark(stage: str) -> None:
        nonlocal last_t
        if not timings_enabled:
            return
        now = time.perf_counter()
        sys.stderr.write(
            f"[btp_vox] timing {stage}: +{(now - last_t):.3f}s (total {(now - t0):.3f}s)\n"
        )
        last_t = now

    opts = options or PipelineOptions()
    scene = load_scene(input_vox)
    mark("load_scene")

    if bool(print_nodes):
        _print_scene_nodes(scene)
    plat_suffix = str(getattr(opts, "plat_suffix", "") or "").strip()
    if plat_suffix in ("plat-t", "-plat-t"):
        plat_suffix = "-plat-t"
    elif plat_suffix in ("plat-f", "-plat-f"):
        plat_suffix = "-plat-f"
    elif plat_suffix == "cutout":
        plat_suffix = "-cutout"
    use_plat_suffix = (not bool(opts.plat_cutout))
    if bool(opts.plat_cutout):
        mesher_result = _build_plat_cutout_quads(scene)
        any_cutout = True
    else:
        mesher_result = build_quads(scene)
        any_cutout = False
        if use_plat_suffix:
            qpm = list(mesher_result.quads_per_model)
            replaced = False
            for midx, m in enumerate(scene.models):
                name = str(getattr(m, "name", "") or "")
                mode: str | None = None
                if name.endswith("-plat-t"):
                    mode = "t"
                elif name.endswith("-plat-f"):
                    mode = "f"
                elif plat_suffix and name.endswith(plat_suffix):
                    mode = "f" if plat_suffix.endswith("plat-f") else "t"

                if mode is None:
                    continue

                qpm[int(midx)] = _build_plat_cutout_quads_for_model(scene, int(midx), mode=mode)
                replaced = True

            if replaced:
                mesher_result = MesherResult(quads_per_model=qpm)
                any_cutout = True

    cull_letters = str(getattr(opts, "cull", "") or "").strip().lower()
    if cull_letters:
        allowed = set("tblrfk")
        bad = sorted({c for c in cull_letters if c not in allowed})
        if bad:
            raise ValueError(f"--cull: invalid letters: {''.join(bad)} (allowed: tblrfk)")

        cull_faces: set[tuple[int, int]] = set()
        if "t" in cull_letters:
            cull_faces.add((2, 1))
        if "b" in cull_letters:
            cull_faces.add((2, -1))
        if "l" in cull_letters:
            cull_faces.add((0, -1))
        if "r" in cull_letters:
            cull_faces.add((0, 1))
        if "f" in cull_letters:
            cull_faces.add((1, 1))
        if "k" in cull_letters:
            cull_faces.add((1, -1))

        mesher_result = MesherResult(
            quads_per_model=[
                [
                    q
                    for q in quads
                    if (int(q.axis), int(q.normal_sign)) not in cull_faces
                ]
                for quads in mesher_result.quads_per_model
            ]
        )
    mark("build_quads")

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
        style=("baked" if bool(opts.plat_cutout) or bool(any_cutout) else str(opts.atlas.style)),
        alpha=("rgba" if bool(opts.plat_cutout) or bool(any_cutout) else str(opts.texture_alpha)),
    )
    mark("build_atlas")

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
        pivot=str(opts.pivot),
        uv2_mode=str(getattr(opts, "uv2_mode", "copy")),
        export_uv2=bool(getattr(opts, "export_uv2", False)),
        export_vertex_color=bool(getattr(opts, "export_vertex_color", False)),
    )
    mark("build_model_meshes")

    character_flat = bool(getattr(opts, "character_flat", False))
    character_apart = bool(getattr(opts, "character_apart", False)) or character_flat
    no_merge_nodes = bool(getattr(opts, "no_merge_nodes", False)) or character_apart

    if no_merge_nodes:
        meshes, model_to_mesh = _build_model_meshes(
            scene,
            mesher_result,
            atlas_result,
            scale=float(opts.scale),
            flip_v=bool(opts.flip_v),
            pivot=str(opts.pivot),
            export_uv2=bool(getattr(opts, "export_uv2", False)),
            uv2_mode=str(getattr(opts, "uv2_mode", "copy")),
            export_vertex_color=bool(getattr(opts, "export_vertex_color", False)),
        )
        nodes, root_node_ids, meshes = _build_scene_nodes_no_extra(scene, model_to_mesh, meshes, scale=float(opts.scale))
        if character_apart:
            nodes, root_node_ids = _collapse_scene_nodes_inplace(scene, nodes, root_node_ids, model_to_mesh)
            if character_flat:
                nodes, root_node_ids = _flatten_character_roots(nodes, root_node_ids)
    else:
        nodes, root_node_ids = _build_scene_nodes_two_level(scene, {m["model_id"]: i for i, m in enumerate(meshes)}, scale=float(opts.scale))
    mark("build_scene_nodes")

    nodes, root_node_ids = _collapse_auto_wrappers(nodes, root_node_ids)

    nodes, root_node_ids = _collapse_same_name_single_child(nodes, root_node_ids)

    nodes = _ensure_mesh_node_names(nodes, meshes)

    if debug_transforms_out:
        _write_transform_debug(debug_transforms_out, scene=scene, meshes=meshes, stage="pre_axis")

    meshes = _to_y_up_left_handed(meshes)
    nodes = _to_y_up_left_handed_nodes(nodes, meshes)
    mark("axis_convert")

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
            alpha_mode=("MASK" if bool(any_cutout) else None),
            alpha_cutoff=(float(opts.plat_cutoff) if bool(any_cutout) else None),
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
            alpha_mode=("MASK" if bool(any_cutout) else None),
            alpha_cutoff=(float(opts.plat_cutoff) if bool(any_cutout) else None),
        )

    mark(f"write_{str(output_format)}")

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
        mark("write_uv_json")


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


def _ensure_mesh_node_names(nodes: list[dict], meshes: list[dict]) -> list[dict]:
    def is_auto_name(n: str) -> bool:
        ns = str(n or "")
        return ns.startswith("trn_") or ns.startswith("grp_") or ns.startswith("shp_") or ns.startswith("node_")

    out: list[dict] = []
    for i, nd in enumerate(nodes):
        mm = dict(nd)
        mesh_id = mm.get("mesh")
        if mesh_id is not None:
            cur = str(mm.get("name") or "")
            if (not cur) or is_auto_name(cur):
                try:
                    mi = int(mesh_id)
                except Exception:
                    mi = -1
                mesh_name = None
                if 0 <= mi < len(meshes):
                    mesh_name = meshes[mi].get("name")
                mm["name"] = str(mesh_name) if mesh_name else f"mesh_{mi if mi >= 0 else i}"
        out.append(mm)
    return out


def _collapse_auto_wrappers(nodes: list[dict], root_node_ids: list[int], *, aggressive: bool = False) -> tuple[list[dict], list[int]]:
    def is_auto_name(n: str) -> bool:
        ns = str(n or "")
        return ns.startswith("trn_") or ns.startswith("grp_") or ns.startswith("shp_") or ns.startswith("node_")

    def is_identity_tr(nd: dict) -> bool:
        t = nd.get("translation")
        r = nd.get("rotation")
        if t is None and r is None:
            return True
        if t is None:
            t = (0.0, 0.0, 0.0)
        if r is None:
            r = (0.0, 0.0, 0.0, 1.0)
        return (
            float(t[0]) == 0.0
            and float(t[1]) == 0.0
            and float(t[2]) == 0.0
            and float(r[0]) == 0.0
            and float(r[1]) == 0.0
            and float(r[2]) == 0.0
            and float(r[3]) == 1.0
        )

    def push_tr_to_child(*, parent: dict, child: dict) -> None:
        pt = parent.get("translation") or (0.0, 0.0, 0.0)
        pr = parent.get("rotation") or (0.0, 0.0, 0.0, 1.0)
        ct = child.get("translation") or (0.0, 0.0, 0.0)
        cr = child.get("rotation") or (0.0, 0.0, 0.0, 1.0)

        rt = _quat_rotate_vec(tuple(float(x) for x in pr), (float(ct[0]), float(ct[1]), float(ct[2])))
        nt = (float(pt[0]) + float(rt[0]), float(pt[1]) + float(rt[1]), float(pt[2]) + float(rt[2]))
        nr = _quat_norm(_quat_mul(tuple(float(x) for x in pr), tuple(float(x) for x in cr)))
        child["translation"] = nt
        child["rotation"] = nr

    # Iteratively bypass wrapper nodes by lifting their children to parents/roots.
    changed = True
    while changed:
        changed = False

        parents: list[list[int]] = [[] for _ in range(len(nodes))]
        for pid, p in enumerate(nodes):
            for ch in list(p.get("children") or []):
                ci = int(ch)
                if 0 <= ci < len(nodes):
                    parents[ci].append(int(pid))

        # Update roots first.
        new_roots: list[int] = []
        for rid in list(root_node_ids):
            r = int(rid)
            if not (0 <= r < len(nodes)):
                continue
            nd = nodes[r]
            if nd.get("mesh") is None and is_auto_name(str(nd.get("name") or "")):
                can_collapse = bool(aggressive) or is_identity_tr(nd)
                if can_collapse:
                    chs = [int(c) for c in list(nd.get("children") or [])]
                    if aggressive and (not is_identity_tr(nd)):
                        for cid in chs:
                            if 0 <= cid < len(nodes):
                                push_tr_to_child(parent=nd, child=nodes[cid])
                    new_roots.extend(chs)
                    nd["children"] = []
                    nd.pop("translation", None)
                    nd.pop("rotation", None)
                    changed = True
                    continue
            new_roots.append(r)

        # Deduplicate roots preserving order.
        seen_root: set[int] = set()
        root_node_ids = []
        for r in new_roots:
            if r in seen_root:
                continue
            seen_root.add(r)
            root_node_ids.append(int(r))

        # Collapse internal wrappers.
        for nid, nd in enumerate(nodes):
            if nd.get("mesh") is not None:
                continue
            if not is_auto_name(str(nd.get("name") or "")):
                continue
            if (not aggressive) and (not is_identity_tr(nd)):
                continue
            chs = [int(c) for c in list(nd.get("children") or [])]
            if not chs:
                continue

            if aggressive and (not is_identity_tr(nd)):
                for cid in chs:
                    if 0 <= cid < len(nodes):
                        push_tr_to_child(parent=nd, child=nodes[cid])

            # Replace nid with its children in each parent.
            for pid in parents[nid]:
                pch = [int(c) for c in list(nodes[pid].get("children") or [])]
                replaced: list[int] = []
                for c in pch:
                    if int(c) == int(nid):
                        replaced.extend(chs)
                    else:
                        replaced.append(int(c))
                # Dedup within the parent's children.
                seen: set[int] = set()
                deduped: list[int] = []
                for c in replaced:
                    if c in seen:
                        continue
                    seen.add(c)
                    deduped.append(int(c))
                nodes[pid]["children"] = deduped

            nd["children"] = []
            nd.pop("translation", None)
            nd.pop("rotation", None)
            changed = True

    return nodes, root_node_ids


def _collapse_same_name_single_child(nodes: list[dict], root_node_ids: list[int]) -> tuple[list[dict], list[int]]:
    def is_auto_name(n: str) -> bool:
        ns = str(n or "")
        return ns.startswith("trn_") or ns.startswith("grp_") or ns.startswith("shp_") or ns.startswith("node_")

    def is_identity_tr(nd: dict) -> bool:
        t = nd.get("translation")
        r = nd.get("rotation")
        tt = (0.0, 0.0, 0.0) if t is None else tuple(float(x) for x in t)
        rr = (0.0, 0.0, 0.0, 1.0) if r is None else tuple(float(x) for x in r)
        return tt == (0.0, 0.0, 0.0) and rr == (0.0, 0.0, 0.0, 1.0)

    def push_tr_to_child(*, parent: dict, child: dict) -> None:
        pt = parent.get("translation") or (0.0, 0.0, 0.0)
        pr = parent.get("rotation") or (0.0, 0.0, 0.0, 1.0)
        ct = child.get("translation") or (0.0, 0.0, 0.0)
        cr = child.get("rotation") or (0.0, 0.0, 0.0, 1.0)

        rt = _quat_rotate_vec(tuple(float(x) for x in pr), (float(ct[0]), float(ct[1]), float(ct[2])))
        nt = (float(pt[0]) + float(rt[0]), float(pt[1]) + float(rt[1]), float(pt[2]) + float(rt[2]))
        nr = _quat_norm(_quat_mul(tuple(float(x) for x in pr), tuple(float(x) for x in cr)))
        child["translation"] = nt
        child["rotation"] = nr

    changed = True
    while changed:
        changed = False

        # Build parent pointers.
        parents: dict[int, set[int]] = {}
        for pid, pnd in enumerate(nodes):
            for cid in list(pnd.get("children") or []):
                try:
                    cii = int(cid)
                except Exception:
                    continue
                if 0 <= cii < len(nodes):
                    parents.setdefault(cii, set()).add(pid)

        # Collapse any parent node that only wraps a single child and duplicates the name.
        for pid, pnd in enumerate(nodes):
            if pnd.get("mesh") is not None:
                continue

            chs = [int(c) for c in list(pnd.get("children") or []) if isinstance(c, (int, str))]
            if len(chs) != 1:
                continue
            cid = int(chs[0])
            if not (0 <= cid < len(nodes)):
                continue

            cnd = nodes[cid]

            pn = str(pnd.get("name") or "")
            cn = str(cnd.get("name") or "")
            if not cn:
                continue

            # Cases:
            # 1) exact same meaningful name (tent1 -> tent1)
            # 2) parent is auto/empty but child has a meaningful name
            same_name = bool(pn) and (pn == cn)
            parent_unhelpful = (not pn) or is_auto_name(pn)
            child_meaningful = not is_auto_name(cn)
            if (not same_name) and (not (parent_unhelpful and child_meaningful)):
                continue

            if not is_identity_tr(pnd):
                push_tr_to_child(parent=pnd, child=cnd)

            # Rewire roots
            root_node_ids = [cid if int(rid) == pid else int(rid) for rid in root_node_ids]

            # Rewire all parents
            for gpid in list(parents.get(pid, set())):
                gp = nodes[gpid]
                new_children: list[int] = []
                for x in list(gp.get("children") or []):
                    try:
                        xi = int(x)
                    except Exception:
                        continue
                    if xi == pid:
                        new_children.append(cid)
                    else:
                        new_children.append(xi)
                gp["children"] = new_children

            pnd["children"] = []
            pnd.pop("translation", None)
            pnd.pop("rotation", None)
            changed = True

    return nodes, root_node_ids


def _apply_pivot_compensation(nodes: list[dict], meshes: list[dict]) -> list[dict]:
    """Compensate node translations for pivot changes to maintain world position."""
    out: list[dict] = []
    for nd in nodes:
        mm = dict(nd)
        mesh_id = mm.get("mesh")
        if mesh_id is None:
            out.append(mm)
            continue

        try:
            mi = int(mesh_id)
        except Exception:
            mi = -1
        if not (0 <= mi < len(meshes)):
            out.append(mm)
            continue

        pivot_offset = meshes[mi].get("pivot_offset")
        if pivot_offset is None:
            out.append(mm)
            continue

        tr = mm.get("translation")
        if tr is None:
            tr = (0.0, 0.0, 0.0)

        rot = mm.get("rotation") or (0.0, 0.0, 0.0, 1.0)
        dp = _quat_rotate_vec(tuple(float(x) for x in rot), (float(pivot_offset[0]), float(pivot_offset[1]), float(pivot_offset[2])))
        mm["translation"] = (float(tr[0]) + float(dp[0]), float(tr[1]) + float(dp[1]), float(tr[2]) + float(dp[2]))
        out.append(mm)
    return out


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


def _build_plat_cutout_quads(scene: VoxScene) -> MesherResult:
    per_model: list[list[Quad]] = []

    for midx, model in enumerate(scene.models):
        vox = np.asarray(model.voxels)
        if vox.size == 0:
            per_model.append([])
            continue

        sx, sy, sz = (int(model.size[0]), int(model.size[1]), int(model.size[2]))
        if sx <= 0 or sy <= 0 or sz <= 0:
            per_model.append([])
            continue

        filled = vox != 0
        if not bool(np.any(filled)):
            per_model.append([])
            continue

        # For each (x,y), pick the highest z with a voxel.
        # filled_xyz: (sx,sy,sz)
        any_xy = np.any(filled, axis=2)  # (sx,sy)
        z_indices = np.arange(sz, dtype=np.int32)
        z_masked = np.where(filled, z_indices[None, None, :], -1)
        top_z = np.max(z_masked, axis=2).astype(np.int32)  # (sx,sy)

        # Build top-down color plate. colors is (V, U) == (sy, sx)
        xs = np.arange(sx, dtype=np.int32)[:, None]
        ys = np.arange(sy, dtype=np.int32)[None, :]
        top_colors_xy = np.where(any_xy, vox[xs, ys, top_z], 0).astype(np.int32)  # (sx,sy)
        colors_vu = top_colors_xy.T.copy()

        # Place quad at top surface plane: z = (max top_z) + 1
        z_plane = int(np.max(top_z)) + 1
        origin = (0, 0, z_plane)
        size_u = sx
        size_v = sy

        # axis=2 (Z), normal_sign=+1
        verts = (
            (0.0, 0.0, float(z_plane)),
            (float(size_u), 0.0, float(z_plane)),
            (float(size_u), float(size_v), float(z_plane)),
            (0.0, float(size_v), float(z_plane)),
        )
        per_model.append(
            [
                Quad(
                    model_index=int(midx),
                    origin=origin,
                    size_u=int(size_u),
                    size_v=int(size_v),
                    axis=2,
                    normal_sign=1,
                    colors=colors_vu,
                    verts=verts,
                    normal=(0.0, 0.0, 1.0),
                )
            ]
        )

    return MesherResult(quads_per_model=per_model)


def _build_plat_cutout_quads_for_model(scene: VoxScene, midx: int, *, mode: str = "t") -> list[Quad]:
    model = scene.models[midx]
    vox = np.asarray(model.voxels)
    if vox.size == 0:
        return []

    sx, sy, sz = (int(model.size[0]), int(model.size[1]), int(model.size[2]))
    if sx <= 0 or sy <= 0 or sz <= 0:
        return []

    filled = vox != 0
    if not bool(np.any(filled)):
        return []

    if str(mode) == "f":
        # Front (+Y): project along +Y, build a quad on the Y=max+1 plane spanning XZ.
        any_xz = np.any(filled, axis=1)  # (sx,sz)
        y_indices = np.arange(sy, dtype=np.int32)
        y_masked = np.where(filled, y_indices[None, :, None], -1)
        front_y = np.max(y_masked, axis=1).astype(np.int32)  # (sx,sz)

        xs = np.arange(sx, dtype=np.int32)[:, None]
        zs = np.arange(sz, dtype=np.int32)[None, :]
        front_colors_xz = np.where(any_xz, vox[xs, front_y, zs], 0).astype(np.int32)  # (sx,sz)
        colors_vu = front_colors_xz.T.copy()  # (sz,sx)

        y_plane = int(np.max(front_y)) + 1
        origin = (0, y_plane, 0)
        size_u = sx
        size_v = sz
        verts = (
            (0.0, float(y_plane), 0.0),
            (float(size_u), float(y_plane), 0.0),
            (float(size_u), float(y_plane), float(size_v)),
            (0.0, float(y_plane), float(size_v)),
        )
        axis = 1
        normal_sign = 1
        normal = (0.0, 1.0, 0.0)
    else:
        # Top (+Z): project along +Z, build a quad on the Z=max+1 plane spanning XY.
        any_xy = np.any(filled, axis=2)  # (sx,sy)
        z_indices = np.arange(sz, dtype=np.int32)
        z_masked = np.where(filled, z_indices[None, None, :], -1)
        top_z = np.max(z_masked, axis=2).astype(np.int32)  # (sx,sy)

        xs = np.arange(sx, dtype=np.int32)[:, None]
        ys = np.arange(sy, dtype=np.int32)[None, :]
        top_colors_xy = np.where(any_xy, vox[xs, ys, top_z], 0).astype(np.int32)  # (sx,sy)
        colors_vu = top_colors_xy.T.copy()  # (sy,sx)

        z_plane = int(np.max(top_z)) + 1
        origin = (0, 0, z_plane)
        size_u = sx
        size_v = sy
        verts = (
            (0.0, 0.0, float(z_plane)),
            (float(size_u), 0.0, float(z_plane)),
            (float(size_u), float(size_v), float(z_plane)),
            (0.0, float(size_v), float(z_plane)),
        )
        axis = 2
        normal_sign = 1
        normal = (0.0, 0.0, 1.0)

    return [
        Quad(
            model_index=int(midx),
            origin=origin,
            size_u=int(size_u),
            size_v=int(size_v),
            axis=int(axis),
            normal_sign=int(normal_sign),
            colors=colors_vu,
            verts=verts,
            normal=normal,
        )
    ]


def _assemble_meshes(
    scene: VoxScene,
    mesher_result: MesherResult,
    atlas_result: atlas_mod.AtlasBuildResult,
    *,
    scale: float,
    flip_v: bool,
    pivot: str = "corner",
    export_uv2: bool = False,
    uv2_mode: str = "copy",
    export_vertex_color: bool = False,
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
        
        requested_pivot = pivot
        pivot_for_geom = requested_pivot
        is_plat_t = _is_plat_t_model(scene, midx)
        if is_plat_t:
            pivot_for_geom = "top_center"

        if pivot_for_geom == "corner":
            p = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
        else:
            # Use actual geometry bounds (not the model grid size) so the pivot is visually centered.
            bmin = pos_arr.min(axis=0)
            bmax = pos_arr.max(axis=0)
            cx = float(bmin[0] + bmax[0]) * 0.5
            cy = float(bmin[1] + bmax[1]) * 0.5
            
            if pivot_for_geom == "top_center":
                cz = float(bmax[2])  # Pivot at top (max Z)
            else:  # center or bottom_center
                cz = float(bmin[2] + bmax[2]) * 0.5  # Always use center for Z
            
            p = np.asarray((cx, cy, cz), dtype=np.float32)

        # For bottom_center: calculate half_height and move vertices up
        half_height_for_node = None
        if pos_arr.size:
            pos_arr = pos_arr.copy()
            pos_arr[:, 0] -= float(p[0])
            pos_arr[:, 1] -= float(p[1])
            pos_arr[:, 2] -= float(p[2])
            
            if requested_pivot == "bottom_center":
                bmin_centered = pos_arr.min(axis=0)
                bmax_centered = pos_arr.max(axis=0)
                height = float(bmax_centered[2] - bmin_centered[2])
                half_height_for_node = height * 0.5
                # Move vertices up by half-height (pivot at bottom)
                pos_arr[:, 2] += half_height_for_node

                if is_plat_t:
                    extra_plat_offset = _compute_plat_base_half_height(scene, midx, scale)
                    if float(extra_plat_offset):
                        pos_arr[:, 2] += float(extra_plat_offset)

        texcoords_arr = np.asarray(texcoords, dtype=np.float32)
        texcoords1_arr = None
        if bool(export_uv2):
            if str(uv2_mode) == "lightmap":
                quad_count = int(texcoords_arr.shape[0] // 4)
                grid = int(math.ceil(math.sqrt(float(max(1, quad_count)))))
                pad = 0.05
                cell = 1.0 / float(grid)
                inner = cell * (1.0 - 2.0 * pad)
                uv1 = np.empty_like(texcoords_arr)
                for qi in range(quad_count):
                    gx = int(qi % grid)
                    gy = int(qi // grid)
                    x0 = float(gx) * cell + cell * pad
                    y0 = float(gy) * cell + cell * pad
                    x1 = x0 + inner
                    y1 = y0 + inner
                    base = qi * 4
                    uv1[base + 0] = (x0, y0)
                    uv1[base + 1] = (x1, y0)
                    uv1[base + 2] = (x1, y1)
                    uv1[base + 3] = (x0, y1)
                texcoords1_arr = uv1
            else:
                texcoords1_arr = texcoords_arr.copy()

        color0_arr = None
        if bool(export_vertex_color):
            n = int(pos_arr.shape[0])
            color0_arr = np.ones((n, 4), dtype=np.float32)

        geom = {
            "positions": pos_arr,
            "normals": np.asarray(normals, dtype=np.float32),
            "texcoords": texcoords_arr,
            "texcoords1": texcoords1_arr,
            "color0": color0_arr,
            "indices": np.asarray(indices, dtype=np.uint32),
            "pivot_p": (float(p[0]), float(p[1]), float(p[2])),
        }
        if half_height_for_node is not None:
            geom["half_height"] = half_height_for_node
        model_geom_cache[midx] = geom
        return geom

    def merge_geoms(geoms: list[tuple[int, dict]]) -> dict | None:
        if not geoms:
            return None
        pos_list: list[np.ndarray] = []
        nrm_list: list[np.ndarray] = []
        uv_list: list[np.ndarray] = []
        uv1_list: list[np.ndarray] = []
        any_uv1 = False
        col_list: list[np.ndarray] = []
        any_col = False
        idx_list: list[np.ndarray] = []
        base = 0
        for _midx, g in geoms:
            pos = np.asarray(g["positions"], dtype=np.float32)
            nrm = np.asarray(g["normals"], dtype=np.float32)
            uv = np.asarray(g["texcoords"], dtype=np.float32)
            uv1 = g.get("texcoords1")
            if uv1 is not None:
                any_uv1 = True
                uv1_list.append(np.asarray(uv1, dtype=np.float32))
            col = g.get("color0")
            if col is not None:
                any_col = True
                col_list.append(np.asarray(col, dtype=np.float32))
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
            "texcoords1": (np.vstack(uv1_list) if any_uv1 else None),
            "color0": (np.vstack(col_list) if any_col else None),
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
                    "texcoords1": merged["texcoords1"],
                    "color0": merged.get("color0"),
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
    export_uv2: bool = False,
    uv2_mode: str = "copy",
    export_vertex_color: bool = False,
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
        
        requested_pivot = pivot
        pivot_for_geom = requested_pivot
        is_plat_t = _is_plat_t_model(scene, midx)
        if is_plat_t:
            pivot_for_geom = "top_center"

        if pivot_for_geom == "corner":
            p = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
        else:
            # Use actual geometry bounds (not the model grid size) so the pivot is visually centered.
            bmin = pos_arr.min(axis=0)
            bmax = pos_arr.max(axis=0)
            cx = float(bmin[0] + bmax[0]) * 0.5
            cy = float(bmin[1] + bmax[1]) * 0.5
            
            if pivot_for_geom == "top_center":
                cz = float(bmax[2])  # Pivot at top (max Z)
            else:  # center or bottom_center
                cz = float(bmin[2] + bmax[2]) * 0.5  # Always use center for Z
            
            p = np.asarray((cx, cy, cz), dtype=np.float32)

        # For bottom_center: calculate half_height and move vertices up
        half_height_for_node = None
        if pos_arr.size:
            pos_arr = pos_arr.copy()
            pos_arr[:, 0] -= float(p[0])
            pos_arr[:, 1] -= float(p[1])
            pos_arr[:, 2] -= float(p[2])
            
            if requested_pivot == "bottom_center":
                bmin_centered = pos_arr.min(axis=0)
                bmax_centered = pos_arr.max(axis=0)
                height = float(bmax_centered[2] - bmin_centered[2])
                half_height_for_node = height * 0.5
                # Move vertices up by half-height (pivot at bottom)
                pos_arr[:, 2] += half_height_for_node
                if is_plat_t:
                    extra_plat_offset = _compute_plat_base_half_height(scene, midx, scale)
                    if float(extra_plat_offset):
                        pos_arr[:, 2] += float(extra_plat_offset)

        mesh_index = len(meshes)

        texcoords_arr = np.asarray(texcoords, dtype=np.float32)
        texcoords1_arr = None
        if bool(export_uv2):
            if str(uv2_mode) == "lightmap":
                quad_count = int(pos_arr.shape[0] // 4)
                if quad_count > 0:
                    cols = int(math.ceil(math.sqrt(float(quad_count))))
                    rows = int(math.ceil(float(quad_count) / float(cols)))
                    cell = 1.0 / float(max(cols, rows))
                    inner = cell * 0.9
                    pad = (cell - inner) * 0.5
                    uv1 = np.zeros((pos_arr.shape[0], 2), dtype=np.float32)
                    for qi in range(quad_count):
                        cx = int(qi % cols)
                        cy = int(qi // cols)
                        x0 = cx * cell + pad
                        y0 = cy * cell + pad
                        x1 = x0 + inner
                        y1 = y0 + inner
                        base = qi * 4
                        uv1[base + 0] = (x0, y0)
                        uv1[base + 1] = (x1, y0)
                        uv1[base + 2] = (x1, y1)
                        uv1[base + 3] = (x0, y1)
                    texcoords1_arr = uv1
            else:
                texcoords1_arr = texcoords_arr.copy()

        color0_arr = None
        if bool(export_vertex_color):
            n = int(pos_arr.shape[0])
            color0_arr = np.ones((n, 4), dtype=np.float32)

        mesh_dict = {
            "name": scene.models[midx].name,
            "model_id": int(midx),
            "positions": pos_arr,
            "normals": np.asarray(normals, dtype=np.float32),
            "texcoords": texcoords_arr,
            "texcoords1": texcoords1_arr,
            "color0": color0_arr,
            "indices": np.asarray(indices, dtype=np.uint32),
        }
        if half_height_for_node is not None:
            mesh_dict["half_height"] = half_height_for_node
        meshes.append(mesh_dict)
        model_to_mesh[int(midx)] = int(mesh_index)

    return meshes, model_to_mesh


def _build_scene_nodes(scene: VoxScene, model_to_mesh: dict[int, int], *, scale: float = 1.0) -> tuple[list[dict], list[int]]:
    nodes: list[dict] = []
    # Pre-create nodes matching VOX node indices.
    for nd in scene.nodes:
        out: dict = {"name": nd.name, "children": list(nd.children)}
        if nd.kind == "trn":
            t = tuple(float(x) for x in nd.translation)
            out["translation"] = (t[0] * float(scale), t[1] * float(scale), t[2] * float(scale))
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


def _build_scene_nodes_no_extra(
    scene: VoxScene,
    model_to_mesh: dict[int, int],
    meshes: list[dict],
    *,
    scale: float = 1.0,
) -> tuple[list[dict], list[int], list[dict]]:
    # Preserve VOX scene graph 1:1 (nTRN/nGRP/nSHP) with no extra nodes.
    # If an nSHP references multiple models, merge those model meshes into a single mesh
    # and attach it directly to that nSHP node.

    nodes: list[dict] = []
    for nd in scene.nodes:
        out: dict = {"name": nd.name, "children": list(nd.children)}
        if nd.kind == "trn":
            t = tuple(float(x) for x in nd.translation)
            out["translation"] = (t[0] * float(scale), t[1] * float(scale), t[2] * float(scale))
            out["rotation"] = tuple(float(x) for x in nd.rotation)
        nodes.append(out)

    def merge_meshes(mesh_ids: list[int]) -> int:
        pos_list: list[np.ndarray] = []
        nrm_list: list[np.ndarray] = []
        uv_list: list[np.ndarray] = []
        uv1_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        any_uv1 = False
        any_col = False
        idx_list: list[np.ndarray] = []
        base = 0

        for mid in mesh_ids:
            m = meshes[int(mid)]
            pos = np.asarray(m["positions"], dtype=np.float32)
            nrm = np.asarray(m["normals"], dtype=np.float32)
            uv = np.asarray(m["texcoords"], dtype=np.float32)
            pos_list.append(pos)
            nrm_list.append(nrm)
            uv_list.append(uv)

            uv1 = m.get("texcoords1")
            if uv1 is not None:
                any_uv1 = True
                uv1_list.append(np.asarray(uv1, dtype=np.float32))

            col = m.get("color0")
            if col is not None:
                any_col = True
                col_list.append(np.asarray(col, dtype=np.float32))

            idx = np.asarray(m["indices"], dtype=np.uint32)
            idx_list.append(idx + np.uint32(base))
            base += int(pos.shape[0])

        merged = {
            "name": "merged",
            "model_id": -1,
            "positions": np.vstack(pos_list) if pos_list else np.zeros((0, 3), dtype=np.float32),
            "normals": np.vstack(nrm_list) if nrm_list else np.zeros((0, 3), dtype=np.float32),
            "texcoords": np.vstack(uv_list) if uv_list else np.zeros((0, 2), dtype=np.float32),
            "texcoords1": (np.vstack(uv1_list) if any_uv1 else None),
            "color0": (np.vstack(col_list) if any_col else None),
            "indices": np.concatenate(idx_list) if idx_list else np.zeros((0,), dtype=np.uint32),
        }
        new_id = len(meshes)
        meshes.append(merged)
        return int(new_id)

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
            nodes[nid]["mesh"] = int(merge_meshes(mesh_ids))

    return nodes, [int(r) for r in scene.root_node_ids], meshes


def _collapse_scene_nodes_inplace(
    scene: VoxScene,
    nodes: list[dict],
    root_node_ids: list[int],
    model_to_mesh: dict[int, int],
) -> tuple[list[dict], list[int]]:
    def is_auto_name(n: str) -> bool:
        ns = str(n or "")
        return ns.startswith("trn_") or ns.startswith("grp_") or ns.startswith("shp_") or ns.startswith("node_")

    # Rename auto-named shape nodes that carry a mesh to a meaningful model name.
    for nid, nd in enumerate(scene.nodes):
        if nd.kind != "shp":
            continue
        if nid < 0 or nid >= len(nodes):
            continue
        out = nodes[nid]
        if "mesh" not in out:
            continue
        if not is_auto_name(str(out.get("name") or "")):
            continue
        model_ids = [int(x) for x in nd.model_ids]
        if len(model_ids) == 1 and 0 <= model_ids[0] < len(scene.models):
            out["name"] = str(scene.models[model_ids[0]].name)

    # Collapse wrapper trn/grp nodes that are auto-named and have a single child.
    # We never create new nodes; we only redirect parent links.
    changed = True
    while changed:
        changed = False

        # Build parent map based on current graph.
        parents: list[list[int]] = [[] for _ in range(len(nodes))]
        for pid, p in enumerate(nodes):
            for ch in list(p.get("children") or []):
                ci = int(ch)
                if 0 <= ci < len(nodes):
                    parents[ci].append(int(pid))

        for nid, nd in enumerate(scene.nodes):
            if nid < 0 or nid >= len(nodes):
                continue
            if nd.kind not in ("trn", "grp"):
                continue
            out = nodes[nid]
            if out.get("mesh") is not None:
                continue
            if not is_auto_name(str(out.get("name") or "")):
                continue
            children = list(out.get("children") or [])
            if len(children) != 1:
                continue
            child_id = int(children[0])
            if not (0 <= child_id < len(nodes)):
                continue

            if nd.kind == "trn":
                t0 = out.get("translation") or (0.0, 0.0, 0.0)
                r0 = out.get("rotation") or (0.0, 0.0, 0.0, 1.0)
                ct0 = nodes[child_id].get("translation") or (0.0, 0.0, 0.0)
                cr0 = nodes[child_id].get("rotation") or (0.0, 0.0, 0.0, 1.0)
                rt = _quat_rotate_vec(tuple(float(x) for x in r0), tuple(float(x) for x in ct0))
                nt = (float(t0[0]) + float(rt[0]), float(t0[1]) + float(rt[1]), float(t0[2]) + float(rt[2]))
                nr = _quat_norm(_quat_mul(tuple(float(x) for x in r0), tuple(float(x) for x in cr0)))
                nodes[child_id]["translation"] = nt
                nodes[child_id]["rotation"] = nr

            # Redirect all parents (and roots) from nid to child_id.
            for pid in parents[nid]:
                pch = list(nodes[pid].get("children") or [])
                nodes[pid]["children"] = [child_id if int(c) == int(nid) else int(c) for c in pch]

            root_node_ids = [child_id if int(r) == int(nid) else int(r) for r in root_node_ids]

            # Detach node.
            out["children"] = []
            out.pop("translation", None)
            out.pop("rotation", None)
            changed = True

    # Deduplicate root ids while preserving order.
    seen: set[int] = set()
    new_roots: list[int] = []
    for r in root_node_ids:
        ri = int(r)
        if ri in seen:
            continue
        seen.add(ri)
        new_roots.append(ri)

    return nodes, new_roots


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

    # Build a parent link table so we can recover user names often stored on nTRN wrappers.
    parent_of: list[int | None] = [None] * len(scene.nodes)
    for pid, pnd in enumerate(scene.nodes):
        for ch in pnd.children:
            ci = int(ch)
            if 0 <= ci < len(parent_of) and parent_of[ci] is None:
                parent_of[ci] = int(pid)

    def is_auto_name(n: str) -> bool:
        ns = str(n)
        return (
            ns.startswith("shp_")
            or ns.startswith("trn_")
            or ns.startswith("grp_")
            or ns.startswith("node_")
        )

    def pick_parent_name(shp_id: int) -> str:
        nd = scene.nodes[shp_id]
        base = str(nd.name)
        if not is_auto_name(base):
            return base
        cur = parent_of[shp_id]
        # Prefer nearest TRN ancestor with a custom name.
        while cur is not None and 0 <= int(cur) < len(scene.nodes):
            anc = scene.nodes[int(cur)]
            if anc.kind == "trn":
                an = str(anc.name)
                if not is_auto_name(an):
                    return an
            cur = parent_of[int(cur)]
        return base

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

        parent_name = pick_parent_name(int(nid))

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


def _to_y_up_left_handed_nodes(nodes: list[dict], meshes: list[dict]) -> list[dict]:
    # Helper function to find half_height in children
    def find_half_height_in_children(node_idx: int, nodes: list[dict], meshes: list[dict]) -> float | None:
        """Recursively search for half_height in child nodes"""
        node = nodes[node_idx]
        mesh_id = node.get("mesh")
        
        # Check current node
        if mesh_id is not None:
            try:
                mi = int(mesh_id)
                if 0 <= mi < len(meshes):
                    half_height = meshes[mi].get("half_height")
                    if half_height is not None:
                        return float(half_height)
            except:
                pass
        
        # Check children
        child_ids = node.get("children", [])
        for child_id in child_ids:
            child_half_height = find_half_height_in_children(child_id, nodes, meshes)
            if child_half_height is not None:
                return child_half_height
        
        return None
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
        
        # For bottom_center: move node down by half_height to compensate for vertex movement
        # Vertices moved up -> node moves down -> visual position stays the same
        if tr is not None:
            # First check if this node has a mesh
            mesh_id = mm.get("mesh")
            half_height = None
            
            if mesh_id is not None:
                try:
                    mi = int(mesh_id)
                    if 0 <= mi < len(meshes):
                        half_height = meshes[mi].get("half_height")
                except:
                    pass
            
            # If no half_height found, search in children
            if half_height is None:
                node_idx = len(out)  # Current node index
                half_height = find_half_height_in_children(node_idx, nodes, meshes)
            
            # Apply adjustment if half_height found
            if half_height is not None:
                tx, ty, tz = mm["translation"]
                mm["translation"] = (float(tx), float(ty) - float(half_height), float(tz))

        out.append(mm)

    return out


def _is_plat_t_model(scene: VoxScene, midx: int) -> bool:
    model_name = scene.models[midx].name
    if not isinstance(model_name, str):
        return False
    return model_name.endswith('-plat-t')


def _compute_plat_base_half_height(scene: VoxScene, midx: int, scale: float) -> float:
    model_name = scene.models[midx].name
    if not isinstance(model_name, str) or not model_name.endswith('-plat-t'):
        return 0.0

    base_name = model_name[:-7]
    height_vox = 0.0

    for other_model in scene.models:
        if other_model.name == base_name:
            base_vox = np.asarray(other_model.voxels)
            if base_vox.size:
                try:
                    z_coords = base_vox[:, 2]
                except Exception:
                    z_coords = np.asarray([])
                if z_coords.size:
                    height_vox = float(z_coords.max() - z_coords.min() + 1)
                    break
            size_z = float(other_model.size[2])
            if size_z > 0.0:
                height_vox = size_z
            break

    if height_vox <= 0.0:
        size_z = float(scene.models[midx].size[2])
        if size_z > 0.0:
            height_vox = size_z

    if height_vox <= 0.0:
        return 0.0

    return height_vox * float(scale) * 0.5


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


def _quat_inv(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    # For unit quaternions, inverse == conjugate.
    x, y, z, w = _quat_norm(q)
    return (-float(x), -float(y), -float(z), float(w))


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


def _flatten_character_roots(nodes: list[dict], root_node_ids: list[int]) -> tuple[list[dict], list[int]]:
    def is_auto_name(n: str) -> bool:
        ns = str(n or "")
        return ns.startswith("trn_") or ns.startswith("grp_") or ns.startswith("shp_") or ns.startswith("node_")

    # Compute world transforms for current node graph (glTF composition rules).
    world_t: list[tuple[float, float, float] | None] = [None] * len(nodes)
    world_r: list[tuple[float, float, float, float] | None] = [None] * len(nodes)

    def walk(nid: int, pt: tuple[float, float, float], pr: tuple[float, float, float, float]) -> None:
        if nid < 0 or nid >= len(nodes):
            return
        if world_t[nid] is not None:
            return
        nd = nodes[nid]
        lt = nd.get("translation") or (0.0, 0.0, 0.0)
        lr = nd.get("rotation") or (0.0, 0.0, 0.0, 1.0)
        rt = _quat_rotate_vec(tuple(float(x) for x in pr), tuple(float(x) for x in lt))
        wt = (float(pt[0]) + float(rt[0]), float(pt[1]) + float(rt[1]), float(pt[2]) + float(rt[2]))
        wr = _quat_norm(_quat_mul(tuple(float(x) for x in pr), tuple(float(x) for x in lr)))
        world_t[nid] = wt
        world_r[nid] = wr
        for ch in list(nd.get("children") or []):
            walk(int(ch), wt, wr)

    for rid in root_node_ids:
        walk(int(rid), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    # Flatten under each meaningful root node.
    for rid in list(root_node_ids):
        r = int(rid)
        if r < 0 or r >= len(nodes):
            continue
        rname = str(nodes[r].get("name") or "")
        if is_auto_name(rname):
            continue

        rt = world_t[r] or (0.0, 0.0, 0.0)
        rr = world_r[r] or (0.0, 0.0, 0.0, 1.0)
        rri = _quat_inv(rr)

        # Collect all mesh descendants.
        mesh_desc: list[int] = []
        seen: set[int] = set()

        def collect(nid: int) -> None:
            if nid in seen:
                return
            seen.add(int(nid))
            nd = nodes[nid]
            if nd.get("mesh") is not None:
                mesh_desc.append(int(nid))
            for ch in list(nd.get("children") or []):
                collect(int(ch))

        for ch in list(nodes[r].get("children") or []):
            collect(int(ch))

        if not mesh_desc:
            continue

        # Recompute each mesh node's local transform relative to the root, and detach its children.
        new_children: list[int] = []
        for mid in mesh_desc:
            mt = world_t[mid] or (0.0, 0.0, 0.0)
            mr = world_r[mid] or (0.0, 0.0, 0.0, 1.0)

            mn = str(nodes[mid].get("name") or "")
            if (not mn) or is_auto_name(mn):
                mesh_id = nodes[mid].get("mesh")
                nodes[mid]["name"] = (f"mesh_{int(mesh_id)}" if mesh_id is not None else f"node_{int(mid)}")

            dt = (float(mt[0]) - float(rt[0]), float(mt[1]) - float(rt[1]), float(mt[2]) - float(rt[2]))
            lt = _quat_rotate_vec(rri, dt)
            lr = _quat_norm(_quat_mul(rri, mr))

            nodes[mid]["translation"] = (float(lt[0]), float(lt[1]), float(lt[2]))
            nodes[mid]["rotation"] = (float(lr[0]), float(lr[1]), float(lr[2]), float(lr[3]))
            nodes[mid]["children"] = []
            new_children.append(int(mid))

        nodes[r]["children"] = new_children

    return nodes, root_node_ids


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




if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
