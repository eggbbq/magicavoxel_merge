"""Texture atlas construction for the BTP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

import numpy as np
import os
from PIL import Image
import sys
import time

from .btp_mesher import MesherResult
from .voxio import VoxScene


@dataclass(slots=True)
class QuadUV:
    u0: float
    v0: float
    u1: float
    v1: float


@dataclass(slots=True)
class AtlasBuildResult:
    width: int
    height: int
    texture_png: bytes
    quad_uvs: Dict[Tuple[int, int], QuadUV]
    model_uv_rects: Dict[int, Tuple[float, float, float, float]]


@dataclass(slots=True)
class _QuadAnchor:
    src_midx: int
    src_qidx: int
    off_u: int
    off_v: int


@dataclass(slots=True)
class _QuadTexSpec:
    colors: np.ndarray
    core_w: int
    core_h: int
    sample_scale: int


def build_atlas(
    scene: VoxScene,
    mesher_result: MesherResult,
    *,
    pad: int = 2,
    inset: float = 1.0,
    texel_scale: int = 1,
    square: bool = False,
    pot: bool = False,
    layout: str = "by-model",
    tight_blocks: bool = False,
    style: str = "baked",
    alpha: str = "auto",
    reuse_subrects: bool = True,
    compress_solid_quads: bool = False,
    face_alias_uv_remap: bool = False,
) -> AtlasBuildResult:
    """Pack quads into a simple atlas and return texture bytes + UV lookup."""

    timings_enabled = bool(os.environ.get("BTP_VOX_TIMINGS"))
    t_mark = time.perf_counter() if timings_enabled else 0.0

    def mark(stage: str) -> None:
        nonlocal t_mark
        if not timings_enabled:
            return
        now = time.perf_counter()
        sys.stderr.write(f"[btp_vox] atlas stage {stage}: +{(now - t_mark):.3f}s\n")
        t_mark = now

    if style not in ("baked", "solid"):
        raise ValueError("style must be 'baked' or 'solid'")
    if layout not in ("by-model", "global"):
        raise ValueError("layout must be 'by-model' or 'global'")
    if alpha not in ("auto", "rgba", "rgb"):
        raise ValueError("alpha must be 'auto', 'rgba', or 'rgb'")

    quads_per_model = mesher_result.quads_per_model
    texel_scale_i = int(max(1, int(texel_scale)))

    quad_specs = _build_quad_tex_specs(
        quads_per_model,
        texel_scale=texel_scale_i,
        compress_solid_quads=bool(compress_solid_quads),
    )
    mark("quad_specs")
    alias_applied = _apply_face_alias_from_model_name(
        scene,
        quads_per_model,
        quad_specs,
        remap_uv=bool(face_alias_uv_remap),
    )
    mark("face_alias")

    quad_anchors, owner_keys = _build_quad_reuse_map(
        quads_per_model,
        quad_specs=quad_specs,
        layout=str(layout),
        enable=bool(reuse_subrects) or bool(alias_applied),
    )
    mark("reuse_map")
    all_quad_count = int(sum(len(quads) for quads in quads_per_model))
    reused_count = int(max(0, all_quad_count - len(owner_keys)))
    if timings_enabled:
        sys.stderr.write(
            f"[btp_vox] atlas reuse_subrects={bool(reuse_subrects)} owners={len(owner_keys)} "
            f"reused={reused_count} total={all_quad_count} "
            f"compress_solid_quads={bool(compress_solid_quads)} "
            f"face_alias_uv_remap={bool(face_alias_uv_remap)} "
            f"face_alias_applied={int(alias_applied)}\n"
        )

    # Two layouts:
    # - by-model: each model gets its own packed block, then blocks are packed into atlas
    # - global: all quads are packed together into a single atlas (usually better fill rate)
    quad_core_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    model_block_sizes: Dict[int, Tuple[int, int]] = {}
    model_positions: Dict[int, Tuple[int, int]] = {}
    owner_block_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}

    if layout == "global":
        rects_all: List[Tuple[int, int, int]] = []
        rid_to_owner: dict[int, tuple[int, int]] = {}
        rid = 0
        for midx, qidx in owner_keys:
            spec = quad_specs[(midx, qidx)]
            tex_w = int(spec.core_w) + pad * 2
            tex_h = int(spec.core_h) + pad * 2
            rects_all.append((rid, tex_w, tex_h))
            rid_to_owner[rid] = (int(midx), int(qidx))
            rid += 1

        if timings_enabled:
            sys.stderr.write(f"[btp_vox] atlas layout=global rects={len(rects_all)} pot={bool(pot)} square={bool(square)}\n")

        # NOTE: For large scenes, MaxRects search over many POT bins becomes extremely slow.
        # Use a fast shelf pack, then optionally expand to POT/square. This trades some
        # fill efficiency for dramatically better runtime.
        fast_pack_threshold = max(1, int(os.environ.get("BTP_VOX_FAST_PACK_THRESHOLD", "128")))
        use_fast_pack = len(rects_all) >= fast_pack_threshold
        if pot or square:
            if use_fast_pack:
                atlas_w, atlas_h, positions = _pack_rects_best(rects_all, pot=bool(pot), square=bool(square))
            else:
                atlas_w, atlas_h, positions = _pack_best_bin(rects_all, square=bool(square))
        else:
            atlas_w, atlas_h, positions = _pack_rects(rects_all)

        if timings_enabled:
            sys.stderr.write(
                f"[btp_vox] atlas pack=global fast={bool(use_fast_pack)} size={int(atlas_w)}x{int(atlas_h)}\n"
            )

        for rid, (x, y) in positions.items():
            src_key = rid_to_owner[int(rid)]
            owner_block_positions[src_key] = (int(x), int(y))

        for midx, quads in enumerate(quads_per_model):
            for qidx, quad in enumerate(quads):
                anchor = quad_anchors[(midx, qidx)]
                src_key = (int(anchor.src_midx), int(anchor.src_qidx))
                src_ox, src_oy = owner_block_positions[src_key]
                src_spec = quad_specs[src_key]
                src_h, src_w = int(src_spec.colors.shape[0]), int(src_spec.colors.shape[1])
                px_per_cell_x = int(max(1, int(src_spec.core_w) // max(1, src_w)))
                px_per_cell_y = int(max(1, int(src_spec.core_h) // max(1, src_h)))
                off_x = int(anchor.off_u) * int(px_per_cell_x)
                off_y = int(anchor.off_v) * int(px_per_cell_y)
                core_x = int(src_ox + pad + off_x)
                core_y = int(src_oy + pad + off_y)
                quad_core_positions[(midx, qidx)] = (core_x, core_y)

    else:
        quad_local_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
        owners_by_model: dict[int, list[tuple[int, int]]] = {}
        for midx, qidx in owner_keys:
            owners_by_model.setdefault(int(midx), []).append((int(midx), int(qidx)))

        if timings_enabled:
            quad_count = sum(len(quads) for quads in quads_per_model)
            sys.stderr.write(
                f"[btp_vox] atlas layout=by-model rects={int(quad_count)} pot={bool(pot)} square={bool(square)} tight_blocks={bool(tight_blocks)}\n"
            )

        for midx, quads in enumerate(quads_per_model):
            rects: List[Tuple[int, int, int]] = []
            for _omidx, qidx in owners_by_model.get(int(midx), []):
                spec = quad_specs[(midx, qidx)]
                tex_w = int(spec.core_w) + pad * 2
                tex_h = int(spec.core_h) + pad * 2
                rects.append((qidx, tex_w, tex_h))

            if bool(tight_blocks):
                block_w, block_h, positions = _pack_rects(rects)
            else:
                if pot or square:
                    fast_pack_threshold = max(1, int(os.environ.get("BTP_VOX_FAST_PACK_THRESHOLD", "128")))
                    use_fast_pack = len(rects) >= fast_pack_threshold
                    if use_fast_pack:
                        block_w, block_h, positions = _pack_rects_best(rects, pot=bool(pot), square=bool(square))
                    else:
                        block_w, block_h, positions = _pack_best_bin(rects, square=bool(square))
                else:
                    block_w, block_h, positions = _pack_rects(rects)
            for qidx, pos in positions.items():
                quad_local_positions[(midx, qidx)] = pos
            model_block_sizes[midx] = (block_w, block_h)

        model_rects = [(midx, sz[0], sz[1]) for midx, sz in model_block_sizes.items()]
        if pot or square:
            fast_pack_threshold = max(1, int(os.environ.get("BTP_VOX_FAST_PACK_THRESHOLD", "128")))
            use_fast_pack = len(model_rects) >= fast_pack_threshold
            if use_fast_pack:
                atlas_w, atlas_h, model_positions = _pack_rects_best(model_rects, pot=bool(pot), square=bool(square))
            else:
                atlas_w, atlas_h, model_positions = _pack_best_bin(model_rects, square=bool(square))
        else:
            atlas_w, atlas_h, model_positions = _pack_rects(model_rects)

        for midx, quads in enumerate(quads_per_model):
            mx, my = model_positions.get(midx, (0, 0))
            for qidx, _quad in enumerate(quads):
                anchor = quad_anchors[(midx, qidx)]
                src_key = (int(anchor.src_midx), int(anchor.src_qidx))
                src_qx, src_qy = quad_local_positions[src_key]
                src_block_ox = int(mx + src_qx)
                src_block_oy = int(my + src_qy)
                owner_block_positions[src_key] = (src_block_ox, src_block_oy)

                src_spec = quad_specs[src_key]
                src_h, src_w = int(src_spec.colors.shape[0]), int(src_spec.colors.shape[1])
                px_per_cell_x = int(max(1, int(src_spec.core_w) // max(1, src_w)))
                px_per_cell_y = int(max(1, int(src_spec.core_h) // max(1, src_h)))
                off_x = int(anchor.off_u) * int(px_per_cell_x)
                off_y = int(anchor.off_v) * int(px_per_cell_y)
                core_x = int(src_block_ox + pad + off_x)
                core_y = int(src_block_oy + pad + off_y)
                quad_core_positions[(midx, qidx)] = (core_x, core_y)

        if timings_enabled:
            sys.stderr.write(
                f"[btp_vox] atlas pack=by-model size={int(atlas_w)}x{int(atlas_h)} models={len(model_rects)}\n"
            )
    mark("pack")

    if atlas_w == 0 or atlas_h == 0:
        atlas_w = atlas_h = 1

    if pot:
        atlas_w = _next_pow2(int(atlas_w))
        atlas_h = _next_pow2(int(atlas_h))
    if square:
        m = max(int(atlas_w), int(atlas_h))
        atlas_w = int(m)
        atlas_h = int(m)

    if timings_enabled:
        sys.stderr.write(
            f"[btp_vox] atlas final size={int(atlas_w)}x{int(atlas_h)} pot={bool(pot)} square={bool(square)}\n"
        )

    atlas_arr = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    quad_uvs: Dict[Tuple[int, int], QuadUV] = {}
    model_uv: Dict[int, Tuple[float, float, float, float]] = {}

    palette = scene.palette_rgba

    for midx, qidx in owner_keys:
        spec = quad_specs[(midx, qidx)]
        tex_w = int(spec.core_w)
        tex_h = int(spec.core_h)
        full_w = tex_w + pad * 2
        full_h = tex_h + pad * 2
        ox, oy = owner_block_positions[(midx, qidx)]
        if style == "solid":
            block = _quad_block_rgba_solid(spec.colors, palette, pad, full_w, full_h)
        else:
            block = _quad_block_rgba(spec.colors, palette, int(spec.sample_scale), pad, full_w, full_h)
        atlas_arr[oy : oy + full_h, ox : ox + full_w, :] = block
    mark("raster")

    for midx, quads in enumerate(quads_per_model):
        u_min: float | None = None
        v_min: float | None = None
        u_max: float | None = None
        v_max: float | None = None
        for qidx, quad in enumerate(quads):
            spec = quad_specs[(midx, qidx)]
            tex_w = int(spec.core_w)
            tex_h = int(spec.core_h)
            core_x, core_y = quad_core_positions[(midx, qidx)]

            inset_u = min(inset, max(0.0, (tex_w - 1.0) / 2.0))
            inset_v = min(inset, max(0.0, (tex_h - 1.0) / 2.0))

            u0 = (core_x + inset_u) / float(atlas_w)
            v0 = (core_y + inset_v) / float(atlas_h)
            u1 = (core_x + tex_w - inset_u) / float(atlas_w)
            v1 = (core_y + tex_h - inset_v) / float(atlas_h)

            quad_uvs[(midx, qidx)] = QuadUV(u0=u0, v0=v0, u1=u1, v1=v1)

            u_min = float(core_x) if u_min is None else min(u_min, float(core_x))
            v_min = float(core_y) if v_min is None else min(v_min, float(core_y))
            u_max = float(core_x + tex_w) if u_max is None else max(u_max, float(core_x + tex_w))
            v_max = float(core_y + tex_h) if v_max is None else max(v_max, float(core_y + tex_h))

        # Top-left origin convention (matches the baked atlas array). Use --uv-flip-v in the pipeline
        # if your engine treats v=0 as bottom.
        if u_min is None or v_min is None or u_max is None or v_max is None:
            model_uv[midx] = (
                0.0,
                0.0,
                1.0 / float(max(1, atlas_w)),
                1.0 / float(max(1, atlas_h)),
            )
        else:
            model_uv[midx] = (
                float(u_min) / float(atlas_w),
                float(v_min) / float(atlas_h),
                float(u_max) / float(atlas_w),
                float(v_max) / float(atlas_h),
            )

    img_rgba = Image.fromarray(atlas_arr, mode="RGBA")
    if alpha == "rgb":
        img = img_rgba.convert("RGB")
    elif alpha == "auto":
        # If there is no transparency anywhere, strip alpha to reduce file size.
        has_alpha = bool(np.any(atlas_arr[:, :, 3] != 255))
        img = img_rgba if has_alpha else img_rgba.convert("RGB")
    else:
        img = img_rgba

    texture_png = _image_to_bytes(img)

    return AtlasBuildResult(
        width=int(atlas_w),
        height=int(atlas_h),
        texture_png=texture_png,
        quad_uvs=quad_uvs,
        model_uv_rects=model_uv,
    )


def _build_quad_tex_specs(
    quads_per_model: list[list],
    *,
    texel_scale: int,
    compress_solid_quads: bool,
) -> dict[tuple[int, int], _QuadTexSpec]:
    specs: dict[tuple[int, int], _QuadTexSpec] = {}
    ts = int(max(1, int(texel_scale)))

    for midx, quads in enumerate(quads_per_model):
        for qidx, quad in enumerate(quads):
            key = (int(midx), int(qidx))
            arr = np.asarray(quad.colors, dtype=np.int32)
            if arr.ndim != 2:
                arr = np.asarray(arr, dtype=np.int32).reshape((int(quad.size_v), int(quad.size_u)))
            arr = np.ascontiguousarray(arr)

            if arr.size == 0:
                arr = np.zeros((1, 1), dtype=np.int32)

            is_solid = bool(compress_solid_quads) and bool(np.all(arr == arr.flat[0]))
            if is_solid:
                colors = np.asarray([[int(arr.flat[0])]], dtype=np.int32)
                core_w = 1
                core_h = 1
                sample_scale = 1
            else:
                h, w = int(arr.shape[0]), int(arr.shape[1])
                colors = arr
                core_w = int(max(1, w * ts))
                core_h = int(max(1, h * ts))
                sample_scale = int(ts)

            specs[key] = _QuadTexSpec(
                colors=np.ascontiguousarray(colors, dtype=np.int32),
                core_w=int(core_w),
                core_h=int(core_h),
                sample_scale=int(sample_scale),
            )

    return specs


def _apply_face_alias_from_model_name(
    scene: VoxScene,
    quads_per_model: list[list],
    quad_specs: dict[tuple[int, int], _QuadTexSpec],
    *,
    remap_uv: bool = False,
) -> int:
    """Apply face-texture alias rules encoded in model names.

    Naming rule example:
      cube@lrfk@tb
    Means:
      r,f,k -> l
      b -> t
    """

    applied = 0
    for midx, quads in enumerate(quads_per_model):
        if midx < 0 or midx >= len(scene.models):
            continue
        name = str(getattr(scene.models[midx], "name", "") or "")
        rules = _parse_face_alias_rules(name)
        if not rules:
            continue

        face_to_quad_keys: dict[str, list[tuple[int, int]]] = {}
        for qidx, quad in enumerate(quads):
            face = _quad_face_letter(getattr(quad, "axis", None), getattr(quad, "normal_sign", None))
            if face is None:
                continue
            face_to_quad_keys.setdefault(face, []).append((int(midx), int(qidx)))

        # Choose one representative quad per face (largest area first, stable tie-break by qidx).
        face_rep: dict[str, tuple[int, int]] = {}
        for face, keys in face_to_quad_keys.items():
            if not keys:
                continue
            keys_sorted = sorted(
                keys,
                key=lambda k: (
                    -int(quad_specs[k].core_w) * int(quad_specs[k].core_h),
                    int(k[1]),
                ),
            )
            face_rep[face] = keys_sorted[0]

        for src_face, dst_face in rules.items():
            src_keys = face_to_quad_keys.get(src_face) or []
            dst_rep = face_rep.get(dst_face)
            if not src_keys or dst_rep is None:
                continue

            dst_spec = quad_specs[dst_rep]
            for key in src_keys:
                # Remap UV orientation per face so aliases stay visually consistent in Unity
                # (notably for f/k == +Z/-Z after axis conversion).
                quad_specs[key] = _alias_spec_from_face(
                    dst_spec,
                    src_face=src_face,
                    dst_face=dst_face,
                    remap_uv=bool(remap_uv),
                )
                applied += 1

    return int(applied)


def _alias_spec_from_face(
    dst_spec: _QuadTexSpec,
    *,
    src_face: str,
    dst_face: str,
    remap_uv: bool,
) -> _QuadTexSpec:
    if (not remap_uv) or (src_face == dst_face):
        return _QuadTexSpec(
            colors=dst_spec.colors,
            core_w=int(dst_spec.core_w),
            core_h=int(dst_spec.core_h),
            sample_scale=int(dst_spec.sample_scale),
        )

    src_arr = np.asarray(dst_spec.colors, dtype=np.int32)
    canon = _face_local_to_canonical(src_arr, dst_face)
    remapped = _face_canonical_to_local(canon, src_face)
    remapped = np.ascontiguousarray(remapped, dtype=np.int32)

    sample_scale = int(max(1, int(dst_spec.sample_scale)))
    h, w = int(remapped.shape[0]), int(remapped.shape[1])
    return _QuadTexSpec(
        colors=remapped,
        core_w=int(max(1, w * sample_scale)),
        core_h=int(max(1, h * sample_scale)),
        sample_scale=sample_scale,
    )


def _face_local_to_canonical(colors: np.ndarray, face: str) -> np.ndarray:
    # Canonical frame (Unity-oriented):
    # - side faces keep "up" along +Y and horizontal around the model
    # - t/b/l use identity to preserve existing authoring behavior
    if face in ("t", "b", "l"):
        return colors
    if face == "r":
        return np.fliplr(colors)
    if face == "f":
        return colors.T
    if face == "k":
        return np.fliplr(colors.T)
    return colors


def _face_canonical_to_local(colors: np.ndarray, face: str) -> np.ndarray:
    if face in ("t", "b", "l"):
        return colors
    if face == "r":
        return np.fliplr(colors)
    if face == "f":
        return colors.T
    if face == "k":
        return np.fliplr(colors).T
    return colors


def _parse_face_alias_rules(model_name: str) -> dict[str, str]:
    allowed = set("tblrfk")
    parts = str(model_name or "").split("@")
    if len(parts) <= 1:
        return {}

    out: dict[str, str] = {}
    for raw in parts[1:]:
        token = str(raw or "").strip().lower()
        # Allow readable spellings like "@tb lrfk" (treated as "@tblrfk").
        token = re.sub(r"\s+", "", token)
        if len(token) < 2:
            continue
        if any(c not in allowed for c in token):
            continue
        target = token[0]
        for src in token[1:]:
            if src == target:
                continue
            if src in out:
                continue
            out[src] = target
    return out


def _quad_face_letter(axis: int | None, normal_sign: int | None) -> str | None:
    try:
        a = int(axis) if axis is not None else -999
        s = int(normal_sign) if normal_sign is not None else 0
    except Exception:
        return None

    # Keep exactly the same letter convention as --cull (MagicaVoxel model space):
    # t=+Z, b=-Z, l=-X, r=+X, f=+Y, k=-Y
    if a == 2 and s > 0:
        return "t"
    if a == 2 and s < 0:
        return "b"
    if a == 0 and s < 0:
        return "l"
    if a == 0 and s > 0:
        return "r"
    if a == 1 and s > 0:
        return "f"
    if a == 1 and s < 0:
        return "k"
    return None


def _build_quad_reuse_map(
    quads_per_model: list[list],
    *,
    quad_specs: dict[tuple[int, int], _QuadTexSpec],
    layout: str,
    enable: bool,
) -> tuple[dict[tuple[int, int], _QuadAnchor], list[tuple[int, int]]]:
    anchors: dict[tuple[int, int], _QuadAnchor] = {}
    timings_enabled = bool(os.environ.get("BTP_VOX_TIMINGS"))
    subrect_limit = int(os.environ.get("BTP_VOX_REUSE_SUBRECT_LIMIT", "2000"))
    max_candidates = int(os.environ.get("BTP_VOX_REUSE_MAX_CANDIDATES", "128"))

    all_keys: list[tuple[int, int]] = []
    quad_colors: dict[tuple[int, int], np.ndarray] = {}
    quad_area: dict[tuple[int, int], int] = {}
    quad_shape: dict[tuple[int, int], tuple[int, int]] = {}

    for midx, quads in enumerate(quads_per_model):
        for qidx, _quad in enumerate(quads):
            key = (int(midx), int(qidx))
            spec = quad_specs[key]
            arr = np.asarray(spec.colors, dtype=np.int32)
            h, w = int(arr.shape[0]), int(arr.shape[1])
            quad_colors[key] = arr
            quad_shape[key] = (h, w)
            quad_area[key] = int(h * w)
            all_keys.append(key)

    if not enable:
        for key in all_keys:
            anchors[key] = _QuadAnchor(src_midx=int(key[0]), src_qidx=int(key[1]), off_u=0, off_v=0)
        return anchors, all_keys

    def process_scope(scope_keys: list[tuple[int, int]]) -> None:
        order = sorted(
            scope_keys,
            key=lambda k: (-int(quad_area[k]), -int(quad_shape[k][0]), -int(quad_shape[k][1]), int(k[0]), int(k[1])),
        )
        owners: list[tuple[int, int]] = []
        exact_owner: dict[tuple[int, int, bytes], tuple[int, int]] = {}
        owners_by_value: dict[int, list[tuple[int, int]]] = {}
        can_subrect = (subrect_limit <= 0) or (len(scope_keys) <= int(subrect_limit))
        attempts = 0

        def _add_owner(key: tuple[int, int], sig: tuple[int, int, bytes]) -> None:
            owners.append(key)
            exact_owner[sig] = key
            anchors[key] = _QuadAnchor(src_midx=int(key[0]), src_qidx=int(key[1]), off_u=0, off_v=0)
            vals = np.unique(quad_colors[key].reshape((-1,)))
            for vv in vals.tolist():
                owners_by_value.setdefault(int(vv), []).append(key)

        for key in order:
            arr = quad_colors[key]
            h, w = quad_shape[key]
            sig = (int(h), int(w), arr.tobytes())
            eq = exact_owner.get(sig)
            if eq is not None:
                anchors[key] = _QuadAnchor(src_midx=int(eq[0]), src_qidx=int(eq[1]), off_u=0, off_v=0)
                continue

            found_src: tuple[int, int] | None = None
            found_off: tuple[int, int] | None = None

            if can_subrect and owners:
                owner_candidates = owners
                vals, cnts = np.unique(arr.reshape((-1,)), return_counts=True)
                if vals.size > 0:
                    anchor_val = int(vals[int(np.argmin(cnts))])
                    by_val = owners_by_value.get(anchor_val)
                    if by_val:
                        owner_candidates = by_val

                if max_candidates > 0 and len(owner_candidates) > max_candidates:
                    owner_candidates = owner_candidates[: max_candidates]

                for owner_key in owner_candidates:
                    attempts += 1
                    oh, ow = quad_shape[owner_key]
                    if oh < h or ow < w:
                        continue
                    off = _find_subrect_offset(quad_colors[owner_key], arr)
                    if off is None:
                        continue
                    found_src = owner_key
                    found_off = off
                    break

            if found_src is not None and found_off is not None:
                anchors[key] = _QuadAnchor(
                    src_midx=int(found_src[0]),
                    src_qidx=int(found_src[1]),
                    off_u=int(found_off[0]),
                    off_v=int(found_off[1]),
                )
            else:
                _add_owner(key, sig)

        if timings_enabled:
            sys.stderr.write(
                f"[btp_vox] reuse scope={len(scope_keys)} can_subrect={int(can_subrect)} "
                f"owners={len(owners)} attempts={int(attempts)} max_candidates={int(max_candidates)}\n"
            )

    if layout == "global":
        process_scope(all_keys)
    else:
        for midx, quads in enumerate(quads_per_model):
            keys = [(int(midx), int(qidx)) for qidx in range(len(quads))]
            process_scope(keys)

    owner_keys = [
        key
        for key in all_keys
        if int(anchors[key].src_midx) == int(key[0]) and int(anchors[key].src_qidx) == int(key[1])
    ]
    return anchors, owner_keys


def _find_subrect_offset(host: np.ndarray, child: np.ndarray) -> tuple[int, int] | None:
    host = np.asarray(host, dtype=np.int32)
    child = np.asarray(child, dtype=np.int32)

    if host.ndim != 2 or child.ndim != 2:
        return None

    host_h, host_w = int(host.shape[0]), int(host.shape[1])
    child_h, child_w = int(child.shape[0]), int(child.shape[1])

    if child_h > host_h or child_w > host_w:
        return None

    if child_h == host_h and child_w == host_w:
        if np.array_equal(host, child):
            return (0, 0)
        return None

    # Ultra-common path with solid-quad compression: avoid huge sliding windows.
    if child_h == 1 and child_w == 1:
        hits = np.argwhere(host == int(child[0, 0]))
        if hits.size == 0:
            return None
        off_v, off_u = hits[0]
        return (int(off_u), int(off_v))

    # Heuristic: sliding-window can explode memory/time on large hosts.
    view_h = host_h - child_h + 1
    view_w = host_w - child_w + 1
    if view_h <= 0 or view_w <= 0:
        return None
    window_ops = int(view_h) * int(view_w) * int(child_h) * int(child_w)
    if window_ops <= 2_000_000:
        try:
            windows = np.lib.stride_tricks.sliding_window_view(host, (child_h, child_w))
        except Exception:
            windows = None
        if windows is not None:
            matches = np.all(windows == child, axis=(2, 3))
            found = np.argwhere(matches)
            if found.size != 0:
                off_v, off_u = found[0]
                return (int(off_u), int(off_v))
            return None

    # Sparse anchor scan fallback: usually much faster for large matrices.
    vals, cnts = np.unique(child.reshape((-1,)), return_counts=True)
    anchor_val = int(vals[int(np.argmin(cnts))])
    anchor_idx = np.argwhere(child == anchor_val)
    if anchor_idx.size == 0:
        return None
    ay, ax = anchor_idx[0]

    hits = np.argwhere(host == anchor_val)
    if hits.size == 0:
        return None

    max_v = host_h - child_h
    max_u = host_w - child_w
    for hy, hx in hits:
        off_v = int(hy - ay)
        off_u = int(hx - ax)
        if off_v < 0 or off_u < 0 or off_v > max_v or off_u > max_u:
            continue
        if np.array_equal(host[off_v : off_v + child_h, off_u : off_u + child_w], child):
            return (int(off_u), int(off_v))

    return None


def _pack_rects(rects: List[Tuple[int, int, int]]) -> Tuple[int, int, Dict[int, Tuple[int, int]]]:
    positions: Dict[int, Tuple[int, int]] = {}
    if not rects:
        return 0, 0, positions

    total_area = sum(w * h for _, w, h in rects)
    row_width = max(max(w for _, w, _ in rects), int(np.ceil(np.sqrt(total_area))))

    x = 0
    y = 0
    row_h = 0
    max_w = 0

    for rid, w, h in rects:
        if x + w > row_width and x != 0:
            x = 0
            y += row_h
            row_h = 0
        positions[rid] = (x, y)
        x += w
        row_h = max(row_h, h)
        max_w = max(max_w, x)

    total_h = y + row_h
    return int(max_w), int(total_h), positions


def _pack_rects_with_row_width(
    rects: list[tuple[int, int, int]],
    *,
    row_width: int,
) -> tuple[int, int, Dict[int, tuple[int, int]]]:
    positions: Dict[int, Tuple[int, int]] = {}
    if not rects:
        return 0, 0, positions

    row_width = max(1, int(row_width))

    x = 0
    y = 0
    row_h = 0
    max_w = 0

    for rid, w, h in rects:
        if x + int(w) > row_width and x != 0:
            x = 0
            y += row_h
            row_h = 0
        positions[int(rid)] = (int(x), int(y))
        x += int(w)
        row_h = max(int(row_h), int(h))
        max_w = max(int(max_w), int(x))

    total_h = int(y + row_h)
    return int(max_w), int(total_h), positions


def _pack_rects_best(
    rects: list[tuple[int, int, int]],
    *,
    pot: bool,
    square: bool,
) -> tuple[int, int, Dict[int, tuple[int, int]]]:
    """Fast shelf pack, but try multiple widths to reduce POT/square expansion waste."""

    if not rects:
        return 1, 1, {}

    # Sort by height then width to improve shelf packing stability.
    rects_sorted = sorted(rects, key=lambda t: (t[2], t[1]), reverse=True)

    total_area = int(sum(int(w) * int(h) for _, w, h in rects_sorted))
    max_w = int(max(int(w) for _, w, _ in rects_sorted))
    max_h = int(max(int(h) for _, _, h in rects_sorted))
    base = int(np.ceil(np.sqrt(max(1, total_area))))

    # Candidate widths around sqrt(area). Wider makes atlas shorter; narrower makes it taller.
    # When POT is requested, prefer widths near power-of-two boundaries to reduce expansion waste.
    candidates: set[int] = {max_w}
    if pot:
        # Try a few power-of-two widths around the sqrt(area) estimate.
        base_pow = 1 << max(1, int(base - 1).bit_length())
        for k in range(-2, 4):
            w = int(base_pow) << max(0, k)
            if w > 0:
                candidates.add(w)
        # Also try slightly under/over POT boundaries.
        for w in list(candidates):
            candidates.add(int(w * 0.9))
            candidates.add(int(w * 1.1))
    else:
        candidates.update(
            {
                base,
                int(base * 1.25),
                int(base * 1.5),
                int(base * 2.0),
                int(base * 0.75),
            }
        )
    candidates = [w for w in candidates if w >= max_w]
    candidates.sort()

    best = None
    for rw in candidates:
        w0, h0, pos0 = _pack_rects_with_row_width(rects_sorted, row_width=int(rw))
        if w0 <= 0 or h0 <= 0:
            continue

        w1 = int(w0)
        h1 = int(h0)
        if pot:
            w1 = _next_pow2(w1)
            h1 = _next_pow2(h1)
        if square:
            m = max(w1, h1)
            w1 = h1 = int(m)

        area1 = int(w1) * int(h1)
        aspect = float(max(w1, h1)) / float(max(1, min(w1, h1)))
        score = float(area1) * (1.0 + max(0.0, aspect - 2.0) * 0.05)

        if best is None or score < best[0]:
            best = (score, w0, h0, pos0)

    if best is None:
        return _pack_rects(rects_sorted)

    _score, w_best, h_best, pos_best = best
    # Return pre-POT size here; caller will POT/square the final atlas.
    return int(w_best), int(h_best), pos_best


def _next_pow2(x: int) -> int:
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _pack_maxrect(rects_sorted: list[tuple[int, int, int]], width: int, height: int) -> Dict[int, Tuple[int, int]] | None:
    # MaxRects (best short side fit), no rotation.
    free: list[tuple[int, int, int, int]] = [(0, 0, int(width), int(height))]
    out: Dict[int, Tuple[int, int]] = {}

    def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

    def _prune(frs: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        pruned: list[tuple[int, int, int, int]] = []
        for i, a in enumerate(frs):
            ax, ay, aw, ah = a
            contained = False
            for j, b in enumerate(frs):
                if i == j:
                    continue
                bx, by, bw, bh = b
                if ax >= bx and ay >= by and ax + aw <= bx + bw and ay + ah <= by + bh:
                    contained = True
                    break
            if not contained and aw > 0 and ah > 0:
                pruned.append(a)
        return pruned

    for rid, w, h in rects_sorted:
        bw = int(w)
        bh = int(h)
        best_i = -1
        best_x = 0
        best_y = 0
        best_ss = None
        best_ls = None
        for i, (fx, fy, fw, fh) in enumerate(free):
            if bw <= fw and bh <= fh:
                ss = min(fw - bw, fh - bh)
                ls = max(fw - bw, fh - bh)
                if best_ss is None or ss < best_ss or (ss == best_ss and ls < best_ls):
                    best_ss = ss
                    best_ls = ls
                    best_i = i
                    best_x = fx
                    best_y = fy
        if best_i < 0:
            return None

        out[int(rid)] = (int(best_x), int(best_y))
        placed = (int(best_x), int(best_y), bw, bh)

        new_free: list[tuple[int, int, int, int]] = []
        for fr in free:
            if not _intersects(fr, placed):
                new_free.append(fr)
                continue
            fx, fy, fw, fh = fr
            px, py, pw, ph = placed
            pr = px + pw
            pb = py + ph
            fr_r = fx + fw
            fr_b = fy + fh
            if px > fx:
                new_free.append((fx, fy, px - fx, fh))
            if pr < fr_r:
                new_free.append((pr, fy, fr_r - pr, fh))
            if py > fy:
                new_free.append((fx, fy, fw, py - fy))
            if pb < fr_b:
                new_free.append((fx, pb, fw, fr_b - pb))
        free = _prune(new_free)

    return out


def _pack_best_bin(
    rects: list[tuple[int, int, int]],
    *,
    square: bool,
) -> tuple[int, int, Dict[int, Tuple[int, int]]]:
    # Mirror magicavoxel_merge approach: try POT widths/heights and choose best.
    if not rects:
        return 1, 1, {}

    rects_sorted = sorted(rects, key=lambda t: (t[2], t[1]), reverse=True)
    total_area = int(sum(int(w) * int(h) for _, w, h in rects_sorted))
    max_rw = int(max(int(w) for _, w, _ in rects_sorted))
    max_rh = int(max(int(h) for _, _, h in rects_sorted))

    min_kw = max(4, int(max_rw - 1).bit_length())
    min_kh = max(4, int(max_rh - 1).bit_length())
    widths = [1 << k for k in range(min_kw, 14)]
    heights = [1 << k for k in range(min_kh, 14)]

    best_pos = None
    best_w = None
    best_h = None
    best_area = None
    best_max = None
    best_score = None

    for w in widths:
        for h in heights:
            if int(w) < max_rw or int(h) < max_rh:
                continue
            if int(w) * int(h) < total_area:
                continue
            pos = _pack_maxrect(rects_sorted, int(w), int(h))
            if pos is None:
                continue
            area = int(w) * int(h)
            if square:
                m = max(int(w), int(h))
                if best_area is None or m < best_max or (m == best_max and area < best_area):
                    best_area = area
                    best_max = m
                    best_pos = pos
                    best_w = int(w)
                    best_h = int(h)
            else:
                aspect = float(max(int(w), int(h))) / float(min(int(w), int(h)))
                penalty = 0.0
                if aspect > 2.0:
                    penalty = (aspect - 2.0) * 0.25
                score = float(area) * (1.0 + penalty)
                if best_score is None or score < best_score:
                    best_score = score
                    best_area = area
                    best_pos = pos
                    best_w = int(w)
                    best_h = int(h)

    if best_pos is None or best_w is None or best_h is None:
        # Fallback to simple shelf packing
        w0, h0, pos0 = _pack_rects(rects)
        return int(w0), int(h0), pos0

    return int(best_w), int(best_h), best_pos


def _image_to_bytes(img: Image.Image) -> bytes:
    import io

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def _quad_block_rgba(
    colors: np.ndarray,
    palette: np.ndarray,
    texel_scale: int,
    pad: int,
    full_w: int,
    full_h: int,
) -> np.ndarray:
    texel_scale = max(1, int(texel_scale))
    core_h = full_h - pad * 2
    core_w = full_w - pad * 2

    # Vectorized palette lookup and texel scaling.
    # `colors` is stored as a (V, U) grid (rows=V, cols=U).
    # Palette index 0 is treated as fully transparent.
    idx0 = colors.astype(np.int32)
    idx = np.clip(idx0 - 1, 0, 255)
    rgba = palette[idx]  # (V, U, 4)
    if rgba.size:
        rgba = rgba.copy()
        rgba[idx0 == 0, 3] = 0
    if texel_scale != 1:
        rgba = np.repeat(np.repeat(rgba, texel_scale, axis=0), texel_scale, axis=1)

    core = rgba[:core_h, :core_w, :].astype(np.uint8, copy=False)

    if pad > 0:
        block = np.pad(core, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    else:
        block = core

    return block


def _quad_block_rgba_solid(
    colors: np.ndarray,
    palette: np.ndarray,
    pad: int,
    full_w: int,
    full_h: int,
) -> np.ndarray:
    core_h = full_h - pad * 2
    core_w = full_w - pad * 2
    core = np.zeros((core_h, core_w, 4), dtype=np.uint8)

    # Pick the most common non-zero palette index.
    flat = colors.reshape((-1,))
    flat = flat[flat > 0]
    if flat.size == 0:
        rgba = np.asarray([255, 0, 255, 255], dtype=np.uint8)
    else:
        # bincount needs non-negative ints
        bc = np.bincount(flat.astype(np.int32))
        idx = int(np.argmax(bc))
        rgba = palette[max(0, min(255, idx - 1))]

    core[:, :, :] = rgba
    if pad > 0:
        return np.pad(core, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return core
