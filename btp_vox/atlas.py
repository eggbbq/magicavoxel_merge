"""Texture atlas construction for the BTP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

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
) -> AtlasBuildResult:
    """Pack quads into a simple atlas and return texture bytes + UV lookup."""

    if style not in ("baked", "solid"):
        raise ValueError("style must be 'baked' or 'solid'")
    if layout not in ("by-model", "global"):
        raise ValueError("layout must be 'by-model' or 'global'")

    quads_per_model = mesher_result.quads_per_model

    # Two layouts:
    # - by-model: each model gets its own packed block, then blocks are packed into atlas
    # - global: all quads are packed together into a single atlas (usually better fill rate)
    quad_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    model_block_sizes: Dict[int, Tuple[int, int]] = {}
    model_positions: Dict[int, Tuple[int, int]] = {}

    if layout == "global":
        rects_all: List[Tuple[int, int, int]] = []
        rid_to_mq: dict[int, tuple[int, int]] = {}
        rid = 0
        for midx, quads in enumerate(quads_per_model):
            for qidx, quad in enumerate(quads):
                tex_w = int(max(1, quad.size_u * texel_scale)) + pad * 2
                tex_h = int(max(1, quad.size_v * texel_scale)) + pad * 2
                rects_all.append((rid, tex_w, tex_h))
                rid_to_mq[rid] = (midx, qidx)
                rid += 1

        # NOTE: For large scenes, MaxRects search over many POT bins becomes extremely slow.
        # Use a fast shelf pack, then optionally expand to POT/square. This trades some
        # fill efficiency for dramatically better runtime.
        use_fast_pack = len(rects_all) >= 512
        if pot or square:
            if use_fast_pack:
                atlas_w, atlas_h, positions = _pack_rects(rects_all)
            else:
                atlas_w, atlas_h, positions = _pack_best_bin(rects_all, square=bool(square))
        else:
            atlas_w, atlas_h, positions = _pack_rects(rects_all)

        for rid, (x, y) in positions.items():
            midx, qidx = rid_to_mq[int(rid)]
            quad_positions[(midx, qidx)] = (int(x), int(y))

        # In global layout, each model's rect is computed from its quads' extents.
        for midx, quads in enumerate(quads_per_model):
            if not quads:
                model_block_sizes[midx] = (1, 1)
                model_positions[midx] = (0, 0)
                continue
            u_min = None
            v_min = None
            u_max = None
            v_max = None
            for qidx, quad in enumerate(quads):
                ox, oy = quad_positions[(midx, qidx)]
                tex_w = int(max(1, quad.size_u * texel_scale)) + pad * 2
                tex_h = int(max(1, quad.size_v * texel_scale)) + pad * 2
                u_min = ox if u_min is None else min(u_min, ox)
                v_min = oy if v_min is None else min(v_min, oy)
                u_max = (ox + tex_w) if u_max is None else max(u_max, ox + tex_w)
                v_max = (oy + tex_h) if v_max is None else max(v_max, oy + tex_h)
            model_positions[midx] = (int(u_min or 0), int(v_min or 0))
            model_block_sizes[midx] = (int((u_max or 1) - (u_min or 0)), int((v_max or 1) - (v_min or 0)))

    else:
        quad_local_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for midx, quads in enumerate(quads_per_model):
            rects: List[Tuple[int, int, int]] = []
            for qidx, quad in enumerate(quads):
                tex_w = int(max(1, quad.size_u * texel_scale)) + pad * 2
                tex_h = int(max(1, quad.size_v * texel_scale)) + pad * 2
                rects.append((qidx, tex_w, tex_h))

            if bool(tight_blocks):
                block_w, block_h, positions = _pack_rects(rects)
            else:
                if pot or square:
                    use_fast_pack = len(rects) >= 512
                    if use_fast_pack:
                        block_w, block_h, positions = _pack_rects(rects)
                    else:
                        block_w, block_h, positions = _pack_best_bin(rects, square=bool(square))
                else:
                    block_w, block_h, positions = _pack_rects(rects)
            for qidx, pos in positions.items():
                quad_local_positions[(midx, qidx)] = pos
            model_block_sizes[midx] = (block_w, block_h)

        model_rects = [(midx, sz[0], sz[1]) for midx, sz in model_block_sizes.items()]
        if pot or square:
            use_fast_pack = len(model_rects) >= 512
            if use_fast_pack:
                atlas_w, atlas_h, model_positions = _pack_rects(model_rects)
            else:
                atlas_w, atlas_h, model_positions = _pack_best_bin(model_rects, square=bool(square))
        else:
            atlas_w, atlas_h, model_positions = _pack_rects(model_rects)

        for midx, quads in enumerate(quads_per_model):
            mx, my = model_positions.get(midx, (0, 0))
            for qidx, _quad in enumerate(quads):
                qx, qy = quad_local_positions[(midx, qidx)]
                quad_positions[(midx, qidx)] = (int(mx + qx), int(my + qy))

    if atlas_w == 0 or atlas_h == 0:
        atlas_w = atlas_h = 1

    if pot:
        atlas_w = _next_pow2(int(atlas_w))
        atlas_h = _next_pow2(int(atlas_h))
    if square:
        m = max(int(atlas_w), int(atlas_h))
        atlas_w = int(m)
        atlas_h = int(m)

    atlas_arr = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    quad_uvs: Dict[Tuple[int, int], QuadUV] = {}
    model_uv: Dict[int, Tuple[float, float, float, float]] = {}

    palette = scene.palette_rgba

    for midx, quads in enumerate(quads_per_model):
        block_origin = model_positions.get(midx, (0, 0))
        block_w, block_h = model_block_sizes.get(midx, (1, 1))
        mx, my = block_origin

        # Top-left origin convention (matches the baked atlas array). Use --flip-v in the pipeline
        # if your engine treats v=0 as bottom.
        model_uv[midx] = (
            float(mx) / float(atlas_w),
            float(my) / float(atlas_h),
            float(mx + block_w) / float(atlas_w),
            float(my + block_h) / float(atlas_h),
        )

        for qidx, quad in enumerate(quads):
            tex_w = int(max(1, quad.size_u * texel_scale))
            tex_h = int(max(1, quad.size_v * texel_scale))
            full_w = tex_w + pad * 2
            full_h = tex_h + pad * 2

            ox, oy = quad_positions[(midx, qidx)]

            if style == "solid":
                block = _quad_block_rgba_solid(quad.colors, palette, pad, full_w, full_h)
            else:
                block = _quad_block_rgba(quad.colors, palette, texel_scale, pad, full_w, full_h)
            atlas_arr[oy : oy + full_h, ox : ox + full_w, :] = block

            inset_u = min(inset, max(0.0, (tex_w - 1.0) / 2.0))
            inset_v = min(inset, max(0.0, (tex_h - 1.0) / 2.0))

            u0 = (ox + pad + inset_u) / float(atlas_w)
            v0 = (oy + pad + inset_v) / float(atlas_h)
            u1 = (ox + pad + tex_w - inset_u) / float(atlas_w)
            v1 = (oy + pad + tex_h - inset_v) / float(atlas_h)

            quad_uvs[(midx, qidx)] = QuadUV(u0=u0, v0=v0, u1=u1, v1=v1)

    img = Image.fromarray(atlas_arr, mode="RGBA")
    texture_png = _image_to_bytes(img)

    return AtlasBuildResult(
        width=int(atlas_w),
        height=int(atlas_h),
        texture_png=texture_png,
        quad_uvs=quad_uvs,
        model_uv_rects=model_uv,
    )


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
    widths = [1 << k for k in range(4, 14)]
    heights = [1 << k for k in range(4, 14)]

    best_pos = None
    best_w = None
    best_h = None
    best_area = None
    best_max = None
    best_score = None

    for w in widths:
        for h in heights:
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
    idx = colors.astype(np.int32)
    idx = np.clip(idx - 1, 0, 255)
    rgba = palette[idx]  # (V, U, 4)
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
