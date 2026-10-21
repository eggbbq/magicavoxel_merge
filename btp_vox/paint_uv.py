"""Deterministic voxel surface unfolding for painting, without geometry changes.

Outer bounding planes form a box net (left/front/right/back, top/bottom).
Recessed planes are separate charts: projecting different depths onto one face
would overlap paintable surfaces. Coplanar quads retain their spatial adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .atlas import (
    AtlasBuildResult, QuadUV, _image_to_bytes, _next_pow2,
    _pack_rects_best, _pack_rects_fixed,
)
from .btp_mesher import MesherResult, Quad
from .voxio import VoxScene


_FACES = {(0, -1): "l", (0, 1): "r", (1, -1): "k",
          (1, 1): "f", (2, -1): "b", (2, 1): "t"}


def unique_scene_models(scene: VoxScene) -> VoxScene:
    """Give each shape occurrence its own UVs, including mirrored limb instances.

    Voxel arrays can stay shared; neither the mesher nor this module edits them.
    Keep rule suffixes intact for cutout/culling, but never apply face aliases.
    """
    models = list(scene.models)
    nodes = [replace(n, children=list(n.children), model_ids=list(n.model_ids))
             for n in scene.nodes]
    used: set[int] = set()
    for node in nodes:
        if node.kind != "shp":
            continue
        ids = []
        for mid in node.model_ids:
            if mid in used:
                ids.append(len(models))
                models.append(replace(scene.models[mid]))
            else:
                used.add(mid)
                ids.append(mid)
        node.model_ids = ids
    return replace(scene, models=models, nodes=nodes, root_node_ids=list(scene.root_node_ids))


@dataclass
class _Chart:
    model: int
    name: str
    kind: str
    width: int
    height: int
    # Pixel corners in the same order as Quad.verts, before atlas placement.
    quads: dict[int, np.ndarray]
    x: int = 0
    y: int = 0


@dataclass
class PaintLayout:
    width: int
    height: int
    texel_scale: int
    pad: int
    charts: list[_Chart]
    quads_per_model: list[list[Quad]]

    def _records(self) -> list[dict]:
        records = []
        for cid, chart in enumerate(self.charts):
            quads = []
            for qid, points in chart.quads.items():
                q = self.quads_per_model[chart.model][qid]
                points = points + (chart.x, chart.y)
                quads.append({
                    "quad": qid, "face": _FACES[(q.axis, q.normal_sign)],
                    "plane": int(q.verts[0][q.axis]),
                    "pixels": points.tolist(),
                })
            records.append({
                "id": cid, "model": chart.model, "name": chart.name,
                "kind": chart.kind,
                "rect": [chart.x, chart.y, chart.width, chart.height],
                "quads": quads,
            })
        return records

    def write_json(self, path: str | Path, *, flip_v: bool = False) -> None:
        """Pixel positions always address the PNG, even when exported UVs flip V."""
        _write(path, json.dumps({
            "version": 1, "mode": "paint", "width": self.width, "height": self.height,
            "pixel_origin": "top-left", "uv_flip_v": flip_v,
            "texel_scale": self.texel_scale, "padding": self.pad,
            "face_space": "MagicaVoxel: t=+Z b=-Z l=-X r=+X f=+Y k=-Y",
            "charts": self._records(),
        }, ensure_ascii=False, indent=2) + "\n")

    def write_template(self, path: str | Path, *, flip_v: bool = False) -> None:
        """Transparent overlay in PNG pixel space; labels never enter the texture."""
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                 f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
                 '<title>Voxel painting UV guide</title>',
                 f'<desc>PNG pixel coordinates, top-left origin; exported UV flip V: {flip_v}. '
                 'Face letters use MagicaVoxel axes.</desc>']
        for record in self._records():
            label = escape(f'{record["name"]} [model {record["model"]}] {record["kind"]}')
            lines.append(f'<g id="chart-{record["id"]}" fill="none" stroke="#222" '
                         f'stroke-width="0.6"><title>{label}</title>')
            faces: dict[tuple[str, int], list[np.ndarray]] = {}
            for quad in record["quads"]:
                points = quad["pixels"]
                coords = " ".join(f"{x},{y}" for x, y in points)
                title = escape(f'{record["name"]} [{record["model"]}] '
                               f'{quad["face"]} depth={quad["plane"]} quad={quad["quad"]}')
                lines.append(f'<polygon points="{coords}"><title>{title}</title></polygon>')
                faces.setdefault((quad["face"], quad["plane"]), []).append(np.asarray(points))
            lines.append('</g>')
            for (face, plane), polygons in faces.items():
                # Anchor on the largest actual quad, rather than a possibly empty
                # centre of a disconnected or concave planar projection.
                points = max(polygons, key=lambda p: int(np.prod(np.ptp(p, axis=0))))
                x, y = points.mean(axis=0)
                font = max(2, min(12, int(np.min(np.ptp(points, axis=0))) // 4))
                text = escape(f'{record["name"]} [{record["model"]}] {face}:{plane}')
                lines.append(f'<text x="{x:g}" y="{y:g}" text-anchor="middle" '
                             f'font-family="sans-serif" font-size="{font}" '
                             f'fill="#111" stroke="white" stroke-width="0.4" '
                             f'paint-order="stroke">{text}</text>')
        lines.append('</svg>')
        _write(path, "\n".join(lines) + "\n")


def _write(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _project(points: np.ndarray, face: str, size: np.ndarray) -> np.ndarray:
    """A box net with five shared edges, no reflected outward face orientations."""
    x, y, z = points.T
    sx, sy, sz = size
    u, v = {
        "l": (y, sy + sz - z),
        "f": (sy + x, sy + sz - z),
        "r": (sy + sx + sy - y, sy + sz - z),
        "k": (2 * sy + 2 * sx - x, sy + sz - z),
        "t": (sy + x, y),
        "b": (sy + x, sy + sz + sy - y),
    }[face]
    return np.stack((u, v), axis=1)


def _model_charts(midx: int, name: str, quads: list[Quad], scale: int, pad: int) -> list[_Chart]:
    if not quads:
        return []
    vertices = np.asarray([q.verts for q in quads], dtype=np.int64)
    low = vertices.min(axis=(0, 1))
    high = vertices.max(axis=(0, 1))
    size = high - low
    outer: dict[int, np.ndarray] = {}
    recessed: dict[tuple[int, int, int], dict[int, np.ndarray]] = {}
    for qid, quad in enumerate(quads):
        face = _FACES[(quad.axis, quad.normal_sign)]
        plane = int(vertices[qid, 0, quad.axis])
        boundary = high[quad.axis] if quad.normal_sign > 0 else low[quad.axis]
        points = _project(vertices[qid] - low, face, size) * scale
        if plane == boundary:
            outer[qid] = points
        else:
            recessed.setdefault((quad.axis, quad.normal_sign, plane), {})[qid] = points

    charts = []
    if outer:
        # Crop the unused part of a partial net (e.g. a culled or cutout model).
        charts.append(_crop_chart(midx, name, "box-net", outer, pad))
    for (axis, sign, plane), points in sorted(recessed.items()):
        face = _FACES[(axis, sign)]
        charts.append(_crop_chart(midx, name, f"plane-{face}-{plane}", points, pad))
    return charts


def _crop_chart(midx: int, name: str, kind: str, quads: dict[int, np.ndarray], pad: int) -> _Chart:
    points = np.concatenate(list(quads.values()))
    low = points.min(axis=0)
    width, height = points.max(axis=0) - low + 2 * pad
    return _Chart(midx, name, kind, int(width), int(height),
                  {qid: p - low + pad for qid, p in quads.items()})


def _blit_quad(block: np.ndarray, mask: np.ndarray, points: np.ndarray, quad: Quad,
               palette: np.ndarray, scale: int) -> None:
    """Rasterize the original color grid through the same affine map as the UVs.

    Using Quad.verts rather than axis conventions also handles rotated cutouts.
    All transforms here are signed permutations of the two texel axes.
    """
    du = (points[1] - points[0]) // quad.size_u
    dv = (points[3] - points[0]) // quad.size_v
    colors = np.asarray(quad.colors, dtype=np.int32)
    rgba = palette[np.clip(colors - 1, 0, 255)].copy()
    rgba[colors == 0, 3] = 0
    rgba = np.repeat(np.repeat(rgba, scale, axis=0), scale, axis=1)
    # Source grid columns follow du, rows follow dv. Make output columns X,
    # rows Y, then reverse either axis when the projection changes direction.
    if du[0] == 0:
        rgba = rgba.transpose(1, 0, 2)
    x_vector = du if du[0] != 0 else dv
    y_vector = dv if dv[1] != 0 else du
    if x_vector[0] < 0:
        rgba = rgba[:, ::-1]
    if y_vector[1] < 0:
        rgba = rgba[::-1]
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    block[y0:y1, x0:x1] = rgba
    mask[y0:y1, x0:x1] = True


def _bleed(block: np.ndarray, mask: np.ndarray, pad: int) -> None:
    """Extend chart colours into gutters, keeping every surface texel intact."""
    for _ in range(pad):
        previous = mask.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            sy = slice(max(0, -dy), min(mask.shape[0], mask.shape[0] - dy))
            sx = slice(max(0, -dx), min(mask.shape[1], mask.shape[1] - dx))
            ty = slice(max(0, dy), min(mask.shape[0], mask.shape[0] + dy))
            tx = slice(max(0, dx), min(mask.shape[1], mask.shape[1] + dx))
            fill = previous[sy, sx] & ~mask[ty, tx]
            block[ty, tx][fill] = block[sy, sx][fill]
            mask[ty, tx][fill] = True


def build_paint_atlas(
    scene: VoxScene, mesher_result: MesherResult, *, pad: int = 2,
    texel_scale: int = 1, square: bool = False, pot: bool = False,
    fixed_size: tuple[int, int] | None = None, alpha: str = "auto",
) -> tuple[AtlasBuildResult, PaintLayout]:
    if pad < 0 or texel_scale < 1:
        raise ValueError("paint UV requires tex-pad >= 0 and tex-texel-scale >= 1")
    if alpha not in ("auto", "rgb", "rgba"):
        raise ValueError("alpha must be 'auto', 'rgb', or 'rgba'")
    charts = []
    for midx, quads in enumerate(mesher_result.quads_per_model):
        charts.extend(_model_charts(midx, scene.models[midx].name, quads, texel_scale, pad))

    # Pack all charts of one model together. No rotation or content-dependent
    # deduplication: the net remains readable and identical colours stay unique.
    model_rects = []
    for midx in range(len(scene.models)):
        local = [(cid, c.width, c.height) for cid, c in enumerate(charts) if c.model == midx]
        if not local:
            continue
        w, h, positions = _pack_rects_best(local, pot=False, square=False)
        model_rects.append((midx, w, h))
        for cid, (x, y) in positions.items():
            charts[cid].x, charts[cid].y = x, y

    if fixed_size is not None:
        if min(fixed_size) <= 0:
            raise ValueError("fixed_size must contain positive width and height")
        width, height, model_positions = _pack_rects_fixed(
            model_rects, width=fixed_size[0], height=fixed_size[1])
    else:
        width, height, model_positions = _pack_rects_best(model_rects, pot=pot, square=square)
        width, height = max(1, width), max(1, height)
        if pot:
            width, height = _next_pow2(width), _next_pow2(height)
        if square:
            width = height = max(width, height)

    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    quad_uvs = {}
    model_uvs = {}
    for midx, w, h in model_rects:
        x, y = model_positions[midx]
        model_uvs[midx] = (x / width, y / height, (x + w) / width, (y + h) / height)
    for chart in charts:
        ox, oy = model_positions[chart.model]
        chart.x += ox
        chart.y += oy
        block = np.zeros((chart.height, chart.width, 4), dtype=np.uint8)
        mask = np.zeros((chart.height, chart.width), dtype=bool)
        for qid, points in chart.quads.items():
            quad = mesher_result.quads_per_model[chart.model][qid]
            _blit_quad(block, mask, points, quad, scene.palette_rgba, texel_scale)
            uv = (points + (chart.x, chart.y)) / (width, height)
            quad_uvs[(chart.model, qid)] = QuadUV(*uv[[0, 1, 3]].reshape(-1).tolist())
        _bleed(block, mask, pad)
        pixels[chart.y:chart.y + chart.height, chart.x:chart.x + chart.width] = block

    image = Image.fromarray(pixels)
    if alpha == "rgb" or (alpha == "auto" and bool(np.all(pixels[:, :, 3] == 255))):
        image = image.convert("RGB")
    result = AtlasBuildResult(width, height, _image_to_bytes(image), quad_uvs, model_uvs)
    return result, PaintLayout(width, height, texel_scale, pad, charts, mesher_result.quads_per_model)
