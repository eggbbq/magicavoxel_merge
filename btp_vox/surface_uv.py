"""Rigid, integer-grid UV unfolding of axis-aligned voxel surfaces.

Join connected coplanar rectangles first, then fold adjacent panels out across
their shared edges. Only non-overlapping placements are accepted. Mesh geometry
and texel density never change; ambiguous non-manifold edges become seams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .btp_mesher import Quad


@dataclass(frozen=True)
class SurfaceJoint:
    left: int
    right: int
    a: tuple[int, int, int]
    b: tuple[int, int, int]


def surface_joints(quads: list[Quad]) -> list[SurfaceJoint]:
    """Find shared edge intervals, including greedy-mesh T junctions.

    Point contacts do not connect surfaces. Split at every endpoint so an
    ambiguous part of an edge cannot hide an otherwise valid connection.
    """
    lines: dict[tuple, list[tuple[int, int, int]]] = {}
    for qid, quad in enumerate(quads):
        vertices = np.asarray(quad.verts, dtype=np.int64)
        for i in range(4):
            a, b = vertices[i], vertices[(i + 1) % 4]
            axes = np.flatnonzero(a != b)
            if len(axes) != 1:
                raise ValueError("surface unwrap requires axis-aligned rectangular quads")
            axis = int(axes[0])
            fixed = tuple(int(a[j]) for j in range(3) if j != axis)
            lines.setdefault((axis, *fixed), []).append(
                (int(min(a[axis], b[axis])), int(max(a[axis], b[axis])), qid))

    joints = []
    for key, edges in sorted(lines.items()):
        axis, *fixed = key
        events: dict[int, list[tuple[int, int]]] = {}
        for low, high, qid in edges:
            events.setdefault(low, []).append((qid, 1))
            events.setdefault(high, []).append((qid, -1))
        active: set[int] = set()
        stops = sorted(events)
        for i, low in enumerate(stops[:-1]):
            for qid, delta in events[low]:
                if delta < 0:
                    active.discard(qid)
            for qid, delta in events[low]:
                if delta > 0:
                    active.add(qid)
            high = stops[i + 1]
            if len(active) != 2:
                continue
            left, right = sorted(active)
            a, b = list(fixed), list(fixed)
            a.insert(axis, low)
            b.insert(axis, high)
            joints.append(SurfaceJoint(left, right, tuple(a), tuple(b)))
    return joints


def map_surface_point(quad: Quad, pixels: np.ndarray, point) -> np.ndarray:
    """Evaluate a quad's affine map at a 3D point, in exact integer units."""
    vertices = np.asarray(quad.verts, dtype=np.int64)
    u = (vertices[1] - vertices[0]) // quad.size_u
    v = (vertices[3] - vertices[0]) // quad.size_v
    delta = np.asarray(point, dtype=np.int64) - vertices[0]
    return (pixels[0] + int(delta @ u) * ((pixels[1] - pixels[0]) // quad.size_u)
            + int(delta @ v) * ((pixels[3] - pixels[0]) // quad.size_v))


def _rectangles(points: dict[int, np.ndarray]) -> np.ndarray:
    arrays = np.asarray(list(points.values()))
    return np.concatenate((arrays.min(axis=1), arrays.max(axis=1)), axis=1)


def _bounds(rects: np.ndarray) -> np.ndarray:
    return np.r_[rects[:, :2].min(axis=0), rects[:, 2:].max(axis=0)]


def _overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.any(np.all(
        np.minimum(a[:, None, 2:], b[None, :, 2:])
        > np.maximum(a[:, None, :2], b[None, :, :2]), axis=2)))


def _cross(a, b) -> int:
    return int(a[0] * b[1] - a[1] * b[0])


_ROTATIONS = tuple(np.asarray(m, dtype=np.int64) for m in (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
))


def unfold_surface(quads: list[Quad]) -> list[dict[int, np.ndarray]]:
    """Return connected islands in voxel units; no mirrored outward faces.

    Panels can be concave or contain holes. Collision tests use their actual
    rectangles, not just bounding boxes. Stable tie-breaking makes output
    reproducible. This is a greedy seam heuristic, not an optimal net solver.
    """
    if not quads:
        return []
    joints = surface_joints(quads)
    roots = list(range(len(quads)))

    def root(qid):
        while roots[qid] != qid:
            roots[qid] = roots[roots[qid]]
            qid = roots[qid]
        return qid

    for joint in joints:
        left, right = quads[joint.left], quads[joint.right]
        if (left.axis, left.normal_sign) == (right.axis, right.normal_sign):
            a, b = root(joint.left), root(joint.right)
            roots[max(a, b)] = min(a, b)

    # Outward-oriented projections: rotating these frames never mirrors paint.
    frames = {
        (0, -1): ((0, 1, 0), (0, 0, -1)),
        (0, 1): ((0, -1, 0), (0, 0, -1)),
        (1, -1): ((-1, 0, 0), (0, 0, -1)),
        (1, 1): ((1, 0, 0), (0, 0, -1)),
        (2, -1): ((1, 0, 0), (0, -1, 0)),
        (2, 1): ((1, 0, 0), (0, 1, 0)),
    }
    panels: dict[int, dict[int, np.ndarray]] = {}
    for qid, quad in enumerate(quads):
        frame = np.asarray(frames[(quad.axis, quad.normal_sign)], dtype=np.int64)
        panels.setdefault(root(qid), {})[qid] = np.asarray(quad.verts, dtype=np.int64) @ frame.T
    rects = {pid: _rectangles(points) for pid, points in panels.items()}
    areas = {pid: int(np.prod(r[:, 2:] - r[:, :2], axis=1).sum()) for pid, r in rects.items()}
    neighbors: dict[int, list[tuple[int, SurfaceJoint, int, int]]] = {pid: [] for pid in panels}
    for joint in joints:
        a, b = root(joint.left), root(joint.right)
        if a != b:
            neighbors[a].append((b, joint, joint.left, joint.right))
            neighbors[b].append((a, joint, joint.right, joint.left))

    remaining = set(panels)
    seeds = iter(sorted(panels, key=lambda pid: (-areas[pid], pid)))
    islands = []
    while remaining:
        seed = next(pid for pid in seeds if pid in remaining)
        remaining.remove(seed)
        points = dict(panels[seed])
        occupied = rects[seed].copy()
        bounds = _bounds(occupied)
        frontier = list(neighbors[seed])
        # A hinge placement is fixed once its parent panel is placed. Cache it;
        # later iterations only recheck collisions and compactness.
        candidates = []
        seen_placements = set()
        while True:
            for child, joint, parent_qid, child_qid in frontier:
                if child not in remaining:
                    continue
                a = map_surface_point(quads[parent_qid], points[parent_qid], joint.a)
                b = map_surface_point(quads[parent_qid], points[parent_qid], joint.b)
                ca = map_surface_point(quads[child_qid], panels[child][child_qid], joint.a)
                cb = map_surface_point(quads[child_qid], panels[child][child_qid], joint.b)
                parent_side = _cross(b - a, points[parent_qid].sum(axis=0) - 4 * a)
                for rid, rotation in enumerate(_ROTATIONS):
                    if not np.array_equal((cb - ca) @ rotation.T, b - a):
                        continue
                    offset = a - ca @ rotation.T
                    child_center = panels[child][child_qid].sum(axis=0) @ rotation.T + 4 * offset
                    if parent_side * _cross(b - a, child_center - 4 * a) >= 0:
                        continue
                    key = (child, rid, *offset.tolist())
                    if key in seen_placements:
                        continue
                    seen_placements.add(key)
                    placed = {qid: p @ rotation.T + offset for qid, p in panels[child].items()}
                    r = _rectangles(placed)
                    length = int(np.abs(b - a).sum())
                    candidates.append((child, parent_qid, child_qid, length, placed, r, _bounds(r)))
            frontier = []
            best = None
            feasible = []
            for candidate in candidates:
                child, parent_qid, child_qid, length, placed, r, rb = candidate
                if child not in remaining or _overlaps(r, occupied):
                    continue
                feasible.append(candidate)
                nb = np.r_[np.minimum(bounds[:2], rb[:2]), np.maximum(bounds[2:], rb[2:])]
                growth = int(np.prod(nb[2:] - nb[:2]) - np.prod(bounds[2:] - bounds[:2]))
                score = (growth / areas[child] + 0.25 * np.sqrt(areas[child]) / length,
                         -length, -areas[child], child, parent_qid, child_qid)
                if best is None or score < best[0]:
                    best = (score, candidate, nb)
            if best is None:
                break
            _, (child, _, _, _, placed, r, _), bounds = best
            remaining.remove(child)
            points.update(placed)
            occupied = np.concatenate((occupied, r))
            frontier = neighbors[child]
            candidates = feasible
        islands.append(points)
    return islands
