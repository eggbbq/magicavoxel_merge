"""Topology-preserving voxel mesher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .voxio import VoxScene


@dataclass(slots=True)
class Quad:
    """Represents a coplanar quad extracted from the voxel grid."""

    model_index: int
    origin: tuple[int, int, int]
    size_u: int
    size_v: int
    axis: int  # 0=x,1=y,2=z
    normal_sign: int  # +1 or -1 along axis
    colors: np.ndarray  # shape (size_v, size_u) palette indices
    verts: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    normal: tuple[float, float, float]


@dataclass(slots=True)
class MesherResult:
    quads_per_model: List[List[Quad]]


def build_quads(scene: VoxScene) -> MesherResult:
    """Generate Block-Topology Preservation quads for every model in a scene."""

    per_model: List[List[Quad]] = []

    for midx, model in enumerate(scene.models):
        voxels = model.voxels
        sx, sy, sz = model.size
        dims = (sx, sy, sz)
        model_quads: List[Quad] = []

        # Pad by 1 voxel on all sides so we can sample outside the model bounds
        # without per-sample bounds checks in Python.
        voxels_p = np.pad(voxels, 1, mode="constant", constant_values=0)

        for axis in range(3):
            u_axis = (axis + 1) % 3
            v_axis = (axis + 2) % 3

            mask = np.zeros((dims[u_axis], dims[v_axis]), dtype=np.int32)
            color_plane = np.zeros((dims[u_axis], dims[v_axis]), dtype=np.int32)
            cursor = [0, 0, 0]
            offset = [0, 0, 0]
            offset[axis] = 1

            for cursor[axis] in range(dims[axis] + 1):
                # Vectorized slice sampling for this plane.
                # We need a/b samples on either side of the plane along `axis`.
                c = int(cursor[axis])
                if axis == 0:
                    # Plane normal to X, mask axes are (Y,Z)
                    a_plane = voxels_p[c, 1 : sy + 1, 1 : sz + 1]
                    b_plane = voxels_p[c + 1, 1 : sy + 1, 1 : sz + 1]
                    # shape: (sy, sz) == (dims[u_axis], dims[v_axis])
                elif axis == 1:
                    # Plane normal to Y, mask axes are (Z,X)
                    a_plane = voxels_p[1 : sx + 1, c, 1 : sz + 1].T
                    b_plane = voxels_p[1 : sx + 1, c + 1, 1 : sz + 1].T
                    # shape: (sz, sx) == (dims[u_axis], dims[v_axis])
                else:
                    # Plane normal to Z, mask axes are (X,Y)
                    a_plane = voxels_p[1 : sx + 1, 1 : sy + 1, c]
                    b_plane = voxels_p[1 : sx + 1, 1 : sy + 1, c + 1]
                    # shape: (sx, sy) == (dims[u_axis], dims[v_axis])

                a_filled = a_plane != 0
                b_filled = b_plane != 0

                m = np.zeros_like(mask, dtype=np.int32)
                same = a_filled == b_filled
                m[~same & a_filled] = 1
                m[~same & ~a_filled] = -1
                mask[:, :] = m

                cp = np.zeros_like(color_plane, dtype=np.int32)
                cp[m == 1] = a_plane[m == 1]
                cp[m == -1] = b_plane[m == -1]
                color_plane[:, :] = cp

                u = 0
                while u < dims[u_axis]:
                    v = 0
                    while v < dims[v_axis]:
                        value = int(mask[u, v])
                        if value == 0:
                            v += 1
                            continue

                        width = 1
                        while u + width < dims[u_axis] and int(mask[u + width, v]) == value:
                            width += 1

                        height = 1
                        grow = True
                        while v + height < dims[v_axis] and grow:
                            for uu in range(width):
                                if int(mask[u + uu, v + height]) != value:
                                    grow = False
                                    break
                            if grow:
                                height += 1

                        origin = [0, 0, 0]
                        origin[axis] = cursor[axis]
                        origin[u_axis] = u
                        origin[v_axis] = v

                        normal_sign = 1 if value > 0 else -1
                        colors = color_plane[u : u + width, v : v + height].T.copy()

                        verts = _quad_vertices(origin, axis, width, height)
                        normal_vec = _axis_normal(axis, normal_sign)

                        model_quads.append(
                            Quad(
                                model_index=midx,
                                origin=tuple(int(val) for val in origin),
                                size_u=int(width),
                                size_v=int(height),
                                axis=axis,
                                normal_sign=normal_sign,
                                colors=colors,
                                verts=verts,
                                normal=normal_vec,
                            )
                        )

                        for vv in range(height):
                            for uu in range(width):
                                mask[u + uu, v + vv] = 0

                        v += height
                    u += 1

        per_model.append(model_quads)

    return MesherResult(quads_per_model=per_model)


def _sample(grid: np.ndarray, x: int, y: int, z: int) -> int:
    sx, sy, sz = grid.shape
    if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
        return int(grid[x, y, z])
    return 0


def _quad_vertices(origin: list[int], axis: int, size_u: int, size_v: int) -> tuple[tuple[float, float, float], ...]:
    u_axis = (axis + 1) % 3
    v_axis = (axis + 2) % 3

    p = [float(origin[0]), float(origin[1]), float(origin[2])]

    du = [0.0, 0.0, 0.0]
    dv = [0.0, 0.0, 0.0]
    du[u_axis] = float(size_u)
    dv[v_axis] = float(size_v)

    p0 = (p[0], p[1], p[2])
    p1 = (p[0] + du[0], p[1] + du[1], p[2] + du[2])
    p2 = (p1[0] + dv[0], p1[1] + dv[1], p1[2] + dv[2])
    p3 = (p[0] + dv[0], p[1] + dv[1], p[2] + dv[2])
    return (p0, p1, p2, p3)


def _axis_normal(axis: int, normal_sign: int) -> tuple[float, float, float]:
    n = [0.0, 0.0, 0.0]
    n[axis] = float(normal_sign)
    return (n[0], n[1], n[2])
