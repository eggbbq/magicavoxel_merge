#!/usr/bin/env python3
"""Add crack-style barycentric colors to a glb by shrinking boundary vertices per plane."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pygltflib import Accessor, BufferView, GLTF2

FLOAT = 5126
_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_WIDTH = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


def _read_accessor(gltf: GLTF2, blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    dtype = _COMPONENT_DTYPES[accessor.componentType]
    width = _TYPE_WIDTH[accessor.type]
    view = gltf.bufferViews[accessor.bufferView]
    base = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    length = accessor.count * width * dtype().nbytes
    data = blob[base : base + length]
    return np.frombuffer(data, dtype=dtype).reshape(accessor.count, width)


def _barycentric(pt: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v0 = b - a
    v1 = c - a
    v2 = pt - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    inv = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * inv
    w = (d00 * d21 - d01 * d20) * inv
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float32)


def _canonical_normal(normal: np.ndarray, plane_offset: float) -> tuple[np.ndarray, float]:
    n = normal.copy()
    off = plane_offset
    if (n[0] < 0) or (n[0] == 0 and n[1] < 0) or (n[0] == 0 and n[1] == 0 and n[2] < 0):
        n = -n
        off = -off
    return n, off


def _quantize(vec: Sequence[float], eps: float) -> tuple[int, ...]:
    scale = 1.0 / max(eps, 1e-12)
    return tuple(int(round(float(v) * scale)) for v in vec)


def _edges_of(face: Iterable[int]) -> list[tuple[int, int]]:
    verts = list(face)
    return [tuple(sorted((verts[i], verts[(i + 1) % len(verts)]))) for i in range(len(verts))]


def _process_primitive(
    positions: np.ndarray,
    indices: np.ndarray,
    shrink_factor: float,
    normal_eps: float,
    plane_eps: float,
) -> np.ndarray:
    vertex_count = positions.shape[0]
    colors = np.zeros((vertex_count, 4), dtype=np.float32)
    accum = np.zeros((vertex_count, 3), dtype=np.float64)
    counts = np.zeros(vertex_count, dtype=np.int32)

    faces = indices.reshape(-1, 3)
    face_normals = np.cross(positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]])
    norms = np.linalg.norm(face_normals, axis=1)
    valid = norms > 1e-9
    face_normals[valid] = (face_normals[valid].T / norms[valid]).T

    plane_to_tris: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for tri_idx, ok in enumerate(valid):
        if not ok:
            continue
        n = face_normals[tri_idx]
        p0 = positions[faces[tri_idx, 0]]
        d = float(np.dot(n, p0))
        n, d = _canonical_normal(n, d)
        plane_key = _quantize((n[0], n[1], n[2], d), normal_eps)
        plane_to_tris[plane_key].append(tri_idx)

    for plane_tris in plane_to_tris.values():
        if not plane_tris:
            continue
        adjacency: dict[int, set[int]] = {tri_idx: set() for tri_idx in plane_tris}
        for tri_idx in plane_tris:
            for edge in _edges_of(faces[tri_idx]):
                tris_for_edge = [idx for idx in plane_tris if idx != tri_idx and set(edge).issubset(faces[idx])]
                for other in tris_for_edge:
                    adjacency[tri_idx].add(other)
                    adjacency[other].add(tri_idx)

        visited: set[int] = set()
        for tri_idx in plane_tris:
            if tri_idx in visited:
                continue
            # flood fill component
            stack = [tri_idx]
            visited.add(tri_idx)
            component: list[int] = []
            component_edges: list[tuple[int, int]] = []
            vertices_in_comp: set[int] = set()
            while stack:
                cur = stack.pop()
                component.append(cur)
                verts = faces[cur]
                vertices_in_comp.update(verts)
                for edge in _edges_of(verts):
                    component_edges.append(edge)
                for nb in adjacency[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

            if not component:
                continue
            centroid = positions[list(vertices_in_comp)].mean(axis=0)

            for cur in component:
                tri = faces[cur]
                pts = positions[tri]
                for local_idx, v_idx in enumerate(tri):
                    shrink_pos = centroid + (positions[v_idx] - centroid) * shrink_factor
                    bary = _barycentric(shrink_pos, pts[0], pts[1], pts[2])
                    accum[v_idx] += bary
                    counts[v_idx] += 1

    mask = counts > 0
    if np.any(mask):
        averaged = (accum[mask] / counts[mask][:, None]).astype(np.float32)
        colors[mask, :3] = averaged
        colors[mask, 3] = 1.0
    return colors


def _append_color_accessor(gltf: GLTF2, blob: bytearray, color_data: np.ndarray) -> int:
    data = color_data.astype(np.float32).tobytes()
    offset = len(blob)
    blob.extend(data)
    pad = (4 - (len(blob) % 4)) % 4
    if pad:
        blob.extend(b"\x00" * pad)

    buffer_view_index = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data)))

    accessor_index = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            bufferView=buffer_view_index,
            componentType=FLOAT,
            count=int(color_data.shape[0]),
            type="VEC4",
        )
    )
    return accessor_index


def process_glb(
    input_path: Path,
    output_path: Path,
    shrink: float,
    normal_eps: float,
    plane_eps: float,
) -> None:
    gltf = GLTF2().load(str(input_path))
    blob = bytearray(gltf.binary_blob() or b"")
    if not gltf.buffers:
        gltf.buffers.append({"byteLength": len(blob)})

    stats: list[tuple[str, int, float, float, int]] = []

    for mesh_idx, mesh in enumerate(gltf.meshes or []):
        mesh_name = mesh.name or f"mesh_{mesh_idx}"
        for prim_idx, prim in enumerate(mesh.primitives or []):
            pos_index = getattr(prim.attributes, "POSITION", None)
            idx_index = prim.indices
            if pos_index is None or idx_index is None:
                continue
            positions = _read_accessor(gltf, blob, pos_index).astype(np.float32)
            indices = _read_accessor(gltf, blob, idx_index).astype(np.int64).reshape(-1)
            if positions.size == 0 or indices.size == 0:
                continue
            colors = _process_primitive(positions, indices, shrink, normal_eps, plane_eps)
            accessor_idx = _append_color_accessor(gltf, blob, colors)
            prim.attributes.COLOR_0 = accessor_idx

            mask = colors[:, 3] > 0.0
            if np.any(mask):
                vals = colors[mask, 0]
                stats.append((mesh_name, prim_idx, float(vals.min()), float(vals.max()), int(mask.sum())))
            else:
                stats.append((mesh_name, prim_idx, 0.0, 0.0, 0))

    if not gltf.buffers:
        gltf.buffers = []
    if gltf.buffers:
        gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(output_path))

    if stats:
        print("Barycentric COLOR_0 ranges per primitive:")
        for mesh_name, prim_idx, vmin, vmax, count in stats:
            print(
                f"  {mesh_name} prim#{prim_idx}: boundary={count} min={vmin:.4f} max={vmax:.4f}"
            )
    else:
        print("No mesh primitives processed; no COLOR_0 written.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Shrink GLB boundary faces and encode barycentric data in vertex colors")
    parser.add_argument("input", type=Path, help="Source .glb file")
    parser.add_argument("output", type=Path, help="Destination .glb file")
    parser.add_argument("--shrink", type=float, default=0.85, help="Shrink factor for boundary vertices (0-1)")
    parser.add_argument("--normal-eps", type=float, default=1e-3, help="Quantization step for plane normals")
    parser.add_argument("--plane-eps", type=float, default=1e-3, help="Quantization step for plane offsets")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if not (0.0 < args.shrink < 1.0):
        raise SystemExit("--shrink must be between 0 and 1")

    process_glb(args.input, args.output, args.shrink, args.normal_eps, args.plane_eps)
    print(f"Saved barycentric-colored GLB to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
