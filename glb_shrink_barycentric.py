#!/usr/bin/env python3
"""Add crack-style barycentric colors to a glb by shrinking boundary vertices per plane."""
from __future__ import annotations

import argparse
from collections import defaultdict
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


def _plane_axes_from_normal(normal: np.ndarray) -> tuple[int, int]:
    axis = int(np.argmax(np.abs(normal)))
    if axis == 0:
        return 1, 2  # normal is X → use Y/Z
    if axis == 1:
        return 0, 2  # normal is Y → use X/Z
    return 0, 1  # normal is Z → use X/Y


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
    axis_data = np.zeros((vertex_count, 4), dtype=np.float64)
    has_data = np.zeros(vertex_count, dtype=bool)

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
        n_key = _quantize((n[0], n[1], n[2]), normal_eps)
        d_scale = int(round(d / max(plane_eps, 1e-12)))
        plane_key = (*n_key, d_scale)
        plane_to_tris[plane_key].append(tri_idx)

    for plane_tris in plane_to_tris.values():
        if not plane_tris:
            continue
        adjacency: dict[int, set[int]] = {tri_idx: set() for tri_idx in plane_tris}
        edge_to_tris: dict[tuple[int, int], list[int]] = defaultdict(list)
        for tri_idx in plane_tris:
            for edge in _edges_of(faces[tri_idx]):
                edge_to_tris[edge].append(tri_idx)
        for edge_tris in edge_to_tris.values():
            if len(edge_tris) == 2:
                a, b = edge_tris
                adjacency[a].add(b)
                adjacency[b].add(a)

        visited: set[int] = set()
        for tri_idx in plane_tris:
            if tri_idx in visited:
                continue
            # flood fill component
            stack = [tri_idx]
            visited.add(tri_idx)
            component: list[int] = []
            vertices_in_comp: set[int] = set()
            while stack:
                cur = stack.pop()
                component.append(cur)
                verts = faces[cur]
                vertices_in_comp.update(verts)
                for nb in adjacency[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

            if not component:
                continue
            axis_u_idx, axis_v_idx = _plane_axes_from_normal(face_normals[component[0]])
            comp_vertices = np.array(sorted(vertices_in_comp))
            coords = positions[comp_vertices]
            u_vals = coords[:, axis_u_idx]
            v_vals = coords[:, axis_v_idx]
            u_min, u_max = u_vals.min(), u_vals.max()
            v_min, v_max = v_vals.min(), v_vals.max()
            center_u = 0.5 * (u_min + u_max)
            center_v = 0.5 * (v_min + v_max)
            half_u = max((u_max - u_min) * 0.5, 1e-6)
            half_v = max((v_max - v_min) * 0.5, 1e-6)

            for v_idx, u_val, v_val in zip(comp_vertices, u_vals, v_vals):
                axis_data[v_idx, 0] = u_val - center_u
                axis_data[v_idx, 1] = v_val - center_v
                axis_data[v_idx, 2] = half_u
                axis_data[v_idx, 3] = half_v
                has_data[v_idx] = True

    mask = has_data
    if np.any(mask):
        colors[mask] = axis_data[mask].astype(np.float32)
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
                uvals = colors[mask, 0]
                vvals = colors[mask, 1]
                stats.append(
                    (
                        mesh_name,
                        prim_idx,
                        float(uvals.min()),
                        float(uvals.max()),
                        float(vvals.min()),
                        float(vvals.max()),
                        int(mask.sum()),
                    )
                )
            else:
                stats.append((mesh_name, prim_idx, 0.0, 0.0, 0.0, 0.0, 0))

    if not gltf.buffers:
        gltf.buffers = []
    if gltf.buffers:
        gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(output_path))

    if stats:
        print("Plane UV COLOR_0 ranges per primitive:")
        for mesh_name, prim_idx, umin, umax, vmin, vmax, count in stats:
            print(
                f"  {mesh_name} prim#{prim_idx}: verts={count} u=[{umin:.4f}, {umax:.4f}] v=[{vmin:.4f}, {vmax:.4f}]"
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
