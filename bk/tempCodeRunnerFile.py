#!/usr/bin/env python3
"""Bake quad bounding boxes into TEXCOORD_1 (uv2) of a GLB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from pygltflib import Accessor, Buffer, BufferView, GLTF2

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


@dataclass
class QuadBox:
    """Axis-aligned quad footprint extracted from two triangles."""

    min_corner: np.ndarray
    size: np.ndarray
    plane_axis: int
    plane_value: int
    normal_dir: int
    tri_indices: List[int]
    expand_neg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))
    expand_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))

    def expanded_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        expanded_min = self.min_corner - self.expand_neg
        expanded_size = self.size + self.expand_neg + self.expand_pos
        expanded_size = expanded_size.copy()
        expanded_size[self.plane_axis] = 0  # keep plane thickness zero
        return expanded_min, expanded_size


def _read_accessor(gltf: GLTF2, blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    dtype = _COMPONENT_DTYPES[accessor.componentType]
    width = _TYPE_WIDTH[accessor.type]
    view = gltf.bufferViews[accessor.bufferView]
    offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    length = accessor.count * width * dtype().nbytes
    raw = blob[offset : offset + length]
    return np.frombuffer(raw, dtype=dtype).reshape(accessor.count, width).copy()


def _append_accessor(
    gltf: GLTF2,
    blob: bytearray,
    array: np.ndarray,
    *,
    component_type: int,
    accessor_type: str,
    normalized: bool = False,
    min_vals: Iterable[float] | None = None,
    max_vals: Iterable[float] | None = None,
) -> int:
    data = np.ascontiguousarray(array)
    byte_data = data.tobytes()
    offset = len(blob)
    blob.extend(byte_data)
    pad = (4 - (len(blob) % 4)) % 4
    if pad:
        blob.extend(b"\x00" * pad)

    view_index = len(gltf.bufferViews)
    gltf.bufferViews.append(
        BufferView(buffer=0, byteOffset=offset, byteLength=len(byte_data))
    )

    accessor_index = len(gltf.accessors)
    accessor = Accessor(
        bufferView=view_index,
        componentType=component_type,
        count=int(data.shape[0]),
        type=accessor_type,
        normalized=normalized,
    )
    if min_vals is not None:
        accessor.min = [float(v) for v in min_vals]
    if max_vals is not None:
        accessor.max = [float(v) for v in max_vals]
    gltf.accessors.append(accessor)
    return accessor_index


def _duplicate_attributes(
    gltf: GLTF2,
    blob: bytes,
    indices: np.ndarray,
    attribute_map: Dict[str, int],
    blob_out: bytearray,
) -> Dict[str, int]:
    updated: Dict[str, int] = {}
    for attr_name, accessor_idx in attribute_map.items():
        if accessor_idx is None:
            continue
        accessor = gltf.accessors[accessor_idx]
        data = _read_accessor(gltf, blob, accessor_idx)
        duplicated = data[indices]
        min_vals = None
        max_vals = None
        if attr_name == "POSITION" and duplicated.size:
            min_vals = duplicated.min(axis=0)
            max_vals = duplicated.max(axis=0)
        new_accessor = _append_accessor(
            gltf,
            blob_out,
            duplicated,
            component_type=accessor.componentType,
            accessor_type=accessor.type,
            normalized=accessor.normalized,
            min_vals=min_vals,
            max_vals=max_vals,
        )
        updated[attr_name] = new_accessor
    return updated


def _interval_overlap(a_min: int, a_size: int, b_min: int, b_size: int) -> bool:
    a_max = a_min + a_size
    b_max = b_min + b_size
    return max(a_min, b_min) < min(a_max, b_max)


def _register_axis_growth(
    qa: QuadBox, qb: QuadBox, axis: int, expand_step: int
) -> None:
    a_min = qa.min_corner[axis]
    a_max = a_min + qa.size[axis]
    b_min = qb.min_corner[axis]
    b_max = b_min + qb.size[axis]
    if qa.size[axis] == 0 or qb.size[axis] == 0:
        return
    if a_max == b_min:
        qa.expand_pos[axis] = max(qa.expand_pos[axis], expand_step)
        qb.expand_neg[axis] = max(qb.expand_neg[axis], expand_step)
    elif b_max == a_min:
        qb.expand_pos[axis] = max(qb.expand_pos[axis], expand_step)
        qa.expand_neg[axis] = max(qa.expand_neg[axis], expand_step)


def _apply_plane_expansion(quads: List[QuadBox], expand_step: int) -> None:
    by_plane: Dict[Tuple[int, int], List[QuadBox]] = {}
    for quad in quads:
        by_plane.setdefault((quad.plane_axis, quad.plane_value), []).append(quad)

    for (plane_axis, _), plane_quads in by_plane.items():
        in_plane_axes = [axis for axis in range(3) if axis != plane_axis]
        axis_u, axis_v = in_plane_axes
        for i in range(len(plane_quads)):
            qa = plane_quads[i]
            for j in range(i + 1, len(plane_quads)):
                qb = plane_quads[j]
                if _interval_overlap(
                    qa.min_corner[axis_v], qa.size[axis_v], qb.min_corner[axis_v], qb.size[axis_v]
                ):
                    _register_axis_growth(qa, qb, axis_u, expand_step)
                if _interval_overlap(
                    qa.min_corner[axis_u], qa.size[axis_u], qb.min_corner[axis_u], qb.size[axis_u]
                ):
                    _register_axis_growth(qa, qb, axis_v, expand_step)


def _collect_quads(
    int_positions: np.ndarray,
) -> Tuple[List[QuadBox], np.ndarray]:
    tri_count = int_positions.shape[0] // 3
    tri_to_quad = np.full(tri_count, -1, dtype=np.int32)
    quads: List[QuadBox] = []
    key_to_quad: Dict[Tuple[int, ...], int] = {}

    for tri_idx in range(tri_count):
        tri_pts = int_positions[tri_idx * 3 : (tri_idx + 1) * 3]
        tri_min = tri_pts.min(axis=0)
        tri_max = tri_pts.max(axis=0)
        tri_size = tri_max - tri_min
        zero_axes = np.where(tri_size == 0)[0]
        if zero_axes.size != 1:
            continue
        plane_axis = int(zero_axes[0])
        plane_value = int(tri_min[plane_axis])
        # determine normal direction along plane axis
        edge1 = tri_pts[1] - tri_pts[0]
        edge2 = tri_pts[2] - tri_pts[0]
        normal = np.cross(edge1, edge2)
        axis_component = normal[plane_axis]
        normal_dir = 1 if axis_component >= 0 else -1
        key = (
            int(tri_min[0]),
            int(tri_min[1]),
            int(tri_min[2]),
            int(tri_max[0]),
            int(tri_max[1]),
            int(tri_max[2]),
        )
        quad_index = key_to_quad.get(key)
        if quad_index is None:
            quad_index = len(quads)
            quads.append(
                QuadBox(
                    min_corner=tri_min.astype(np.int32),
                    size=tri_size.astype(np.int32),
                    plane_axis=plane_axis,
                    plane_value=plane_value,
                    normal_dir=normal_dir,
                    tri_indices=[tri_idx],
                )
            )
            key_to_quad[key] = quad_index
        else:
            quads[quad_index].tri_indices.append(tri_idx)
        tri_to_quad[tri_idx] = quad_index

    return quads, tri_to_quad


def process_glb(
    input_path: Path,
    output_path: Path,
    *,
    scale_factor: float,
    expand_step: int,
    min_offset: int,
    min_scale: float,
    size_scale: float,
) -> None:

    gltf = GLTF2().load(str(input_path))
    blob = bytearray(gltf.binary_blob() or b"")
    stats = []
    quad_logs: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []

    for mesh_idx, mesh in enumerate(gltf.meshes or []):
        mesh_name = mesh.name or f"mesh_{mesh_idx}"
        for prim_idx, prim in enumerate(mesh.primitives or []):
            if prim.indices is None:
                continue
            pos_accessor_idx = getattr(prim.attributes, "POSITION", None)
            if pos_accessor_idx is None:
                continue

            index_data = _read_accessor(gltf, blob, prim.indices).reshape(-1)
            if index_data.size == 0 or index_data.size % 3:
                continue

            positions = _read_accessor(gltf, blob, pos_accessor_idx).astype(np.float64)
            gathered_pos = positions[index_data]
            scaled = np.round(gathered_pos / scale_factor).astype(np.int32)

            quads, tri_to_quad = _collect_quads(scaled)
            if not quads:
                continue

            _apply_plane_expansion(quads, expand_step)

            quad_mins = []
            quad_sizes = []
            for quad in quads:
                expanded_min, expanded_size = quad.expanded_bounds()
                quad_mins.append(expanded_min + min_offset)
                quad_sizes.append(expanded_size)
            quad_mins = np.array(quad_mins, dtype=np.int32)
            quad_sizes = np.array(quad_sizes, dtype=np.int32)

            min_norm = quad_mins.astype(np.float32) / float(min_scale)
            size_norm = quad_sizes.astype(np.float32) / float(size_scale)

            colors = np.zeros((index_data.size, 4), dtype=np.float32)
            uv2 = np.zeros((index_data.size, 2), dtype=np.float32)
            tri_count = index_data.size // 3
            for tri_idx in range(tri_count):
                quad_idx = tri_to_quad[tri_idx]
                if quad_idx < 0:
                    continue
                start = tri_idx * 3
                min_vec = min_norm[quad_idx]
                size_vec = size_norm[quad_idx]
                plane_axis = float(quads[quad_idx].plane_axis)
                normal_dir = float(quads[quad_idx].normal_dir)

                colors[start : start + 3, 0:3] = min_vec
                colors[start : start + 3, 3] = size_vec[0]
                uv2[start : start + 3, 0] = size_vec[1]
                uv2[start : start + 3, 1] = size_vec[2]

            attr_indices = prim.attributes.__dict__.copy()
            duplicated_attrs = _duplicate_attributes(
                gltf, blob, index_data, attr_indices, blob
            )
            for attr_name, new_idx in duplicated_attrs.items():
                setattr(prim.attributes, attr_name, new_idx)

            color_encoded = np.clip(colors, 0.0, 1.0)
            uv2_encoded = np.clip(uv2, 0.0, 1.0)

            color_accessor = _append_accessor(
                gltf, blob, color_encoded, component_type=FLOAT, accessor_type="VEC4"
            )
            prim.attributes.COLOR_0 = color_accessor

            uv2_accessor = _append_accessor(
                gltf, blob, uv2_encoded, component_type=FLOAT, accessor_type="VEC2"
            )
            prim.attributes.TEXCOORD_1 = uv2_accessor

            new_vertex_count = index_data.size
            new_indices = np.arange(new_vertex_count, dtype=np.uint32)
            index_accessor = _append_accessor(
                gltf,
                blob,
                new_indices.reshape(-1, 1),
                component_type=5125,
                accessor_type="SCALAR",
            )
            prim.indices = index_accessor

            normal_vectors = []
            for quad in quads:
                normal_vec = np.zeros(3, dtype=np.int32)
                normal_vec[quad.plane_axis] = quad.normal_dir
                normal_vectors.append(normal_vec)
            quad_normals = np.array(normal_vectors, dtype=np.int32)

            stats.append((mesh_name, prim_idx, len(quads), tri_count))
            quad_logs.append(
                (mesh_name, prim_idx, quad_mins.copy(), quad_sizes.copy(), quad_normals)
            )

    if not gltf.buffers:
        gltf.buffers = [Buffer(byteLength=len(blob))]
    else:
        gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(output_path))

    if stats:
        print("UV2 quad bake summary:")
        for mesh_name, prim_idx, quad_count, tri_count in stats:
            print(
                f"  {mesh_name} prim#{prim_idx}: quads={quad_count} triangles={tri_count}"
            )

        print("\nQuad bounding boxes (integers after scaling & expansion):")
        for mesh_name, prim_idx, quad_mins, quad_sizes, quad_normals in quad_logs:
            for q_idx, (qmin, qsize, qnormal) in enumerate(
                zip(quad_mins, quad_sizes, quad_normals)
            ):
                qmin_list = qmin.astype(int).tolist()
                qsize_list = qsize.astype(int).tolist()
                qnormal_list = qnormal.astype(int).tolist()
                print(
                    f"  {mesh_name} prim#{prim_idx} quad#{q_idx}: min={qmin_list} size={qsize_list} normal={qnormal_list}"
                )
    else:
        print("No primitives processed; GLB unchanged.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bake quad bounding boxes into TEXCOORD_1"
    )
    parser.add_argument("input", type=Path, help="Source .glb path")
    parser.add_argument("output", type=Path, help="Destination .glb path")
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Scale divisor applied before rounding vertices to the voxel grid",
    )
    parser.add_argument(
        "--expand-step",
        type=int,
        default=10,
        help="Amount (grid units) to grow touching quads along shared edges",
    )
    parser.add_argument(
        "--min-offset",
        type=int,
        default=512,
        help="Offset added to quad min so shader space stays positive",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=1000.0,
        help="Divisor used to normalize min coordinates into 0-1",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=1000.0,
        help="Divisor used to normalize size axes into 0-1",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input GLB not found: {args.input}")

    process_glb(
        args.input,
        args.output,
        scale_factor=args.scale_factor,
        expand_step=args.expand_step,
        min_offset=args.min_offset,
        min_scale=args.min_scale,
        size_scale=args.size_scale,
    )
    print(f"Saved UV2 wireframe bake to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())