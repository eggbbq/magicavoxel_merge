
#!/usr/bin/env python3
"""Rebuild GLB meshes so every quad stores wireframe metadata in COLOR_0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from pygltflib import Accessor, Buffer, BufferView, GLTF2

FLOAT = 5126
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125

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
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


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
    dtype = _COMPONENT_DTYPES[component_type]
    data = np.ascontiguousarray(array, dtype=dtype)
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
        accessor.min = list(float(v) for v in min_vals)
    if max_vals is not None:
        accessor.max = list(float(v) for v in max_vals)
    gltf.accessors.append(accessor)
    return accessor_index


def _encode_wireframe_colors(
    quad_vertices: np.ndarray, size_scale: float, axis_eps: float
) -> np.ndarray:
    total = quad_vertices.shape[0]
    colors = np.zeros((total, 4), dtype=np.float32)
    if total == 0:
        return colors

    for start in range(0, total, 6):
        chunk = quad_vertices[start : start + 6]
        if chunk.size == 0:
            break
        count = chunk.shape[0]
        center = chunk.mean(axis=0)
        min_p = chunk.min(axis=0)
        max_p = chunk.max(axis=0)
        size = max_p - min_p

        width = 0.0
        height = 0.0
        if size[0] > axis_eps and size[1] > axis_eps:
            width, height = size[0], size[1]
        elif size[0] > axis_eps and size[2] > axis_eps:
            width, height = size[0], size[2]
        else:
            width, height = size[1], size[2]

        encoded_w = float(np.clip(width / size_scale, 0.0, 1.0))
        encoded_h = float(np.clip(height / size_scale, 0.0, 1.0))

        for local_idx in range(count):
            dir_vec = chunk[local_idx] - center
            u = 0.0
            v = 0.0
            if abs(dir_vec[0]) > axis_eps and abs(dir_vec[1]) > axis_eps:
                u = 1.0 if dir_vec[0] > 0 else 0.0
                v = 1.0 if dir_vec[1] > 0 else 0.0
            elif abs(dir_vec[0]) > axis_eps and abs(dir_vec[2]) > axis_eps:
                u = 1.0 if dir_vec[0] > 0 else 0.0
                v = 1.0 if dir_vec[2] > 0 else 0.0
            else:
                u = 1.0 if dir_vec[1] > 0 else 0.0
                v = 1.0 if dir_vec[2] > 0 else 0.0

            colors[start + local_idx, 0] = u
            colors[start + local_idx, 1] = v
            colors[start + local_idx, 2] = encoded_w
            colors[start + local_idx, 3] = encoded_h

    return colors


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


def process_glb(
    input_path: Path,
    output_path: Path,
    *,
    size_scale: float,
    axis_eps: float,
) -> None:
    gltf = GLTF2().load(str(input_path))
    blob = bytearray(gltf.binary_blob() or b"")
    stats = []

    for mesh_idx, mesh in enumerate(gltf.meshes or []):
        mesh_name = mesh.name or f"mesh_{mesh_idx}"
        for prim_idx, prim in enumerate(mesh.primitives or []):
            if prim.indices is None:
                continue
            pos_accessor_idx = getattr(prim.attributes, "POSITION", None)
            if pos_accessor_idx is None:
                continue

            index_data = _read_accessor(gltf, blob, prim.indices).reshape(-1)
            if index_data.size == 0:
                continue

            positions = _read_accessor(gltf, blob, pos_accessor_idx)
            gathered_pos = positions[index_data]
            color_metadata = _encode_wireframe_colors(gathered_pos, size_scale, axis_eps)

            attr_indices = prim.attributes.__dict__.copy()
            duplicated_attrs = _duplicate_attributes(
                gltf, blob, index_data, attr_indices, blob
            )

            for attr_name, new_idx in duplicated_attrs.items():
                setattr(prim.attributes, attr_name, new_idx)

            color_accessor = _append_accessor(
                gltf, blob, color_metadata, component_type=FLOAT, accessor_type="VEC4"
            )
            prim.attributes.COLOR_0 = color_accessor

            new_vertex_count = index_data.size
            index_dtype = np.uint16 if new_vertex_count <= 65535 else np.uint32
            index_component = UNSIGNED_SHORT if new_vertex_count <= 65535 else UNSIGNED_INT
            new_indices = np.arange(new_vertex_count, dtype=index_dtype)
            index_accessor = _append_accessor(
                gltf,
                blob,
                new_indices.reshape(-1, 1),
                component_type=index_component,
                accessor_type="SCALAR",
            )
            prim.indices = index_accessor

            stats.append((mesh_name, prim_idx, new_vertex_count))

    if not gltf.buffers:
        gltf.buffers = [Buffer(byteLength=len(blob))]
    else:
        gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(output_path))

    if stats:
        print("Wireframe marking complete:")
        for mesh_name, prim_idx, vert_count in stats:
            quads = vert_count // 6
            print(
                f"  {mesh_name} prim#{prim_idx}: verts={vert_count} (approx {quads} quads)"
            )
    else:
        print("No primitives processed; output identical to input.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode quad wireframe metadata into COLOR_0 for each GLB primitive"
    )
    parser.add_argument("input", type=Path, help="Source .glb path")
    parser.add_argument("output", type=Path, help="Destination .glb path")
    parser.add_argument(
        "--size-scale",
        type=float,
        default=100.0,
        help="Physical units represented by COLOR_0.ba = 1.0",
    )
    parser.add_argument(
        "--axis-eps",
        type=float,
        default=1e-3,
        help="Threshold for detecting dominant axes",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input GLB not found: {args.input}")

    process_glb(args.input, args.output, size_scale=args.size_scale, axis_eps=args.axis_eps)
    print(f"Saved wireframe-marked GLB to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())