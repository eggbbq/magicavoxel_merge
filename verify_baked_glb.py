#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from pygltflib import GLTF2


_TYPE_TO_COMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_accessor(gltf: GLTF2, blob: bytes, accessor_idx: int) -> np.ndarray:
    acc = gltf.accessors[accessor_idx]
    view = gltf.bufferViews[acc.bufferView]

    if acc.componentType != 5126:
        raise ValueError(
            f"Only FLOAT accessors supported by this verifier; got {acc.componentType}"
        )

    comp = _TYPE_TO_COMP[acc.type]
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    length = acc.count * comp * 4
    data = blob[offset : offset + length]
    return np.frombuffer(data, dtype=np.float32).reshape(acc.count, comp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", type=Path)
    ap.add_argument("--min-offset", type=float, default=512.0)
    ap.add_argument("--min-scale", type=float, default=1000.0)
    ap.add_argument("--size-scale", type=float, default=1000.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    gltf = GLTF2().load_binary(str(args.glb))
    blob = gltf.binary_blob() or b""

    for mesh_idx, mesh in enumerate(gltf.meshes or []):
        mesh_name = mesh.name or f"mesh_{mesh_idx}"
        for prim_idx, prim in enumerate(mesh.primitives or []):
            pos_idx = getattr(prim.attributes, "POSITION", None)
            col_idx = getattr(prim.attributes, "COLOR_0", None)
            tan_idx = getattr(prim.attributes, "TANGENT", None)

            if pos_idx is None:
                continue

            pos = _read_accessor(gltf, blob, pos_idx)
            col = _read_accessor(gltf, blob, col_idx) if col_idx is not None else None
            tan = _read_accessor(gltf, blob, tan_idx) if tan_idx is not None else None

            print(f"{mesh_name} prim#{prim_idx}:")
            print(f"  POSITION: {pos.shape}")
            if col is not None:
                print(f"  COLOR_0:  {col.shape}  min={col.min(axis=0)} max={col.max(axis=0)}")
            else:
                print("  COLOR_0:  <missing>")
            if tan is not None:
                print(f"  TANGENT:  {tan.shape}  min={tan.min(axis=0)} max={tan.max(axis=0)}")
            else:
                print("  TANGENT:  <missing>")

            if col is None or tan is None:
                continue

            # Our encoding layout:
            # COLOR_0.rgb = min.xyz / min_scale
            # COLOR_0.a   = size.x   / size_scale
            # TANGENT.xy  = size.yz  / size_scale
            decoded_min = col[:, 0:3] * float(args.min_scale) - float(args.min_offset)
            decoded_size = np.zeros((col.shape[0], 3), dtype=np.float32)
            decoded_size[:, 0] = col[:, 3] * float(args.size_scale)
            decoded_size[:, 1] = tan[:, 0] * float(args.size_scale)
            decoded_size[:, 2] = tan[:, 1] * float(args.size_scale)

            tri_count = col.shape[0] // 3
            max_tri = tri_count
            if args.limit and args.limit > 0:
                max_tri = min(max_tri, args.limit)

            for t in range(max_tri):
                v0 = t * 3
                v1 = v0 + 1
                v2 = v0 + 2
                # print one line per triangle (using v0's encoded payload)
                p0 = [round(float(x), 4) for x in pos[v0]]
                p1 = [round(float(x), 4) for x in pos[v1]]
                p2 = [round(float(x), 4) for x in pos[v2]]

                c0 = [round(float(x), 6) for x in col[v0]]
                tan0 = [round(float(x), 6) for x in tan[v0]]

                dmin = [int(round(float(x))) for x in decoded_min[v0]]
                dsz = [int(round(float(x))) for x in decoded_size[v0]]

                print(
                    f"  tri#{t}: p0={p0} p1={p1} p2={p2} | color={c0} tan={tan0} | decMin={dmin} decSize={dsz}"
                )

            # only dump first primitive by default
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
