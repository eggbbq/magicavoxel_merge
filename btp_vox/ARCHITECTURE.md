# BTP Vox Pipeline Architecture

This document sketches the initial layout for the new Block-Topology Preservation (BTP) voxel converter.

## Goals
- Input: MagicaVoxel `.vox` files only.
- Output: `.glb` with baked atlas PNG plus companion UV JSON.
- Meshing: preserve block topology by only merging co-planar quads sharing the same facing; do not merge across corners/edges.
- Tooling: pure Python package with CLI entry point and a shell-based batch runner.

## Module Overview

| Module | Responsibility |
| --- | --- |
| `voxio` | Read MagicaVoxel files into in-memory `VoxScene` structures. Provides palette, per-model voxel grids, translations, names. |
| `btp_mesher` | Convert a `VoxScene` into block-topology-preserving quads. Ensures merges happen only across co-planar regions with identical normals. Emits per-quad metadata for UV packing. |
| `atlas` | Builds texture atlas from quad metadata. Handles padding, inset, texel scaling, and layout strategy (global or per-model). Outputs PNG bytes and UV rectangles. |
| `glb_writer` | Emit binary glTF scenes given mesh data + texture PNG. Wraps the internal glTF helpers (can re-use existing logic). |
| `uv_export` | Generate `{"width": W, "height": H, "ModelName": [u0,v0,u1,v1], ...}` JSON with one field per line. |
| `cli` | Command-line interface around the pipeline. Accepts input/output paths, atlas options, UV export path, etc. |
| `scripts/batch_convert.sh` | Convenience shell script for bulk conversion using the new CLI. |

## Data Flow
`voxio.VoxScene` → `btp_mesher.BtpMesher` → `atlas.AtlasBuilder` → `glb_writer.write_glb` + `uv_export.write_uv_json`.

Each stage returns structured data classes so later stages remain decoupled.

## Next Steps
1. Implement `voxio` parsing (lean on current repository helpers initially).
2. Implement BTP mesher skeleton with clear extension points for future heuristics.
3. Port atlas + GLB writer code to the new namespace.
4. Wire CLI + batch script to drive the pipeline end-to-end.
