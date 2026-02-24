from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class VoxModel:
    size: tuple[int, int, int]
    voxels: np.ndarray  # (X, Y, Z) uint16, 0 = empty, 1..255 = palette index


@dataclass(frozen=True)
class VoxFile:
    models: list[VoxModel]
    palette_rgba: np.ndarray  # (256, 4) uint8
    model_names: list[str]
    model_translations: list[tuple[int, int, int]]


def _read_i32_from(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<i", buf, off)[0], off + 4


def _read_u32_from(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buf, off)[0], off + 4


def _read_str_from(buf: bytes, off: int) -> tuple[str, int]:
    ln, off = _read_u32_from(buf, off)
    s = buf[off : off + ln].decode("utf-8", errors="replace")
    return s, off + ln


def _read_dict_from(buf: bytes, off: int) -> tuple[dict[str, str], int]:
    n, off = _read_u32_from(buf, off)
    d: dict[str, str] = {}
    for _ in range(int(n)):
        k, off = _read_str_from(buf, off)
        v, off = _read_str_from(buf, off)
        d[k] = v
    return d, off


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected end of file")
    return b


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_chunk_header(f: BinaryIO) -> tuple[bytes, int, int]:
    chunk_id = _read_exact(f, 4)
    content_size = _read_u32(f)
    children_size = _read_u32(f)
    return chunk_id, content_size, children_size


def _default_palette_rgba() -> np.ndarray:
    # MagicaVoxel default palette (256 colors). Values are RGBA bytes.
    # Sourced from the public .vox specification and common reference implementations.
    values = [
        0x00000000,
        0xffffffff,
        0xffccffff,
        0xff99ffff,
        0xff66ffff,
        0xff33ffff,
        0xff00ffff,
        0xffffccff,
        0xffccccff,
        0xff99ccff,
        0xff66ccff,
        0xff33ccff,
        0xff00ccff,
        0xffff99ff,
        0xffcc99ff,
        0xff9999ff,
        0xff6699ff,
        0xff3399ff,
        0xff0099ff,
        0xffff66ff,
        0xffcc66ff,
        0xff9966ff,
        0xff6666ff,
        0xff3366ff,
        0xff0066ff,
        0xffff33ff,
        0xffcc33ff,
        0xff9933ff,
        0xff6633ff,
        0xff3333ff,
        0xff0033ff,
        0xffff00ff,
        0xffcc00ff,
        0xff9900ff,
        0xff6600ff,
        0xff3300ff,
        0xff0000ff,
        0xffffffcc,
        0xffccffcc,
        0xff99ffcc,
        0xff66ffcc,
        0xff33ffcc,
        0xff00ffcc,
        0xffffcccc,
        0xffcccccc,
        0xff99cccc,
        0xff66cccc,
        0xff33cccc,
        0xff00cccc,
        0xffff99cc,
        0xffcc99cc,
        0xff9999cc,
        0xff6699cc,
        0xff3399cc,
        0xff0099cc,
        0xffff66cc,
        0xffcc66cc,
        0xff9966cc,
        0xff6666cc,
        0xff3366cc,
        0xff0066cc,
        0xffff33cc,
        0xffcc33cc,
        0xff9933cc,
        0xff6633cc,
        0xff3333cc,
        0xff0033cc,
        0xffff00cc,
        0xffcc00cc,
        0xff9900cc,
        0xff6600cc,
        0xff3300cc,
        0xff0000cc,
        0xffffff99,
        0xffccff99,
        0xff99ff99,
        0xff66ff99,
        0xff33ff99,
        0xff00ff99,
        0xffffcc99,
        0xffcccc99,
        0xff99cc99,
        0xff66cc99,
        0xff33cc99,
        0xff00cc99,
        0xffff9999,
        0xffcc9999,
        0xff999999,
        0xff669999,
        0xff339999,
        0xff009999,
        0xffff6699,
        0xffcc6699,
        0xff996699,
        0xff666699,
        0xff336699,
        0xff006699,
        0xffff3399,
        0xffcc3399,
        0xff993399,
        0xff663399,
        0xff333399,
        0xff003399,
        0xffff0099,
        0xffcc0099,
        0xff990099,
        0xff660099,
        0xff330099,
        0xff000099,
        0xffffff66,
        0xffccff66,
        0xff99ff66,
        0xff66ff66,
        0xff33ff66,
        0xff00ff66,
        0xffffcc66,
        0xffcccc66,
        0xff99cc66,
        0xff66cc66,
        0xff33cc66,
        0xff00cc66,
        0xffff9966,
        0xffcc9966,
        0xff999966,
        0xff669966,
        0xff339966,
        0xff009966,
        0xffff6666,
        0xffcc6666,
        0xff996666,
        0xff666666,
        0xff336666,
        0xff006666,
        0xffff3366,
        0xffcc3366,
        0xff993366,
        0xff663366,
        0xff333366,
        0xff003366,
        0xffff0066,
        0xffcc0066,
        0xff990066,
        0xff660066,
        0xff330066,
        0xff000066,
        0xffffff33,
        0xffccff33,
        0xff99ff33,
        0xff66ff33,
        0xff33ff33,
        0xff00ff33,
        0xffffcc33,
        0xffcccc33,
        0xff99cc33,
        0xff66cc33,
        0xff33cc33,
        0xff00cc33,
        0xffff9933,
        0xffcc9933,
        0xff999933,
        0xff669933,
        0xff339933,
        0xff009933,
        0xffff6633,
        0xffcc6633,
        0xff996633,
        0xff666633,
        0xff336633,
        0xff006633,
        0xffff3333,
        0xffcc3333,
        0xff993333,
        0xff663333,
        0xff333333,
        0xff003333,
        0xffff0033,
        0xffcc0033,
        0xff990033,
        0xff660033,
        0xff330033,
        0xff000033,
        0xffffff00,
        0xffccff00,
        0xff99ff00,
        0xff66ff00,
        0xff33ff00,
        0xff00ff00,
        0xffffcc00,
        0xffcccc00,
        0xff99cc00,
        0xff66cc00,
        0xff33cc00,
        0xff00cc00,
        0xffff9900,
        0xffcc9900,
        0xff999900,
        0xff669900,
        0xff339900,
        0xff009900,
        0xffff6600,
        0xffcc6600,
        0xff996600,
        0xff666600,
        0xff336600,
        0xff006600,
        0xffff3300,
        0xffcc3300,
        0xff993300,
        0xff663300,
        0xff333300,
        0xff003300,
        0xffff0000,
        0xffcc0000,
        0xff990000,
        0xff660000,
        0xff330000,
        0xff0000ee,
        0xff0000dd,
        0xff0000bb,
        0xff0000aa,
        0xff000088,
        0xff000077,
        0xff000055,
        0xff000044,
        0xff000022,
        0xff000011,
        0xff00ee00,
        0xff00dd00,
        0xff00bb00,
        0xff00aa00,
        0xff008800,
        0xff007700,
        0xff005500,
        0xff004400,
        0xff002200,
        0xff001100,
        0xffeeee00,
        0xffdddd00,
        0xffbbbb00,
        0xffaaaa00,
        0xff888800,
        0xff777700,
        0xff555500,
        0xff444400,
        0xff222200,
        0xff111100,
        0xffee0000,
        0xffdd0000,
        0xffbb0000,
        0xffaa0000,
        0xff880000,
        0xff770000,
        0xff550000,
        0xff440000,
        0xff220000,
        0xff110000,
        0xffeeeeee,
        0xffdddddd,
        0xffbbbbbb,
        0xffaaaaaa,
        0xff888888,
        0xff777777,
        0xff555555,
        0xff444444,
        0xff222222,
        0xff111111,
    ]

    # Ensure 256 entries (the above is 256 for the canonical palette; if not, pad).
    if len(values) < 256:
        values = values + [0x00000000] * (256 - len(values))
    values = values[:256]

    arr = np.zeros((256, 4), dtype=np.uint8)
    for i, v in enumerate(values):
        # Stored as 0xAABBGGRR? In the reference palette list above it's 0xRRGGBBAA? 
        # MagicaVoxel RGBA chunk is R,G,B,A bytes.
        # The canonical palette list is usually stored as 0xAABBGGRR in little-endian references.
        # Here we decode assuming v is 0xAABBGGRR.
        rr = (v >> 0) & 0xFF
        gg = (v >> 8) & 0xFF
        bb = (v >> 16) & 0xFF
        aa = (v >> 24) & 0xFF
        arr[i] = (rr, gg, bb, aa)
    return arr


def load_vox(path: str) -> VoxFile:
    with open(path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != b"VOX ":
            raise ValueError("Not a VOX file")

        version = _read_u32(f)
        if version < 150:
            raise ValueError(f"Unsupported VOX version: {version}")

        chunk_id, content_size, children_size = _read_chunk_header(f)
        if chunk_id != b"MAIN":
            raise ValueError("Invalid VOX: missing MAIN chunk")
        if content_size != 0:
            f.seek(content_size, 1)

        end_of_main = f.tell() + children_size

        pack_expected: int | None = None
        current_size: tuple[int, int, int] | None = None
        current_voxels: np.ndarray | None = None
        models: list[VoxModel] = []
        model_names: list[str] = []
        palette = _default_palette_rgba()

        # Scene graph name mapping (best-effort)
        trn_child_to_name: dict[int, str] = {}
        trn_child_to_translation: dict[int, tuple[int, int, int]] = {}
        shp_node_to_model_ids: dict[int, list[int]] = {}

        while f.tell() < end_of_main:
            cid, csize, chsize = _read_chunk_header(f)
            content = _read_exact(f, csize)

            if cid == b"SIZE":
                if csize < 12:
                    raise ValueError("Invalid SIZE chunk")
                sx, sy, sz = struct.unpack("<III", content[:12])
                current_size = (int(sx), int(sy), int(sz))
                current_voxels = np.zeros(current_size, dtype=np.uint16)

            elif cid == b"PACK":
                if csize < 4:
                    raise ValueError("Invalid PACK chunk")
                pack_expected = int(struct.unpack("<I", content[:4])[0])

            elif cid == b"XYZI":
                if current_voxels is None or current_size is None:
                    raise ValueError("XYZI before SIZE")
                if csize < 4:
                    raise ValueError("Invalid XYZI chunk")
                n = struct.unpack("<I", content[:4])[0]
                expected = 4 + n * 4
                if csize < expected:
                    raise ValueError("Invalid XYZI chunk length")

                # Each voxel: x, y, z, colorIndex
                raw = np.frombuffer(content[4:expected], dtype=np.uint8)
                raw = raw.reshape((n, 4))
                xs = raw[:, 0].astype(np.int32)
                ys = raw[:, 1].astype(np.int32)
                zs = raw[:, 2].astype(np.int32)
                cs = raw[:, 3].astype(np.uint16)

                sx, sy, sz = current_size
                valid = (xs >= 0) & (xs < sx) & (ys >= 0) & (ys < sy) & (zs >= 0) & (zs < sz)
                xs = xs[valid]
                ys = ys[valid]
                zs = zs[valid]
                cs = cs[valid]

                current_voxels[xs, ys, zs] = cs

                models.append(VoxModel(size=current_size, voxels=current_voxels))
                model_names.append(f"model_{len(models) - 1}")
                current_size = None
                current_voxels = None

            elif cid == b"RGBA":
                if csize < 256 * 4:
                    raise ValueError("Invalid RGBA chunk")
                pal = np.frombuffer(content[: 256 * 4], dtype=np.uint8).reshape((256, 4))
                palette = pal.copy()

            elif cid == b"nTRN":
                off = 0
                node_id, off = _read_i32_from(content, off)
                node_dict, off = _read_dict_from(content, off)
                child_id, off = _read_i32_from(content, off)
                # reserved_id, layer_id
                _, off = _read_i32_from(content, off)
                _, off = _read_i32_from(content, off)
                num_frames, off = _read_i32_from(content, off)
                if num_frames > 0:
                    frame_dict, off = _read_dict_from(content, off)
                    t = frame_dict.get("_t")
                    if t:
                        parts = t.replace(",", " ").split()
                        if len(parts) >= 3:
                            try:
                                tx, ty, tz = int(parts[0]), int(parts[1]), int(parts[2])
                                trn_child_to_translation[int(child_id)] = (tx, ty, tz)
                            except ValueError:
                                pass

                name = node_dict.get("_name")
                if name:
                    trn_child_to_name[int(child_id)] = name

            elif cid == b"nSHP":
                off = 0
                node_id, off = _read_i32_from(content, off)
                _, off = _read_dict_from(content, off)
                num_models, off = _read_i32_from(content, off)
                mids: list[int] = []
                for _ in range(int(num_models)):
                    mid, off = _read_i32_from(content, off)
                    _, off = _read_dict_from(content, off)
                    mids.append(int(mid))
                shp_node_to_model_ids[int(node_id)] = mids

            # Skip children of this chunk (we don't currently parse transforms / multiple models)
            if chsize:
                f.seek(chsize, 1)

        # If we didn't parse any models, file is invalid.
        if not models:
            raise ValueError("Missing SIZE/XYZI in VOX")

        # If PACK exists, it's valid for the file to contain multiple SIZE/XYZI pairs.
        if pack_expected is not None and pack_expected != len(models):
            # Some exporters may not strictly match PACK count; keep best-effort.
            pass

        # Apply best-effort naming from scene graph:
        # If a transform node names a shape node, assign that name to the model ids referenced by the shape.
        model_translations: list[tuple[int, int, int]] = [(0, 0, 0)] * len(models)
        if trn_child_to_name and shp_node_to_model_ids:
            for shp_id, mids in shp_node_to_model_ids.items():
                name = trn_child_to_name.get(int(shp_id))
                if not name:
                    continue
                for mid in mids:
                    if 0 <= mid < len(model_names):
                        model_names[mid] = name

        # Apply best-effort translations from scene graph:
        if trn_child_to_translation and shp_node_to_model_ids:
            for shp_id, mids in shp_node_to_model_ids.items():
                tr = trn_child_to_translation.get(int(shp_id))
                if tr is None:
                    continue
                for mid in mids:
                    if 0 <= mid < len(model_translations):
                        model_translations[mid] = tr
        return VoxFile(models=models, palette_rgba=palette, model_names=model_names, model_translations=model_translations)
