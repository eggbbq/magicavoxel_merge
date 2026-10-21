"""Repack a painted glTF/GLB using only its mesh UVs and final colour image.

This is a separate publishing step: it never reloads VOX or regenerates colours.
Coplanar triangles with continuous UVs form patches. Cutting along geometric
creases breaks an editing box net into compact, independently packable faces.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
from PIL import Image
from pygltflib import Accessor, Buffer, BufferView, GLTF2

from .atlas import _image_to_bytes, _next_pow2, _pack_rects_best, _pack_maxrect


_DTYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _uri_bytes(uri: str, directory: Path) -> bytes:
    if uri.startswith("data:"):
        header, data = uri.split(",", 1)
        return base64.b64decode(data) if ";base64" in header else unquote(data).encode("latin1")
    if "://" in uri:
        raise ValueError("Repacking requires local model/image resources")
    return (directory / unquote(uri)).read_bytes()


class _Document:
    def __init__(self, path: Path):
        self.gltf = GLTF2().load(str(path))
        self.directory = path.parent
        self.data = bytearray()
        offsets = []
        for buffer in self.gltf.buffers:
            self._align()
            offsets.append(len(self.data))
            blob = (_uri_bytes(buffer.uri, path.parent) if buffer.uri else self.gltf.binary_blob())
            if blob is None:
                raise ValueError("Missing model buffer")
            self.data.extend(blob[:buffer.byteLength])
        for view in self.gltf.bufferViews:
            if view.extensions:
                raise ValueError("Compressed/extended buffer views must be decoded before repacking")
            view.byteOffset = (view.byteOffset or 0) + offsets[view.buffer]
            view.buffer = 0

    def _align(self):
        self.data.extend(b"\0" * (-len(self.data) % 4))

    def view(self, index: int) -> bytes:
        view = self.gltf.bufferViews[index]
        offset = view.byteOffset or 0
        return bytes(self.data[offset:offset + view.byteLength])

    def accessor(self, index: int) -> np.ndarray:
        a = self.gltf.accessors[index]
        if a.type not in _SIZES:
            raise ValueError(f"Unsupported vertex attribute type: {a.type}")
        dtype = np.dtype(_DTYPES[a.componentType])
        size = _SIZES[a.type]
        if a.bufferView is None:
            values = np.zeros((a.count, size), dtype=dtype)
        else:
            view = self.gltf.bufferViews[a.bufferView]
            values = np.ndarray((a.count, size), dtype=dtype, buffer=self.data,
                                offset=(view.byteOffset or 0) + (a.byteOffset or 0),
                                strides=(view.byteStride or dtype.itemsize * size, dtype.itemsize)).copy()
        if a.sparse and a.sparse.count:
            sparse = a.sparse
            ids = np.frombuffer(self.view(sparse.indices.bufferView),
                                dtype=_DTYPES[sparse.indices.componentType], count=sparse.count,
                                offset=sparse.indices.byteOffset or 0)
            entries = np.frombuffer(self.view(sparse.values.bufferView), dtype=dtype,
                                    count=sparse.count * size, offset=sparse.values.byteOffset or 0)
            values[ids] = entries.reshape(-1, size)
        return values

    def floats(self, index: int) -> np.ndarray:
        values = self.accessor(index)
        a = self.gltf.accessors[index]
        if a.normalized and values.dtype.kind in "iu":
            values = np.maximum(values.astype(np.float64) / np.iinfo(values.dtype).max, -1)
        return values.astype(np.float64)

    def add_view(self, data: bytes, target: int | None = None) -> int:
        self._align()
        index = len(self.gltf.bufferViews)
        self.gltf.bufferViews.append(BufferView(buffer=0, byteOffset=len(self.data),
                                              byteLength=len(data), target=target))
        self.data.extend(data)
        return index

    def add_accessor(self, values: np.ndarray, source: int | None = None,
                     *, indices: bool = False) -> int:
        if source is None:
            a = Accessor(componentType=5125 if indices else 5126,
                         type="SCALAR" if indices else "VEC2")
        else:
            a = deepcopy(self.gltf.accessors[source])
        a.bufferView = self.add_view(np.ascontiguousarray(values).tobytes(), 34963 if indices else 34962)
        a.byteOffset = 0
        a.count = len(values)
        a.sparse = None
        if a.min:
            a.min = values.min(axis=0).tolist()
        if a.max:
            a.max = values.max(axis=0).tolist()
        index = len(self.gltf.accessors)
        self.gltf.accessors.append(a)
        return index

    def image(self, index: int) -> bytes:
        image = self.gltf.images[index]
        return _uri_bytes(image.uri, self.directory) if image.uri else self.view(image.bufferView)

    def compact(self) -> None:
        """Remove replaced attribute/image payloads; keep skin and animation data."""
        refs = []
        for mesh in self.gltf.meshes:
            for primitive in mesh.primitives:
                refs.extend((vars(primitive.attributes), k) for k, v in vars(primitive.attributes).items() if v is not None)
                if primitive.indices is not None:
                    refs.append((vars(primitive), "indices"))
                for target in primitive.targets or []:
                    fields = target if isinstance(target, dict) else vars(target)
                    refs.extend((fields, k) for k, v in fields.items() if v is not None)
        for skin in self.gltf.skins:
            if skin.inverseBindMatrices is not None:
                refs.append((vars(skin), "inverseBindMatrices"))
        for animation in self.gltf.animations:
            for sampler in animation.samplers:
                refs.extend([(vars(sampler), "input"), (vars(sampler), "output")])
        used = sorted({fields[key] for fields, key in refs})
        mapping = {old: new for new, old in enumerate(used)}
        for fields, key in refs:
            fields[key] = mapping[fields[key]]
        self.gltf.accessors = [self.gltf.accessors[i] for i in used]

        view_refs = []
        for a in self.gltf.accessors:
            if a.bufferView is not None:
                view_refs.append((vars(a), "bufferView"))
            if a.sparse:
                view_refs.extend([(vars(a.sparse.indices), "bufferView"),
                                  (vars(a.sparse.values), "bufferView")])
        for image in self.gltf.images:
            if image.bufferView is not None:
                view_refs.append((vars(image), "bufferView"))
        used = sorted({fields[key] for fields, key in view_refs})
        mapping = {old: new for new, old in enumerate(used)}
        packed = bytearray()
        views = []
        for old in used:
            packed.extend(b"\0" * (-len(packed) % 4))
            view = deepcopy(self.gltf.bufferViews[old])
            raw = self.view(old)
            view.buffer, view.byteOffset = 0, len(packed)
            packed.extend(raw)
            views.append(view)
        for fields, key in view_refs:
            fields[key] = mapping[fields[key]]
        self.gltf.bufferViews = views
        self.gltf.buffers = [Buffer(byteLength=len(packed))]
        self.gltf.set_binary_blob(bytes(packed))


@dataclass
class _Patch:
    triangles: list[int]
    low: np.ndarray
    pixels: np.ndarray
    block: np.ndarray
    owner: int = -1
    x: int = 0
    y: int = 0
    transform: tuple[bool, bool, bool] = (False, False, False)
    mean_error: float = 0.0
    max_error: int = 0
    compressed_solid: bool = False


_TRANSFORMS = [(swap, flip_u, flip_v) for swap in (False, True)
               for flip_u, flip_v in ((False, False), (True, False), (False, True), (True, True))]


def _transform_block(block: np.ndarray, transform: tuple[bool, bool, bool]) -> np.ndarray:
    swap, flip_u, flip_v = transform
    result = block.transpose(1, 0, 2) if swap else block
    if flip_u:
        result = result[:, ::-1]
    if flip_v:
        result = result[::-1]
    return result


def _assign_owners(patches: list[_Patch], *, deduplicate: bool, merge_similar: bool,
                   threshold: float, max_error: int, pad: int) -> list[tuple[int, int, int]]:
    """Compare every child directly to an unchanged representative, never a chain.

    Similarity is 1 - RGB MAE / 255. Alpha must match exactly, including gutters;
    the peak RGB error bound protects small details that a mean can hide.
    """
    exact = {}
    owners_by_shape: dict[tuple, list[int]] = {}
    means: dict[int, np.ndarray] = {}
    rects = []
    budget = (1.0 - threshold) * 255.0
    for index, patch in enumerate(patches):
        patch.transform = (False, False, False)
        patch.mean_error, patch.max_error = 0.0, 0
        variants = [(t, _transform_block(patch.block, t))
                    for t in (_TRANSFORMS if merge_similar else [(False, False, False)])]
        owner = None
        if deduplicate or merge_similar:
            for transform, block in variants:
                hit = exact.get((block.shape, block.tobytes()))
                if hit is not None:
                    owner, patch.transform = hit, transform
                    break
        if owner is None and merge_similar and budget > 0 and max_error > 0:
            average = patch.block[:, :, :3].mean(axis=(0, 1))
            best = None
            for order, (transform, block) in enumerate(variants):
                for candidate in owners_by_shape.get(block.shape, []):
                    # Channel means give a cheap lower bound on full RGB MAE.
                    lower = np.abs(average - means[candidate])
                    if float(lower.mean()) > budget + 1e-12 or float(lower.max()) > max_error:
                        continue
                    representative = patches[candidate].block
                    if not np.array_equal(block[:, :, 3], representative[:, :, 3]):
                        continue
                    delta = np.abs(block[:, :, :3].astype(np.int16) - representative[:, :, :3].astype(np.int16))
                    peak = int(delta.max())
                    if peak > max_error:
                        continue
                    mean = float(delta.mean())
                    if mean > budget + 1e-12:
                        continue
                    score = (mean, peak, candidate, order)
                    if best is None or score < best:
                        best = score
            if best is not None:
                patch.mean_error, patch.max_error, owner, order = best
                patch.transform = variants[order][0]
        if owner is None:
            owner = index
            exact[(patch.block.shape, patch.block.tobytes())] = index
            if merge_similar:
                owners_by_shape.setdefault(patch.block.shape, []).append(index)
                means[index] = patch.block[:, :, :3].mean(axis=(0, 1))
            height, width = patch.pixels.shape[:2]
            rects.append((index, width + 2 * pad, height + 2 * pad))
        patch.owner = owner
    return rects


def _mapped_pixels(patch: _Patch, owner: _Patch, points: np.ndarray) -> np.ndarray:
    if patch.compressed_solid:
        return np.zeros_like(points, dtype=np.float64) + (patch.x + 0.5, patch.y + 0.5)
    local = points - patch.low
    if patch.transform != (False, False, False):
        local = local / (patch.pixels.shape[1], patch.pixels.shape[0])
        swap, flip_u, flip_v = patch.transform
        if swap:
            local = local[..., ::-1]
        if flip_u:
            local[..., 0] = 1 - local[..., 0]
        if flip_v:
            local[..., 1] = 1 - local[..., 1]
        local = local * (owner.pixels.shape[1], owner.pixels.shape[0])
    return local + (patch.x, patch.y)


def _pack_dense(rects: list[tuple[int, int, int]], *, pot: bool):
    """Try MaxRects widths/orders, retaining the shelf result as an upper bound."""
    w, h, positions = _pack_rects_best(rects, pot=pot, square=False)
    if pot:
        w, h = _next_pow2(w), _next_pow2(h)
    best = (w, h, positions)
    def score(width, height):
        return width * height, max(width, height), abs(width - height)
    if not rects:
        return best
    area = sum(rw * rh for _, rw, rh in rects)
    min_w, min_h = max(rw for _, rw, _ in rects), max(rh for _, _, rh in rects)
    root = math.sqrt(area)
    widths = {w, min_w, *[max(min_w, math.ceil(root * factor))
                         for factor in (.5, .75, 1, 1.25, 1.5, 2)]}
    widths.update(1 << k for k in range((min_w - 1).bit_length(), (w * h // min_h).bit_length() + 1))
    if pot:
        widths = {_next_pow2(width) for width in widths}
    orders = [sorted(rects, key=key, reverse=True) for key in (
        lambda r: (r[1] * r[2], max(r[1:]), r[2], r[1]),
        lambda r: (r[2], r[1]), lambda r: (max(r[1:]), min(r[1:]), r[2]))]
    for width in sorted(widths):
        low = max(min_h, math.ceil(area / width))
        upper = min(sum(rh for _, _, rh in rects), best[0] * best[1] // width)
        if low > upper:
            continue
        for order in orders:
            positions = _pack_maxrect(order, width, upper)
            if positions is None:
                continue
            lo, hi = low, upper
            while lo < hi:
                mid = (lo + hi) // 2
                candidate = _pack_maxrect(order, width, mid)
                if candidate is None:
                    lo = mid + 1
                else:
                    hi, positions = mid, candidate
            used_w = max(positions[i][0] + rw for i, rw, _ in rects)
            used_h = max(positions[i][1] + rh for i, _, rh in rects)
            if pot:
                used_w, used_h = _next_pow2(used_w), _next_pow2(used_h)
            if score(used_w, used_h) < score(best[0], best[1]):
                best = used_w, used_h, positions
    return best


def _wrapped(indices: np.ndarray, size: int, mode: int) -> np.ndarray:
    if mode == 33071:  # CLAMP_TO_EDGE
        return np.clip(indices, 0, size - 1)
    if mode == 10497:  # REPEAT
        return indices % size
    if mode == 33648:  # MIRRORED_REPEAT
        values = indices % (2 * size)
        return np.where(values < size, values, 2 * size - 1 - values)
    raise ValueError(f"Unsupported texture wrap mode: {mode}")


def _patches(positions: np.ndarray, uv: np.ndarray, triangles: np.ndarray,
             pixels: np.ndarray, pad: int, wrap_s: int, wrap_t: int) -> list[_Patch]:
    """Split at geometric creases and UV seams, independent of vertex sharing."""
    count = len(triangles)
    parent = list(range(count))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    points = positions[triangles]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-30)
    tolerance = max(float(np.ptp(positions, axis=0).max()) * 1e-6, 1e-12)
    edges = {}
    for tid, triangle in enumerate(triangles):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted(tuple(positions[v]) + tuple(uv[v]) for v in (triangle[a], triangle[b])))
            for other in edges.get(key, []):
                if (lengths[tid] > 0 and lengths[other] > 0
                        and np.dot(normals[tid], normals[other]) > 1 - 1e-6
                        and abs(np.dot(normals[tid], points[other, 0] - points[tid, 0])) <= tolerance):
                    parent[find(tid)] = find(other)
            edges.setdefault(key, []).append(tid)
    groups = {}
    for tid in range(count):
        groups.setdefault(find(tid), []).append(tid)
    height, width = pixels.shape[:2]
    result = []
    for group in groups.values():
        coords = uv[np.unique(triangles[group])] * (width, height)
        # Float32 UVs can land a few millionths of a pixel off an integer edge.
        low = np.floor(coords.min(axis=0) + 1e-4).astype(int)
        high = np.ceil(coords.max(axis=0) - 1e-4).astype(int)
        low = np.clip(low, (0, 0), (width - 1, height - 1))
        high = np.clip(np.maximum(high, low + 1), (1, 1), (width, height))
        x0, y0 = low
        x1, y1 = high
        # Copy the actual source neighbourhood, including across editing-net
        # seams, so linear filtering retains the original edge colours.
        xs = _wrapped(np.arange(x0 - pad, x1 + pad), width, wrap_s)
        ys = _wrapped(np.arange(y0 - pad, y1 + pad), height, wrap_t)
        block = pixels[ys[:, None], xs[None, :]].copy()
        result.append(_Patch(group, low, pixels[y0:y1, x0:x1].copy(), block))
    return result


def _material_uv(gltf, primitive):
    if primitive.material is None:
        return None
    material = gltf.materials[primitive.material]
    pbr = material.pbrMetallicRoughness
    info = pbr.baseColorTexture if pbr else None
    if (material.normalTexture or material.occlusionTexture or material.emissiveTexture
            or (pbr and pbr.metallicRoughnessTexture)
            or set(material.extensions or {}) - {"KHR_materials_unlit"}):
        raise ValueError("Repacking currently supports base-colour textures only; bake other maps into colour first")
    if info is None:
        return None
    if info.extensions:
        raise ValueError("Bake texture transforms before repacking")
    texture = gltf.textures[info.index]
    if texture.extensions or texture.source is None:
        raise ValueError("Repacking requires an ordinary PNG/JPEG colour texture")
    sampler = gltf.samplers[texture.sampler] if texture.sampler is not None else None
    return (texture.source, f"TEXCOORD_{info.texCoord or 0}",
            sampler.wrapS if sampler and sampler.wrapS is not None else 10497,
            sampler.wrapT if sampler and sampler.wrapT is not None else 10497)


def repack(input_model: str | Path, output_model: str | Path, *,
           texture: str | Path | None = None, texture_out: str | Path | None = None,
           pad: int = 2, pot: bool = False, deduplicate: bool = False,
           merge_similar: bool = False, similarity_threshold: float = 0.98,
           similarity_max_error: int = 16,
           compress_solid: bool = False,
           packing: str = "shelf",
           report_out: str | Path | None = None) -> dict:
    source, destination = Path(input_model).resolve(), Path(output_model).resolve()
    if source == destination:
        raise ValueError("Use a new output path to preserve the editing model")
    if destination.suffix.lower() not in (".gltf", ".glb"):
        raise ValueError("Output must be .gltf or .glb")
    if pad < 0:
        raise ValueError("Padding must be >= 0")
    if packing not in ("shelf", "dense"):
        raise ValueError("Packing must be 'shelf' or 'dense'")
    if not np.isfinite(similarity_threshold) or not 0 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be finite and between 0 and 1")
    if not isinstance(similarity_max_error, (int, np.integer)) or not 0 <= similarity_max_error <= 255:
        raise ValueError("Similarity max error must be an integer between 0 and 255")
    document = _Document(source)
    gltf = document.gltf
    if set(gltf.extensionsUsed or []) - {"KHR_materials_unlit", "KHR_mesh_quantization", "KHR_lights_punctual"}:
        raise ValueError("Decode/bake model extensions before repacking")
    source_paths = {source}
    for item in [*gltf.buffers, *gltf.images]:
        if item.uri and not item.uri.startswith("data:"):
            source_paths.add((source.parent / unquote(item.uri)).resolve())
    if texture:
        source_paths.add(Path(texture).resolve())
    output_texture = (Path(texture_out).resolve() if texture_out else destination.with_suffix(".png"))
    write_paths = [destination]
    if destination.suffix.lower() == ".gltf":
        write_paths.extend((destination.with_suffix(".bin"), output_texture))
    elif texture_out:
        write_paths.append(output_texture)
    if report_out:
        write_paths.append(Path(report_out).resolve())
    if len(set(write_paths)) != len(write_paths):
        raise ValueError("Model, buffer, texture and report output paths must be distinct")
    if source_paths & set(write_paths):
        raise ValueError("An output would overwrite an input model, buffer or painted image")

    jobs, image_ids = [], set()
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            material = _material_uv(gltf, primitive)
            if material is None:
                continue
            if primitive.extensions or primitive.mode not in (None, 4):
                raise ValueError("Triangulate/decode the textured mesh before repacking")
            image_id, uv_name, wrap_s, wrap_t = material
            image_ids.add(image_id)
            uv_index = getattr(primitive.attributes, uv_name, None)
            if uv_index is None:
                raise ValueError(f"Missing {uv_name}")
            positions = document.floats(primitive.attributes.POSITION)
            uv = document.floats(uv_index)
            if not np.isfinite(uv).all() or np.any(uv < -1e-7) or np.any(uv > 1 + 1e-7):
                raise ValueError("Bake repeating/out-of-range UVs into a 0..1 image before repacking")
            indices = (document.accessor(primitive.indices).reshape(-1).astype(np.int64)
                       if primitive.indices is not None else np.arange(len(positions)))
            if not len(indices) or len(indices) % 3:
                raise ValueError("Textured triangle indices are empty or incomplete")
            jobs.append((primitive, uv_name, positions, uv, indices.reshape(-1, 3), wrap_s, wrap_t))
    if len(image_ids) != 1:
        raise ValueError("Repacking requires exactly one shared base-colour image")
    image_id = next(iter(image_ids))
    raw = Path(texture).read_bytes() if texture else document.image(image_id)
    original = Image.open(io.BytesIO(raw)).convert("RGBA")
    pixels = np.asarray(original)
    patches, job_patches = [], []
    for _, _, positions, uv, triangles, wrap_s, wrap_t in jobs:
        current = _patches(positions, uv, triangles, pixels, pad, wrap_s, wrap_t)
        job_patches.append(current)
        patches.extend(current)

    solid_texels_before = 0
    if compress_solid:
        for patch in patches:
            if patch.pixels.shape[:2] == (1, 1) or not np.all(patch.block == patch.block[0, 0]):
                continue
            solid_texels_before += patch.pixels.shape[0] * patch.pixels.shape[1]
            colour = patch.block[0, 0].copy()
            patch.pixels = colour.reshape(1, 1, 4).copy()
            patch.block = np.broadcast_to(colour, (1 + 2 * pad, 1 + 2 * pad, 4)).copy()
            patch.compressed_solid = True

    rects = _assign_owners(patches, deduplicate=deduplicate, merge_similar=merge_similar,
                          threshold=similarity_threshold, max_error=similarity_max_error, pad=pad)
    if packing == "dense":
        width, height, placements = _pack_dense(rects, pot=pot)
    else:
        width, height, placements = _pack_rects_best(rects, pot=pot, square=False)
    if pot:
        width, height = _next_pow2(width), _next_pow2(height)
    packed = np.zeros((height, width, 4), dtype=np.uint8)
    for index, _, _ in rects:
        patch = patches[index]
        x, y = placements[index]
        block = patch.block
        packed[y:y + block.shape[0], x:x + block.shape[1]] = block
    for patch in patches:
        x, y = placements[patch.owner]
        patch.x, patch.y = x + pad, y + pad

    duplicated = 0
    for (primitive, uv_name, _, uv, triangles, _, _), current in zip(jobs, job_patches):
        uv_index = getattr(primitive.attributes, uv_name)
        remap = list(range(len(uv)))
        new_uv = list(uv.copy())
        first, split = {}, {}
        new_indices = triangles.copy()
        for cid, patch in enumerate(current):
            for old in np.unique(triangles[patch.triangles]):
                if old not in first:
                    first[old] = cid
                    new = int(old)
                else:
                    new = len(remap)
                    remap.append(int(old))
                    new_uv.append(uv[old].copy())
                    duplicated += 1
                split[(int(old), cid)] = new
                new_uv[new] = _mapped_pixels(patch, patches[patch.owner], uv[old] * original.size) / (width, height)
            for tid in patch.triangles:
                new_indices[tid] = [split[(int(v), cid)] for v in triangles[tid]]
        if len(remap) != len(uv):
            for name, aid in list(vars(primitive.attributes).items()):
                if aid is not None and name != uv_name:
                    setattr(primitive.attributes, name, document.add_accessor(document.accessor(aid)[remap], aid))
            for target in primitive.targets or []:
                fields = target if isinstance(target, dict) else vars(target)
                for name, aid in list(fields.items()):
                    if aid is not None:
                        fields[name] = document.add_accessor(document.accessor(aid)[remap], aid)
        setattr(primitive.attributes, uv_name, document.add_accessor(np.asarray(new_uv, dtype="<f4")))
        primitive.indices = document.add_accessor(new_indices.astype("<u4").reshape(-1, 1), indices=True)

    texture_bytes = _image_to_bytes(Image.fromarray(packed))
    # Embed any other images so moving the new model cannot break their paths.
    for index, image in enumerate(gltf.images):
        data = texture_bytes if index == image_id else document.image(index)
        image.uri = None
        image.mimeType = "image/png" if index == image_id else (image.mimeType or Image.MIME[Image.open(io.BytesIO(data)).format])
        image.bufferView = document.add_view(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".gltf":
        image = gltf.images[image_id]
        image.bufferView = None
        image.uri = quote(Path(os.path.relpath(output_texture, destination.parent)).as_posix(), safe="/")
    document.compact()
    if destination.suffix.lower() == ".glb":
        gltf.save_binary(str(destination))
    else:
        gltf.save_json(str(destination))
    if destination.suffix.lower() == ".gltf" or texture_out:
        output_texture.parent.mkdir(parents=True, exist_ok=True)
        output_texture.write_bytes(texture_bytes)
    report = {
        "input": str(source), "output": str(destination),
        "source_size": list(original.size), "packed_size": [width, height],
        "area_reduction_percent": round((1 - width * height / (original.width * original.height)) * 100, 2),
        "patches": len(patches), "unique_patches": len(rects), "padding": pad,
        "deduplicate": deduplicate, "duplicated_vertices": duplicated,
        "triangles": sum(len(j[4]) for j in jobs),
        "resampled": False,
        "packing": packing,
        "rect_fill_percent": round(sum(w * h for _, w, h in rects) / (width * height) * 100, 2),
    }
    if compress_solid:
        count = sum(p.compressed_solid for p in patches)
        report.update({"compress_solid": True, "solid_compressed_patches": count,
                       "solid_texels_before": solid_texels_before,
                       "solid_texels_after": count,
                       "solid_texels_saved_before_deduplication": solid_texels_before - count})
    if merge_similar:
        matches = [{"patch": i, "owner": p.owner,
                    "transform": {"swap_uv": p.transform[0], "flip_u": p.transform[1], "flip_v": p.transform[2]},
                    "similarity": 1.0 - p.mean_error / 255.0,
                    "mean_rgb_error": p.mean_error, "max_rgb_error": p.max_error}
                   for i, p in enumerate(patches) if p.owner != i]
        report.update({
            "merge_similar": True, "similarity_threshold": similarity_threshold,
            "similarity_max_error": similarity_max_error,
            "exact_merged_patches": sum(m["max_rgb_error"] == 0 for m in matches),
            "similar_merged_patches": sum(m["max_rgb_error"] > 0 for m in matches),
            "transformed_merged_patches": sum(any(m["transform"].values()) for m in matches),
            "lossy": any(m["max_rgb_error"] > 0 for m in matches),
            "max_rgb_error": max((m["max_rgb_error"] for m in matches), default=0),
            "max_mean_rgb_error": max((m["mean_rgb_error"] for m in matches), default=0.0),
            "similarity_matches": matches,
        })
    if report_out:
        path = Path(report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Painted .gltf/.glb model")
    parser.add_argument("--output", required=True, help="New .gltf/.glb model")
    parser.add_argument("--texture", help="Final painted/baked PNG; defaults to the image referenced by the model")
    parser.add_argument("--tex-out", help="Output PNG path; GLB embeds the image and optionally writes this copy")
    parser.add_argument("--tex-pad", type=int, default=2)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--tex-pot", dest="tex_pot", action="store_true", help="Round each atlas dimension up to a power of two")
    size_group.add_argument("--no-tex-pot", dest="tex_pot", action="store_false", help="Use compact non-POT dimensions (default)")
    parser.set_defaults(tex_pot=False)
    parser.add_argument("--deduplicate", action="store_true", help="Share exactly identical final image patches")
    parser.add_argument("--merge-similar", action="store_true", help="Share similar patches, including rotated/flipped matches; may change colours")
    parser.add_argument("--similarity-threshold", type=float, default=0.98, help="Minimum similarity (1 - RGB MAE / 255), 0..1; 1 allows exact matches only")
    parser.add_argument("--similarity-max-error", type=int, default=16, help="Maximum per-channel RGB difference, 0..255; alpha always matches exactly")
    parser.add_argument("--compress-solid-patches", action="store_true", help="Collapse patches whose core and gutters are a single RGBA colour to one texel")
    parser.add_argument("--pack", choices=("shelf", "dense"), default="shelf", help="Packing search: shelf is fast; dense tries MaxRects widths and orders")
    parser.add_argument("--report-out")
    args = parser.parse_args(argv)
    try:
        report = repack(args.input, args.output, texture=args.texture, texture_out=args.tex_out,
                        pad=args.tex_pad, pot=args.tex_pot, deduplicate=args.deduplicate,
                        merge_similar=args.merge_similar, similarity_threshold=args.similarity_threshold,
                        similarity_max_error=args.similarity_max_error,
                        compress_solid=args.compress_solid_patches,
                        packing=args.pack,
                        report_out=args.report_out)
    except ValueError as error:
        parser.error(str(error))
    print(f"Repacked {report['source_size']} -> {report['packed_size']}; "
          f"area reduced {report['area_reduction_percent']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
