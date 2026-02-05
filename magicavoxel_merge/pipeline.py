import numpy as np
from PIL import Image
from pathlib import Path
import os

from .glb import write_glb_scene
from .mesher import greedy_mesh, greedy_mesh_maxrect, greedy_quads, greedy_quads_baked, greedy_quads_maxrect, greedy_quads_baked_maxrect
from .vox import load_vox


def _weld_mesh(
    *,
    positions: np.ndarray,
    normals: np.ndarray,
    texcoords: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    texcoords = np.asarray(texcoords, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.uint32)

    if positions.shape[0] == 0:
        return positions, normals, texcoords, indices

    key = np.concatenate([positions, normals, texcoords], axis=1)
    key_i = np.round(key * 1_000_000.0).astype(np.int64)

    uniq, inv = np.unique(key_i, axis=0, return_inverse=True)
    new_positions = np.zeros((uniq.shape[0], 3), dtype=np.float32)
    new_texcoords = np.zeros((uniq.shape[0], 2), dtype=np.float32)
    new_normals = np.zeros((uniq.shape[0], 3), dtype=np.float32)
    counts = np.zeros((uniq.shape[0],), dtype=np.int32)

    np.add.at(new_positions, inv, positions)
    np.add.at(new_texcoords, inv, texcoords)
    np.add.at(new_normals, inv, normals)
    np.add.at(counts, inv, 1)

    counts_f = counts.astype(np.float32)
    new_positions = new_positions / counts_f[:, None]
    new_texcoords = new_texcoords / counts_f[:, None]

    lens = np.linalg.norm(new_normals, axis=1)
    lens = np.where(lens == 0.0, 1.0, lens)
    new_normals = new_normals / lens[:, None]

    new_indices = inv[indices]
    return new_positions, new_normals, new_texcoords, new_indices.astype(np.uint32)


def vox_to_glb(
    input_path: str,
    output_path: str,
    *,
    axis: str = "y_up",
    merge_strategy: str = "greedy",
    scale: float = 1.0,
    center: bool = False,
    center_bounds: bool = False,
    weld: bool = False,
    cull_mv_faces: str | None = None,
    cull_mv_z: float | None = None,
    atlas_pad: int = 2,
    atlas_inset: float = 1.5,
    atlas_style: str = "solid",
    baked_dedup: bool = True,
    atlas_texel_scale: int = 1,
    atlas_layout: str = "global",
    atlas_square: bool = False,
    handedness: str = "right",
    texture_out: str | None = None,
    preserve_transforms: bool = True,
    flip_v: bool = False,
    mode: str = "palette",
) -> None:
    vox = load_vox(input_path)
    meshes = []

    debug_cull = os.environ.get("MVM_DEBUG_CULL") not in (None, "", "0", "false", "False")

    mv_scene_ground_z = 0.0
    mv_model_occ_min_z: list[float] = [0.0] * len(vox.models)
    mv_model_bottom_world_z: list[float] = [0.0] * len(vox.models)
    if preserve_transforms and getattr(vox, "model_translations", None):
        try:
            # Pick the scene "ground" by the dominant layer, weighted by voxel count.
            # This makes large models define the ground instead of small outliers.
            weights: dict[int, float] = {}
            for i, m in enumerate(vox.models):
                tr_z = float(vox.model_translations[i][2]) if i < len(vox.model_translations) else 0.0
                occ_min_z = 0.0
                if getattr(m, "voxels", None) is not None:
                    filled = np.nonzero(m.voxels)
                    if len(filled) == 3 and filled[2].size > 0:
                        occ_min_z = float(int(filled[2].min()))
                mv_model_occ_min_z[i] = occ_min_z
                bottom_world = tr_z + occ_min_z
                mv_model_bottom_world_z[i] = bottom_world
                w = float(np.count_nonzero(m.voxels)) if getattr(m, "voxels", None) is not None else 1.0
                key = int(round(float(bottom_world)))
                weights[key] = weights.get(key, 0.0) + max(1.0, w)
            if weights:
                mv_scene_ground_z = float(max(weights.items(), key=lambda kv: kv[1])[0])
        except Exception:
            mv_scene_ground_z = 0.0

    # Only cull models that are on/near the detected scene ground.
    mv_ground_eps = 0.5

    name_prefix = Path(input_path).stem

    if mode not in ("palette", "atlas"):
        raise ValueError("mode must be 'palette' or 'atlas'")

    if merge_strategy not in ("greedy", "maxrect"):
        raise ValueError("merge_strategy must be 'greedy' or 'maxrect'")

    if axis not in ("y_up", "mv_zup", "identity"):
        raise ValueError("axis must be 'y_up', 'mv_zup', or 'identity'")

    if atlas_layout not in ("global", "by-model"):
        raise ValueError("atlas_layout must be 'global' or 'by-model'")

    if atlas_style not in ("solid", "baked"):
        raise ValueError("atlas_style must be 'solid' or 'baked'")

    atlas_texel_scale = int(atlas_texel_scale)
    if atlas_texel_scale < 1:
        raise ValueError("atlas_texel_scale must be >= 1")

    if handedness not in ("right", "left"):
        raise ValueError("handedness must be 'right' or 'left'")

    flip_handedness = handedness == "left"

    def _map_axes_vector(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        ax = axis
        if ax == "identity":
            return v
        if ax == "mv_zup":
            ax = "y_up"
        if ax == "y_up":
            x, y, z = v
            return np.asarray([x, z, -y], dtype=np.float32)
        raise ValueError(f"unsupported axis mapping: {ax}")

    def _map_axes_mesh(
        positions: np.ndarray,
        normals: np.ndarray,
        translation: tuple[float, float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float] | None]:
        ax = axis

        if ax == "identity":
            return positions, normals, translation

        # Backward-compatible alias
        if ax == "mv_zup":
            ax = "y_up"

        # MagicaVoxel: X right, Y inward, Z up
        # glTF: Y up
        # Map: (x,y,z) -> (x,z,-y)
        positions = positions.copy()
        normals = normals.copy()

        x = positions[:, 0].copy()
        y = positions[:, 1].copy()
        z = positions[:, 2].copy()
        positions[:, 0] = x
        positions[:, 1] = z
        positions[:, 2] = -y

        nx = normals[:, 0].copy()
        ny = normals[:, 1].copy()
        nz = normals[:, 2].copy()
        normals[:, 0] = nx
        normals[:, 1] = nz
        normals[:, 2] = -ny

        if translation is not None:
            tx, ty, tz = translation
            translation = (tx, tz, -ty)

        return positions, normals, translation

    def _flip_handedness_mesh(
        positions: np.ndarray,
        normals: np.ndarray,
        indices: np.ndarray,
        translation: tuple[float, float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float] | None]:
        # Mirror Z to switch handedness while keeping Y-up.
        positions = positions.copy()
        normals = normals.copy()
        positions[:, 2] *= -1.0
        normals[:, 2] *= -1.0

        # Flip triangle winding: (i0,i1,i2) -> (i0,i2,i1)
        idx = indices.copy()
        if idx.size % 3 != 0:
            raise ValueError("indices length must be multiple of 3")
        idx = idx.reshape((-1, 3))
        idx = idx[:, [0, 2, 1]]
        idx = idx.reshape((-1,)).astype(np.uint32)

        if translation is not None:
            translation = (translation[0], translation[1], -translation[2])
        return positions, normals, idx, translation

    if center and center_bounds:
        raise ValueError("center and center_bounds are mutually exclusive")

    import io

    def _compact_mesh(
        *,
        positions: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if indices.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                indices.astype(np.uint32),
            )
        used = np.unique(indices.astype(np.uint32))
        remap = np.full((positions.shape[0],), -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0], dtype=np.int64)
        new_indices = remap[indices.astype(np.uint32)].astype(np.uint32)
        return (
            positions[used],
            normals[used],
            texcoords[used],
            new_indices,
        )

    def _parse_mv_faces(spec: str | None) -> set[str]:
        if not spec:
            return set()
        parts = [p.strip().lower() for p in str(spec).split(",") if p.strip()]
        out: set[str] = set()
        for p in parts:
            if p == "top":
                out.add("+z")
            elif p == "bottom":
                out.add("-z")
            else:
                out.add(p)
        allowed = {"+x", "-x", "+y", "-y", "+z", "-z"}
        bad = sorted([p for p in out if p not in allowed])
        if bad:
            raise ValueError("cull_mv_faces has invalid entries: " + ",".join(bad))
        return out

    mv_faces_set = _parse_mv_faces(cull_mv_faces)

    def _cull_mv_z_mesh(
        *,
        positions: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
        indices: np.ndarray,
        plane_z: float,
        translation_z: float = 0.0,
        threshold: float = -0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if indices.size == 0:
            return positions, normals, texcoords, indices
        tris = indices.astype(np.uint32).reshape((-1, 3))
        # Downward faces in MV are -Z.
        nz = normals[tris[:, 0], 2]
        drop = nz <= float(threshold)
        z0 = positions[tris[:, 0], 2] + float(translation_z)
        z1 = positions[tris[:, 1], 2] + float(translation_z)
        z2 = positions[tris[:, 2], 2] + float(translation_z)
        below = (z0 <= float(plane_z)) & (z1 <= float(plane_z)) & (z2 <= float(plane_z))
        drop = drop & below

        keep = ~drop
        new_tris = tris[keep]
        new_indices = new_tris.reshape((-1,)).astype(np.uint32)
        return _compact_mesh(positions=positions, normals=normals, texcoords=texcoords, indices=new_indices)

    def _cull_mv_oriented_faces(
        *,
        positions: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
        indices: np.ndarray,
        faces: set[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not faces or indices.size == 0:
            return positions, normals, texcoords, indices

        tris = indices.astype(np.uint32).reshape((-1, 3))
        p0 = positions[tris[:, 0]]
        p1 = positions[tris[:, 1]]
        p2 = positions[tris[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)

        ax = np.abs(fn[:, 0])
        ay = np.abs(fn[:, 1])
        az = np.abs(fn[:, 2])
        dom = np.stack([ax, ay, az], axis=1)
        k = np.argmax(dom, axis=1)

        # Only treat clearly axis-aligned faces as cull candidates.
        domv = dom[np.arange(dom.shape[0]), k]
        oth = (ax + ay + az) - domv
        aligned = (domv > 1e-6) & (domv >= oth * 4.0)

        dir_tags = np.empty((tris.shape[0],), dtype=np.int8)
        # 0 none, 1 +x,2 -x,3 +y,4 -y,5 +z,6 -z
        dir_tags[:] = 0
        sx = fn[:, 0] >= 0.0
        sy = fn[:, 1] >= 0.0
        sz = fn[:, 2] >= 0.0
        dir_tags[(k == 0) & aligned & sx] = 1
        dir_tags[(k == 0) & aligned & (~sx)] = 2
        dir_tags[(k == 1) & aligned & sy] = 3
        dir_tags[(k == 1) & aligned & (~sy)] = 4
        dir_tags[(k == 2) & aligned & sz] = 5
        dir_tags[(k == 2) & aligned & (~sz)] = 6

        want = set(faces)
        drop = np.zeros((tris.shape[0],), dtype=bool)
        if "+x" in want:
            drop |= dir_tags == 1
        if "-x" in want:
            drop |= dir_tags == 2
        if "+y" in want:
            drop |= dir_tags == 3
        if "-y" in want:
            drop |= dir_tags == 4
        if "+z" in want:
            drop |= dir_tags == 5
        if "-z" in want:
            drop |= dir_tags == 6

        keep = ~drop
        new_tris = tris[keep]
        new_indices = new_tris.reshape((-1,)).astype(np.uint32)
        return _compact_mesh(positions=positions, normals=normals, texcoords=texcoords, indices=new_indices)

    def _emit(output_path_local: str, meshes_local: list[dict[str, np.ndarray]], texture_png_local: bytes) -> None:
        if texture_out is not None:
            if texture_out == "__AUTO__":
                tex_path = Path(output_path_local).with_suffix(".png")
            else:
                tex_path = Path(texture_out)
                if tex_path.is_dir() or str(texture_out).endswith("/"):
                    tex_path = tex_path / (Path(output_path_local).stem + ".png")

            tex_path.parent.mkdir(parents=True, exist_ok=True)
            tex_path.write_bytes(texture_png_local)
            glb_dir = Path(output_path_local).parent
            uri = os.path.relpath(tex_path, glb_dir)
            uri = uri.replace(os.sep, "/")
            write_glb_scene(output_path_local, meshes=meshes_local, texture_uri=uri, name_prefix=name_prefix)
        else:
            write_glb_scene(output_path_local, meshes=meshes_local, texture_png=texture_png_local, name_prefix=name_prefix)

    if mode == "palette":
        palette_rgba = vox.palette_rgba
        img = Image.fromarray(palette_rgba.reshape((1, 256, 4)), mode="RGBA")
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        texture_png = bio.getvalue()

        for idx, m in enumerate(vox.models):
            mesh = greedy_mesh_maxrect(m.voxels, m.size, vox.palette_rgba) if merge_strategy == "maxrect" else greedy_mesh(m.voxels, m.size, vox.palette_rgba)
            positions = mesh["positions"]
            indices = mesh["indices"]
            normals = mesh["normals"]
            texcoords = mesh["texcoords"]

            texcoords = texcoords.copy()

            # MV-space z-plane culling must happen before scale/center/axis.
            if cull_mv_z is not None:
                tz_mv = 0.0
                bottom_world = 0.0
                occ_min_z = 0.0
                if preserve_transforms and idx < len(vox.model_translations):
                    tr_z_dbg = float(vox.model_translations[idx][2])
                    occ_min_z = mv_model_occ_min_z[idx] if idx < len(mv_model_occ_min_z) else 0.0
                    bottom_world = mv_model_bottom_world_z[idx] if idx < len(mv_model_bottom_world_z) else (tr_z_dbg + occ_min_z)
                    tz_mv = tr_z_dbg - mv_scene_ground_z

                do_cull = (not preserve_transforms) or (abs(float(bottom_world) - float(mv_scene_ground_z)) <= mv_ground_eps)
                if not do_cull:
                    name_dbg = vox.model_names[idx] if idx < len(vox.model_names) else f"model_{idx}"
                    tr_z_dbg = float(vox.model_translations[idx][2]) if (preserve_transforms and idx < len(vox.model_translations)) else 0.0
                    if debug_cull:
                        print(f"[MVM_DEBUG_CULL] palette cull-mv-z SKIP plane={float(cull_mv_z)} ground_z={mv_scene_ground_z} model={name_dbg} tr_z={tr_z_dbg} occ_min_z={occ_min_z} bottom_world_z={bottom_world} eps={mv_ground_eps}")
                    do_cull = False

                if do_cull:
                    before_tris = int(indices.size // 3)
                    positions, normals, texcoords, indices = _cull_mv_z_mesh(
                        positions=positions,
                        normals=normals,
                        texcoords=texcoords,
                        indices=indices,
                        plane_z=float(cull_mv_z),
                        translation_z=tz_mv,
                    )

                    if debug_cull:
                        after_tris = int(indices.size // 3)
                        name_dbg = vox.model_names[idx] if idx < len(vox.model_names) else f"model_{idx}"
                        tr_z_dbg = float(vox.model_translations[idx][2]) if (preserve_transforms and idx < len(vox.model_translations)) else 0.0
                        print(f"[MVM_DEBUG_CULL] palette cull-mv-z plane={float(cull_mv_z)} ground_z={mv_scene_ground_z} model={name_dbg} tr_z={tr_z_dbg} tz_mv={tz_mv} tris_before={before_tris} tris_after={after_tris} dropped={before_tris - after_tris}")

            if mv_faces_set:
                positions, normals, texcoords, indices = _cull_mv_oriented_faces(
                    positions=positions,
                    normals=normals,
                    texcoords=texcoords,
                    indices=indices,
                    faces=mv_faces_set,
                )

            if scale != 1.0:
                positions = positions * float(scale)

            base_translation = None
            if preserve_transforms and idx < len(vox.model_translations):
                tx, ty, tz = vox.model_translations[idx]
                base_translation = (float(tx) * float(scale), float(ty) * float(scale), float(tz) * float(scale))

            translation = base_translation
            if center_bounds:
                mins = positions.min(axis=0)
                maxs = positions.max(axis=0)
                offset = (mins + maxs) / 2.0
                positions = positions - offset
                if base_translation is not None:
                    translation = (base_translation[0] + float(offset[0]), base_translation[1] + float(offset[1]), base_translation[2] + float(offset[2]))
            elif center:
                offset = (np.asarray(m.size, dtype=np.float32) / 2.0) * float(scale)
                positions = positions - offset
                if base_translation is not None:
                    translation = (base_translation[0] + float(offset[0]), base_translation[1] + float(offset[1]), base_translation[2] + float(offset[2]))

            name = vox.model_names[idx] if idx < len(vox.model_names) else f"model_{idx}"
            positions, normals, translation = _map_axes_mesh(positions, normals, translation)

            if flip_handedness:
                positions, normals, indices, translation = _flip_handedness_mesh(
                    positions=positions,
                    normals=normals,
                    indices=indices,
                    translation=translation,
                )

            if weld:
                positions, normals, texcoords, indices = _weld_mesh(
                    positions=positions,
                    normals=normals,
                    texcoords=texcoords,
                    indices=indices,
                )

            meshes.append({"positions": positions, "indices": indices, "normals": normals, "texcoords": texcoords, "name": name, "translation": translation})

        _emit(output_path, meshes, texture_png)

    # atlas mode
    pad = int(atlas_pad)
    if pad < 0:
        raise ValueError("atlas_pad must be >= 0")

    inset = float(atlas_inset)
    if inset < 0.0:
        raise ValueError("atlas_inset must be >= 0")

    def next_pow2(x: int) -> int:
        p = 1
        while p < x:
            p <<= 1
        return p

    def pack_best(rects_sorted_local: list[tuple[int, int, int]], widths_local: list[int]) -> tuple[int, int, list[tuple[int, int]]]:
        best_area = None
        best_pos = None
        best_w = None
        best_h = None
        best_max = None
        best_score = None
        for w in widths_local:
            h_raw, pos = pack_shelf(rects_sorted_local, w)
            if h_raw >= 10**9:
                continue
            h = next_pow2(h_raw)
            area = w * h
            if atlas_square:
                m = max(w, h)
                if best_area is None or m < best_max or (m == best_max and area < best_area):
                    best_area = area
                    best_max = m
                    best_pos = pos
                    best_w = w
                    best_h = h
            else:
                # Prefer smaller area but avoid extreme aspect ratios that waste space
                # (e.g. very tall or very wide atlases).
                aspect = float(max(w, h)) / float(min(w, h))
                penalty = 0.0
                if aspect > 2.0:
                    penalty = (aspect - 2.0) * 0.25
                score = float(area) * (1.0 + penalty)
                if best_score is None or score < best_score:
                    best_score = score
                    best_area = area
                    best_pos = pos
                    best_w = w
                    best_h = h

        if best_pos is None or best_w is None or best_h is None:
            raise ValueError("Failed to pack atlas")
        return int(best_w), int(best_h), best_pos

    def pack_maxrect(rects_sorted: list[tuple[int, int, int]], width: int, height: int) -> list[tuple[int, int]] | None:
        # MaxRects (best short side fit), no rotation.
        # rects_sorted: (rid, w, h) in preferred placement order.
        free: list[tuple[int, int, int, int]] = [(0, 0, int(width), int(height))]
        out: list[tuple[int, int]] = [(-1, -1)] * len(rects_sorted)

        def _prune(frs: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
            pruned: list[tuple[int, int, int, int]] = []
            for i, a in enumerate(frs):
                ax, ay, aw, ah = a
                contained = False
                for j, b in enumerate(frs):
                    if i == j:
                        continue
                    bx, by, bw, bh = b
                    if ax >= bx and ay >= by and ax + aw <= bx + bw and ay + ah <= by + bh:
                        contained = True
                        break
                if not contained and aw > 0 and ah > 0:
                    pruned.append(a)
            return pruned

        def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

        for rid, w, h in rects_sorted:
            bw = int(w)
            bh = int(h)
            best_i = -1
            best_x = 0
            best_y = 0
            best_ss = None
            best_ls = None

            for i, (fx, fy, fw, fh) in enumerate(free):
                if bw <= fw and bh <= fh:
                    ss = min(fw - bw, fh - bh)
                    ls = max(fw - bw, fh - bh)
                    if best_ss is None or ss < best_ss or (ss == best_ss and ls < best_ls):
                        best_ss = ss
                        best_ls = ls
                        best_i = i
                        best_x = fx
                        best_y = fy

            if best_i < 0:
                return None

            out[int(rid)] = (int(best_x), int(best_y))

            placed = (int(best_x), int(best_y), bw, bh)

            # Split ALL free rectangles that intersect the placed rectangle.
            new_free: list[tuple[int, int, int, int]] = []
            for fr in free:
                if not _intersects(fr, placed):
                    new_free.append(fr)
                    continue

                fx, fy, fw, fh = fr
                px, py, pw, ph = placed
                pr = px + pw
                pb = py + ph
                fr_r = fx + fw
                fr_b = fy + fh

                # Left slice
                if px > fx:
                    new_free.append((fx, fy, px - fx, fh))
                # Right slice
                if pr < fr_r:
                    new_free.append((pr, fy, fr_r - pr, fh))
                # Top slice
                if py > fy:
                    new_free.append((fx, fy, fw, py - fy))
                # Bottom slice
                if pb < fr_b:
                    new_free.append((fx, pb, fw, fr_b - pb))

            free = _prune(new_free)

        return out

    def pack_best_bin(
        rects_sorted_local: list[tuple[int, int, int]],
        widths_local: list[int],
        heights_local: list[int],
    ) -> tuple[int, int, list[tuple[int, int]]]:
        best_area = None
        best_pos = None
        best_w = None
        best_h = None
        best_max = None
        best_score = None

        for w in widths_local:
            for h0 in heights_local:
                pos = pack_maxrect(rects_sorted_local, int(w), int(h0))
                if pos is None:
                    continue
                area = int(w) * int(h0)
                if atlas_square:
                    m = max(int(w), int(h0))
                    if best_area is None or m < best_max or (m == best_max and area < best_area):
                        best_area = area
                        best_max = m
                        best_pos = pos
                        best_w = int(w)
                        best_h = int(h0)
                else:
                    aspect = float(max(int(w), int(h0))) / float(min(int(w), int(h0)))
                    penalty = 0.0
                    if aspect > 2.0:
                        penalty = (aspect - 2.0) * 0.25
                    score = float(area) * (1.0 + penalty)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_area = area
                        best_pos = pos
                        best_w = int(w)
                        best_h = int(h0)

        if best_pos is None or best_w is None or best_h is None:
            raise ValueError("Failed to pack atlas")
        return int(best_w), int(best_h), best_pos

    def pack_shelf(rects: list[tuple[int, int, int]] , width: int) -> tuple[int, list[tuple[int, int]]]:
        # rects: (rid, w, h)
        x = 0
        y = 0
        row_h = 0
        out: list[tuple[int, int]] = [(-1, -1)] * len(rects)
        for rid, w, h in rects:
            if w > width:
                return 10**9, []
            if x + w > width:
                x = 0
                y += row_h
                row_h = 0
            out[rid] = (x, y)
            x += w
            row_h = max(row_h, h)
        return y + row_h, out

    quads_per_model = []
    rects: list[tuple[int, int, int]] = []
    quad_meta = []

    def _quad_mv_dir_tag(q: dict[str, object]) -> str | None:
        n = q.get("normal")
        if not isinstance(n, tuple) or len(n) != 3:
            return None
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        ax, ay, az = abs(nx), abs(ny), abs(nz)
        if ax >= ay and ax >= az and ax > 0.5:
            return "+x" if nx > 0.0 else "-x"
        if ay >= ax and ay >= az and ay > 0.5:
            return "+y" if ny > 0.0 else "-y"
        if az >= ax and az >= ay and az > 0.5:
            return "+z" if nz > 0.0 else "-z"
        return None

    quad_meta_baked_by_model: list[dict[int, int]] | None = None
    quad_meta_baked_global_pos: list[tuple[int, int]] | None = None
    seen_baked_blocks: set[tuple[int, int]] = set()

    rid = 0
    for midx, m in enumerate(vox.models):
        if atlas_style == "baked":
            quads = greedy_quads_baked_maxrect(m.voxels, m.size) if merge_strategy == "maxrect" else greedy_quads_baked(m.voxels, m.size)
        else:
            quads = greedy_quads_maxrect(m.voxels, m.size) if merge_strategy == "maxrect" else greedy_quads(m.voxels, m.size)

        if mv_faces_set:
            keep_quads = []
            for q in quads:
                tag = _quad_mv_dir_tag(q)
                if tag is not None and tag in mv_faces_set:
                    continue
                keep_quads.append(q)
            quads = keep_quads

        if cull_mv_z is not None:
            tz_mv = 0.0
            bottom_world = 0.0
            occ_min_z = 0.0
            if preserve_transforms and midx < len(vox.model_translations):
                tr_z_dbg = float(vox.model_translations[midx][2])
                occ_min_z = mv_model_occ_min_z[midx] if midx < len(mv_model_occ_min_z) else 0.0
                bottom_world = mv_model_bottom_world_z[midx] if midx < len(mv_model_bottom_world_z) else (tr_z_dbg + occ_min_z)
                tz_mv = tr_z_dbg - mv_scene_ground_z

            do_cull = (not preserve_transforms) or (abs(float(bottom_world) - float(mv_scene_ground_z)) <= mv_ground_eps)
            if not do_cull:
                if debug_cull:
                    name_dbg = vox.model_names[midx] if midx < len(vox.model_names) else f"model_{midx}"
                    tr_z_dbg = float(vox.model_translations[midx][2]) if (preserve_transforms and midx < len(vox.model_translations)) else 0.0
                    print(f"[MVM_DEBUG_CULL] atlas cull-mv-z SKIP plane={float(cull_mv_z)} ground_z={mv_scene_ground_z} model={name_dbg} tr_z={tr_z_dbg} occ_min_z={occ_min_z} bottom_world_z={bottom_world} eps={mv_ground_eps}")
            keep_quads = []
            cand = 0
            dropped = 0
            cand_zmax_min = None
            cand_zmax_max = None
            if do_cull:
                for q in quads:
                    tag = _quad_mv_dir_tag(q)
                    if tag == "-z":
                        cand += 1
                        p0, p1, p2, p3 = q["verts"]
                        zmin = min(float(p0[2]), float(p1[2]), float(p2[2]), float(p3[2])) + tz_mv
                        zmax = max(float(p0[2]), float(p1[2]), float(p2[2]), float(p3[2])) + tz_mv
                        if cand_zmax_min is None or zmax < cand_zmax_min:
                            cand_zmax_min = float(zmax)
                        if cand_zmax_max is None or zmax > cand_zmax_max:
                            cand_zmax_max = float(zmax)
                        if zmax <= float(cull_mv_z):
                            dropped += 1
                            continue
                    keep_quads.append(q)
                quads = keep_quads

                if debug_cull:
                    name_dbg = vox.model_names[midx] if midx < len(vox.model_names) else f"model_{midx}"
                    tr_z_dbg = float(vox.model_translations[midx][2]) if (preserve_transforms and midx < len(vox.model_translations)) else 0.0
                    print(f"[MVM_DEBUG_CULL] atlas cull-mv-z plane={float(cull_mv_z)} ground_z={mv_scene_ground_z} model={name_dbg} tr_z={tr_z_dbg} tz_mv={tz_mv} cand_zmax=[{cand_zmax_min},{cand_zmax_max}] quads_in={len(quads) + dropped} cand_negz={cand} dropped={dropped} quads_out={len(quads)}")
            else:
                # Pass through unchanged.
                if debug_cull:
                    name_dbg = vox.model_names[midx] if midx < len(vox.model_names) else f"model_{midx}"
                    tr_z_dbg = float(vox.model_translations[midx][2]) if (preserve_transforms and midx < len(vox.model_translations)) else 0.0
                    print(f"[MVM_DEBUG_CULL] atlas cull-mv-z PASS plane={float(cull_mv_z)} ground_z={mv_scene_ground_z} model={name_dbg} tr_z={tr_z_dbg} tz_mv={tz_mv} quads_in={len(quads)}")

        quads_per_model.append(quads)

        for q in quads:
            tex_w = int(q["w"]) * atlas_texel_scale
            tex_h = int(q["h"]) * atlas_texel_scale
            w = tex_w + pad * 2
            h = tex_h + pad * 2
            rects.append((rid, w, h))
            quad_meta.append((rid, midx, q))
            rid += 1

    import hashlib

    widths = [1 << k for k in range(4, 14)]
    heights = [1 << k for k in range(4, 14)]

    do_baked_dedup = (atlas_style == "baked") and bool(baked_dedup)

    if do_baked_dedup:
        # Deduplicate identical baked blocks.
        if atlas_layout == "global":
            # Global dedup: identical blocks across all models share one packed rect.
            uniq_rects: list[tuple[int, int, int]] = []
            uniq_meta: list[tuple[int, int, dict[str, object]]] = []
            key_to_uid: dict[bytes, int] = {}
            rid_to_uid: dict[int, int] = {}

            uid = 0
            for ridx, midx, q in quad_meta:
                colors = q.get("colors")
                if colors is None:
                    raise ValueError("baked atlas requires quad 'colors'")
                c = np.asarray(colors, dtype=np.int32)
                hq = int(q["h"])
                wq = int(q["w"])
                if c.shape != (hq, wq):
                    raise ValueError("quad colors shape mismatch")
                # Hash includes dimensions and sampling parameters that affect final pixels.
                hsh = hashlib.blake2b(digest_size=16)
                hsh.update(np.asarray([wq, hq, int(atlas_texel_scale), int(pad)], dtype=np.int32).tobytes())
                n_vec = np.asarray(q.get("normal", (0, 0, 0)), dtype=np.int8)
                hsh.update(n_vec.tobytes())
                hsh.update(c.tobytes())
                key = hsh.digest()

                if key in key_to_uid:
                    rid_to_uid[int(ridx)] = key_to_uid[key]
                    continue

                key_to_uid[key] = uid
                rid_to_uid[int(ridx)] = uid

                tex_w = wq * atlas_texel_scale
                tex_h = hq * atlas_texel_scale
                w = tex_w + pad * 2
                h = tex_h + pad * 2
                uniq_rects.append((uid, w, h))
                uniq_meta.append((uid, midx, q))
                uid += 1

            rects_sorted = sorted(uniq_rects, key=lambda t: (t[2], t[1]), reverse=True)
            best_w, best_h, best_pos_uniq = pack_best_bin(rects_sorted, widths, heights)

            quad_meta_baked_global_pos = best_pos_uniq

            # Expand best_pos so each original quad rid maps to its unique rect pos.
            best_pos = [(-1, -1)] * len(rects)
            for ridx in range(len(rects)):
                u = rid_to_uid.get(int(ridx))
                if u is None:
                    continue
                best_pos[ridx] = best_pos_uniq[u]

            # Only bake unique blocks.
            quad_meta = [(uid_, midx_, q_) for (uid_, midx_, q_) in uniq_meta]
        else:
            # by-model dedup: identical blocks dedup within each model only.
            model_block_sizes = []
            model_quad_local_pos: list[list[tuple[int, int]]] = []
            model_uniq_quads: list[list[dict[str, object]]] = []
            model_rid_to_luid: list[dict[int, int]] = []

            rid_base = 0
            for midx, quads in enumerate(quads_per_model):
                key_to_luid: dict[bytes, int] = {}
                rid_to_luid: dict[int, int] = {}
                uniq_quads: list[dict[str, object]] = []
                rects_m: list[tuple[int, int, int]] = []

                luid = 0
                for qidx, q in enumerate(quads):
                    colors = q.get("colors")
                    if colors is None:
                        raise ValueError("baked atlas requires quad 'colors'")
                    c = np.asarray(colors, dtype=np.int32)
                    hq = int(q["h"])
                    wq = int(q["w"])
                    if c.shape != (hq, wq):
                        raise ValueError("quad colors shape mismatch")

                    hsh = hashlib.blake2b(digest_size=16)
                    hsh.update(np.asarray([wq, hq, int(atlas_texel_scale), int(pad)], dtype=np.int32).tobytes())
                    n_vec = np.asarray(q.get("normal", (0, 0, 0)), dtype=np.int8)
                    hsh.update(n_vec.tobytes())
                    hsh.update(c.tobytes())
                    key = hsh.digest()

                    global_rid = rid_base + qidx
                    if key in key_to_luid:
                        rid_to_luid[global_rid] = key_to_luid[key]
                        continue

                    key_to_luid[key] = luid
                    rid_to_luid[global_rid] = luid
                    uniq_quads.append(q)

                    tex_w = wq * atlas_texel_scale
                    tex_h = hq * atlas_texel_scale
                    w = tex_w + pad * 2
                    h = tex_h + pad * 2
                    rects_m.append((luid, w, h))
                    luid += 1

                rects_m_sorted = sorted(rects_m, key=lambda t: (t[2], t[1]), reverse=True)
                w_choices = [1 << k for k in range(4, 13)]
                h_choices = [1 << k for k in range(4, 13)]
                mw, mh, mpos = pack_best_bin(rects_m_sorted, w_choices, h_choices)
                model_block_sizes.append((mw, mh))
                model_quad_local_pos.append(mpos)
                model_uniq_quads.append(uniq_quads)
                model_rid_to_luid.append(rid_to_luid)

                rid_base += len(quads)

            block_rects = [(midx, int(sz[0]), int(sz[1])) for midx, sz in enumerate(model_block_sizes)]
            block_sorted = sorted(block_rects, key=lambda t: (t[2], t[1]), reverse=True)
            best_w, best_h, model_pos = pack_best_bin(block_sorted, widths, heights)

            # best_pos for original quad rids
            best_pos = [(-1, -1)] * len(rects)
            rid_base = 0
            for midx, quads in enumerate(quads_per_model):
                mx, my = model_pos[midx]
                rid_to_luid = model_rid_to_luid[midx]
                for qidx, _q in enumerate(quads):
                    global_rid = rid_base + qidx
                    luid = rid_to_luid.get(global_rid)
                    if luid is None:
                        continue
                    qx, qy = model_quad_local_pos[midx][luid]
                    best_pos[global_rid] = (mx + qx, my + qy)
                rid_base += len(quads)

            # Keep original quad_meta (global rid indexing). We'll skip baking duplicates later.
            quad_meta_baked_by_model = model_rid_to_luid
    else:
        # No baked dedup (or non-baked style): pack all quads normally.
        if atlas_layout == "global":
            rects_sorted = sorted(rects, key=lambda t: (t[2], t[1]), reverse=True)
            best_w, best_h, best_pos = pack_best_bin(rects_sorted, widths, heights)
        else:
            model_block_sizes = []
            model_quad_local_pos: list[list[tuple[int, int]]] = []
            for midx, quads in enumerate(quads_per_model):
                rects_m: list[tuple[int, int, int]] = []
                for qidx, q in enumerate(quads):
                    tex_w = int(q["w"]) * atlas_texel_scale
                    tex_h = int(q["h"]) * atlas_texel_scale
                    w = tex_w + pad * 2
                    h = tex_h + pad * 2
                    rects_m.append((qidx, w, h))

                rects_m_sorted = sorted(rects_m, key=lambda t: (t[2], t[1]), reverse=True)
                w_choices = [1 << k for k in range(4, 13)]
                h_choices = [1 << k for k in range(4, 13)]
                mw, mh, mpos = pack_best_bin(rects_m_sorted, w_choices, h_choices)
                model_block_sizes.append((mw, mh))
                model_quad_local_pos.append(mpos)

            block_rects = [(midx, int(sz[0]), int(sz[1])) for midx, sz in enumerate(model_block_sizes)]
            block_sorted = sorted(block_rects, key=lambda t: (t[2], t[1]), reverse=True)
            best_w, best_h, model_pos = pack_best_bin(block_sorted, widths, heights)

            best_pos = [(-1, -1)] * len(rects)
            rid_base = 0
            for midx, quads in enumerate(quads_per_model):
                mx, my = model_pos[midx]
                for qidx, _q in enumerate(quads):
                    qx, qy = model_quad_local_pos[midx][qidx]
                    best_pos[rid_base + qidx] = (mx + qx, my + qy)
                rid_base += len(quads)

    atlas_arr = np.zeros((best_h, best_w, 4), dtype=np.uint8)

    def pal_color(color_index: int) -> tuple[int, int, int, int]:
        return tuple(int(v) for v in vox.palette_rgba[color_index - 1])

    for ridx, midx, q in quad_meta:
        if atlas_style == "baked" and atlas_layout != "global" and quad_meta_baked_by_model is not None:
            # When baked+by-model dedup is active, skip baking duplicates.
            rid_to_luid = quad_meta_baked_by_model[midx]
            luid = rid_to_luid.get(int(ridx))
            if luid is not None:
                key = (int(midx), int(luid))
                if key in seen_baked_blocks:
                    continue
                seen_baked_blocks.add(key)

        if atlas_style == "baked" and atlas_layout == "global" and quad_meta_baked_global_pos is not None:
            ox, oy = quad_meta_baked_global_pos[int(ridx)]
        else:
            ox, oy = best_pos[ridx]

        tex_w = int(q["w"]) * atlas_texel_scale
        tex_h = int(q["h"]) * atlas_texel_scale
        full_w = tex_w + pad * 2
        full_h = tex_h + pad * 2

        if atlas_style == "baked":
            quad_colors = q.get("colors")
            if quad_colors is None:
                raise ValueError("baked atlas requires quad 'colors'")

            quad_colors = np.asarray(quad_colors, dtype=np.int32)
            if quad_colors.shape != (int(q["h"]), int(q["w"])):
                raise ValueError("quad colors shape mismatch")

            # Map color indices to RGBA.
            rgba = vox.palette_rgba[quad_colors - 1]
            # Upscale each voxel color to texel blocks.
            rgba = np.repeat(np.repeat(rgba, atlas_texel_scale, axis=0), atlas_texel_scale, axis=1)
            # Pad with edge pixels for seam safety.
            rgba = np.pad(rgba, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            if rgba.shape[0] != full_h or rgba.shape[1] != full_w:
                raise ValueError("baked quad rgba shape mismatch")
            atlas_arr[oy : oy + full_h, ox : ox + full_w, :] = rgba.astype(np.uint8)
        else:
            col = pal_color(int(q["color"]))
            atlas_arr[oy : oy + full_h, ox : ox + full_w, :] = np.asarray(col, dtype=np.uint8)

    atlas = Image.fromarray(atlas_arr, mode="RGBA")
    bio = io.BytesIO()
    atlas.save(bio, format="PNG")
    texture_png = bio.getvalue()

    for midx, quads in enumerate(quads_per_model):
        positions = []
        normals = []
        texcoords = []
        indices = []
        # Global quad id base for this model
        rid_base = sum(len(qs) for qs in quads_per_model[:midx])
        local = 0

        for q in quads:
            ridx = rid_base + local
            ox, oy = best_pos[ridx]
            w_vox = float(q["w"])
            h_vox = float(q["h"])
            w = w_vox * float(atlas_texel_scale)
            h = h_vox * float(atlas_texel_scale)

            # UVs use the same top-left origin convention as our baked atlas coordinates.
            # Inset further into the chart to reduce shimmering during minification.
            # Clamp inset so we don't invert UVs on tiny charts.
            inset_u = min(inset, max(0.0, (w - 1.0) / 2.0))
            inset_v = min(inset, max(0.0, (h - 1.0) / 2.0))
            u0 = (ox + pad + inset_u) / float(best_w)
            v0 = (oy + pad + inset_v) / float(best_h)
            u1 = (ox + pad + w - inset_u) / float(best_w)
            v1 = (oy + pad + h - inset_v) / float(best_h)

            if flip_v:
                # glTF commonly uses a top-left image origin convention, but some engines treat
                # v=0 as bottom. Flip V while preserving v0 < v1 ordering.
                v0, v1 = (1.0 - v1), (1.0 - v0)

            p0, p1, p2, p3 = q["verts"]
            n = q["normal"]
            positions.extend([p0, p1, p2, p3])
            normals.extend([n, n, n, n])

            # Robust UV mapping:
            # - The mesher may swap p1/p3 for correct winding.
            # - Quads have declared voxel dimensions (w_vox, h_vox); ensure the UV's U axis
            #   aligns with the edge whose length matches w_vox, and V axis matches h_vox.
            o = np.asarray(p0, dtype=np.float32)
            e1 = np.asarray(p1, dtype=np.float32) - o
            e3 = np.asarray(p3, dtype=np.float32) - o
            len1 = float(np.linalg.norm(e1))
            len3 = float(np.linalg.norm(e3))

            # Decide whether (p0->p1) corresponds to width or height.
            d_keep = abs(len1 - w_vox) + abs(len3 - h_vox)
            d_swap = abs(len1 - h_vox) + abs(len3 - w_vox)
            if d_swap < d_keep:
                # Swap axes: treat e1 as V axis and e3 as U axis.
                u_axis = e3
                v_axis = e1
                u0_, u1_ = u0, u1
                v0_, v1_ = v0, v1
            else:
                u_axis = e1
                v_axis = e3
                u0_, u1_ = u0, u1
                v0_, v1_ = v0, v1

            uu = float(np.dot(u_axis, u_axis))
            vv = float(np.dot(v_axis, v_axis))
            if uu == 0.0:
                uu = 1.0
            if vv == 0.0:
                vv = 1.0

            def _uv_for(p: tuple[float, float, float]) -> tuple[float, float]:
                d = np.asarray(p, dtype=np.float32) - o
                a = float(np.dot(d, u_axis) / uu)
                b = float(np.dot(d, v_axis) / vv)
                # Clamp numerically; should be in {0,1} for rectangle verts.
                a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
                b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
                u = u0_ + (u1_ - u0_) * a
                v = v0_ + (v1_ - v0_) * b
                return (float(u), float(v))

            texcoords.extend([_uv_for(p0), _uv_for(p1), _uv_for(p2), _uv_for(p3)])
            indices.extend([len(positions) - 4, len(positions) - 3, len(positions) - 2, len(positions) - 4, len(positions) - 2, len(positions) - 1])
            local += 1

        positions = np.asarray(positions, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        texcoords = np.asarray(texcoords, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)

        if mv_faces_set:
            positions, normals, texcoords, indices = _cull_mv_oriented_faces(
                positions=positions,
                normals=normals,
                texcoords=texcoords,
                indices=indices,
                faces=mv_faces_set,
            )

        if scale != 1.0:
            positions = positions * float(scale)

        base_translation = None
        if preserve_transforms and midx < len(vox.model_translations):
            tx, ty, tz = vox.model_translations[midx]
            base_translation = (float(tx) * float(scale), float(ty) * float(scale), float(tz) * float(scale))

        translation = base_translation
        if center_bounds:
            mins = positions.min(axis=0)
            maxs = positions.max(axis=0)
            offset = (mins + maxs) / 2.0
            positions = positions - offset
            if base_translation is not None:
                translation = (base_translation[0] + float(offset[0]), base_translation[1] + float(offset[1]), base_translation[2] + float(offset[2]))
        elif center:
            offset = (np.asarray(vox.models[midx].size, dtype=np.float32) / 2.0) * float(scale)
            positions = positions - offset
            if base_translation is not None:
                translation = (base_translation[0] + float(offset[0]), base_translation[1] + float(offset[1]), base_translation[2] + float(offset[2]))

        name = vox.model_names[midx] if midx < len(vox.model_names) else f"model_{midx}"
        positions, normals, translation = _map_axes_mesh(positions, normals, translation)

        if flip_handedness:
            positions, normals, indices, translation = _flip_handedness_mesh(
                positions=positions,
                normals=normals,
                indices=indices,
                translation=translation,
            )

        if weld:
            positions, normals, texcoords, indices = _weld_mesh(
                positions=positions,
                normals=normals,
                texcoords=texcoords,
                indices=indices,
            )

        meshes.append({"positions": positions, "indices": indices, "normals": normals, "texcoords": texcoords, "name": name, "translation": translation})

    _emit(output_path, meshes, texture_png)
