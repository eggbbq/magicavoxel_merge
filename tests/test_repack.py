from __future__ import annotations

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pygltflib import Accessor, Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler, Node, Skin

from btp_vox.pipeline import AtlasOptions, PipelineOptions, convert
from btp_vox.repack import (
    _Document, _Patch, _TRANSFORMS, _assign_owners, _mapped_pixels,
    _transform_block, _pack_dense, main, repack,
)
from btp_vox.atlas import _pack_rects_best
from btp_vox.voxio import VoxModel, VoxNode, VoxScene


def _export(path: Path, *, instance=False, alpha=False):
    vox = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(np.int32) + 1
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, :3] = np.arange(256)[:, None] * np.array([1, 17, 37]) % 256
    palette[:, 3] = 255
    if alpha:
        palette[::2, 3] = 128
    nodes = [VoxNode("shp", "part", model_ids=[0])]
    if instance:
        nodes = [VoxNode("grp", "root", children=[1, 2]),
                 VoxNode("shp", "left", model_ids=[0]), VoxNode("shp", "right", model_ids=[0])]
    scene = VoxScene([VoxModel("part", vox.shape, vox, (0, 0, 0), (0, 0, 0, 1))], palette, nodes, [0])
    with patch("btp_vox.pipeline.load_scene", return_value=scene):
        convert("source.vox", path, output_format="glb" if path.suffix == ".glb" else "gltf",
                options=PipelineOptions(uv_mode="paint", atlas=AtlasOptions(texel_scale=3, pot=True),
                                        export_uv2=True, export_vertex_color=True))


def _save(document, path):
    document.gltf.buffers[0].uri = None
    document.gltf.buffers[0].byteLength = len(document.data)
    document.gltf.set_binary_blob(bytes(document.data))
    document.gltf.save_json(str(path))


def _pixels(document):
    return np.asarray(Image.open(io.BytesIO(document.image(0))).convert("RGBA"))


def _sample(pixels, uv, linear=False):
    height, width = pixels.shape[:2]
    xy = uv * (width, height)
    if not linear:
        x, y = np.clip(np.floor(xy).astype(int), (0, 0), (width - 1, height - 1))
        return pixels[y, x]
    xy -= 0.5
    lo = np.floor(xy).astype(int)
    fraction = xy - lo
    value = np.zeros(4)
    for dy in (0, 1):
        for dx in (0, 1):
            x, y = np.clip(lo + (dx, dy), (0, 0), (width - 1, height - 1))
            weight = (fraction[0] if dx else 1 - fraction[0]) * (fraction[1] if dy else 1 - fraction[1])
            value += pixels[y, x] * weight
    return value


class RepackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "editing.gltf"
        self.output = self.directory / "runtime" / "runtime.gltf"
        _export(self.source)

    def assert_same_surfaces(self, before, after, original_pixels=None):
        old_pixels = _pixels(before) if original_pixels is None else original_pixels
        new_pixels = _pixels(after)
        rng = np.random.default_rng(42)
        barycentric = rng.dirichlet(np.ones(3), 30)
        # Near the original net seams, the new gutters must copy actual source
        # neighbours rather than changing linear filtering to edge clamping.
        barycentric = np.vstack((barycentric, [[.00001, .45003, .54996], [.12345, .00001, .87654]]))
        for old_mesh, new_mesh in zip(before.gltf.meshes, after.gltf.meshes):
            for old, new in zip(old_mesh.primitives, new_mesh.primitives):
                old_i = before.accessor(old.indices).reshape(-1, 3)
                new_i = after.accessor(new.indices).reshape(-1, 3)
                self.assertEqual(old_i.shape, new_i.shape)
                info = before.gltf.materials[old.material].pbrMetallicRoughness.baseColorTexture
                uv_name = f"TEXCOORD_{info.texCoord or 0}"
                for name, index in vars(old.attributes).items():
                    if index is not None and name != uv_name:
                        np.testing.assert_array_equal(before.accessor(index)[old_i],
                                                      after.accessor(getattr(new.attributes, name))[new_i])
                old_uv = before.floats(getattr(old.attributes, uv_name))[old_i]
                new_uv = after.floats(getattr(new.attributes, uv_name))[new_i]
                for old_tri, new_tri in zip(old_uv, new_uv):
                    for weights in barycentric:
                        a, b = weights @ old_tri, weights @ new_tri
                        np.testing.assert_array_equal(_sample(old_pixels, a), _sample(new_pixels, b))
                        np.testing.assert_allclose(_sample(old_pixels, a, True), _sample(new_pixels, b, True), atol=.01)

    def test_repack_uses_only_model_and_texture_and_preserves_surface_colours(self):
        before = _Document(self.source)
        report = repack(self.source, self.output)
        self.assertEqual(report["patches"], 6)
        self.assertFalse(report["resampled"])
        self.assertGreater(report["area_reduction_percent"], 0)
        self.assert_same_surfaces(before, _Document(self.output))

    def test_final_paint_and_bake_pixels_are_used_without_vox_or_layout_json(self):
        pixels = _pixels(_Document(self.source)).copy()
        # Simulate detailed painting and a baked intensity gradient, including alpha.
        yy, xx = np.indices(pixels.shape[:2])
        pixels[:, :, 0] = xx * 19 % 256
        pixels[:, :, 1] = yy * 23 % 256
        pixels[:, :, 3] = (xx + yy) % 256
        final_image = self.directory / "finished painting.png"
        Image.fromarray(pixels).save(final_image)
        repack(self.source, self.output, texture=final_image)
        self.assert_same_surfaces(_Document(self.source), _Document(self.output), pixels)

    def test_authoring_model_buffer_and_image_are_unchanged(self):
        files = [self.source, self.source.with_suffix(".bin"), self.source.with_suffix(".png")]
        hashes = [hashlib.sha256(p.read_bytes()).digest() for p in files]
        repack(self.source, self.output)
        self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).digest() for p in files])

    def test_material_using_secondary_uv_set_keeps_primary_uv_set(self):
        document = _Document(self.source)
        document.gltf.materials[0].pbrMetallicRoughness.baseColorTexture.texCoord = 1
        _save(document, self.source)
        repack(self.source, self.output)
        self.assert_same_surfaces(_Document(self.source), _Document(self.output))

    def test_pot_and_non_pot_are_both_available_without_resampling(self):
        first = repack(self.source, self.output, pot=False)
        other = self.directory / "poweroftwo.glb"
        second = repack(self.source, other, pot=True)
        self.assertTrue(any(n & (n - 1) for n in first["packed_size"]))
        self.assertTrue(all(n & (n - 1) == 0 for n in second["packed_size"]))
        self.assert_same_surfaces(_Document(self.source), _Document(other))

    def test_glb_input_and_embedded_output_remain_self_contained(self):
        source = self.directory / "editing.glb"
        _export(source, alpha=True)
        output = self.directory / "runtime.glb"
        repack(source, output)
        result = _Document(output)
        self.assertIsNone(result.gltf.images[0].uri)
        self.assert_same_surfaces(_Document(source), result)
        # Replaced authoring image/attributes must not stay in the binary payload.
        self.assertEqual(len(result.gltf.accessors), len(_Document(source).gltf.accessors))

    def test_identical_patches_can_be_shared_optionally(self):
        source = self.directory / "two_parts.gltf"
        _export(source, instance=True)
        first = repack(source, self.output, deduplicate=False)
        second = repack(source, self.directory / "shared.gltf", deduplicate=True)
        self.assertEqual(first["patches"], 12)
        self.assertEqual(second["unique_patches"], 6)
        self.assertLess(np.prod(second["packed_size"]), np.prod(first["packed_size"]))
        self.assert_same_surfaces(_Document(source), _Document(self.directory / "shared.gltf"))

    def test_similar_merge_preserves_geometry_and_reports_actual_colour_error(self):
        source = self.directory / "two_parts.gltf"
        _export(source, instance=True)
        before = _Document(source)
        pixels = _pixels(before).copy()
        primitive = before.gltf.meshes[1].primitives[0]
        uv = before.floats(primitive.attributes.TEXCOORD_0)
        low = np.rint(uv.min(axis=0) * (pixels.shape[1], pixels.shape[0])).astype(int)
        high = np.rint(uv.max(axis=0) * (pixels.shape[1], pixels.shape[0])).astype(int)
        x0, y0 = low; x1, y1 = high
        pixels[y0:y1, x0:x1, :3] = np.minimum(pixels[y0:y1, x0:x1, :3].astype(np.int16) + 2, 255)
        painted = self.directory / "finished.png"
        Image.fromarray(pixels).save(painted)
        report = repack(source, self.output, texture=painted, pad=1, merge_similar=True)
        self.assertEqual(report["unique_patches"], 6)
        self.assertEqual(report["similar_merged_patches"], 6)
        self.assertTrue(report["lossy"])
        self.assertEqual(report["max_rgb_error"], 2)
        self.assertTrue(all(m["owner"] < m["patch"] and m["similarity"] >= .98
                            for m in report["similarity_matches"]))
        after = _Document(self.output)
        packed = _pixels(after)
        barycentric = np.random.default_rng(15).dirichlet(np.ones(3), 20)
        for old_mesh, new_mesh in zip(before.gltf.meshes, after.gltf.meshes):
            old, new = old_mesh.primitives[0], new_mesh.primitives[0]
            old_i, new_i = before.accessor(old.indices).reshape(-1, 3), after.accessor(new.indices).reshape(-1, 3)
            for name, index in vars(old.attributes).items():
                if index is not None and name != "TEXCOORD_0":
                    np.testing.assert_array_equal(before.accessor(index)[old_i], after.accessor(getattr(new.attributes, name))[new_i])
            a = before.floats(old.attributes.TEXCOORD_0)[old_i]
            b = after.floats(new.attributes.TEXCOORD_0)[new_i]
            for source_uv, target_uv in zip(a, b):
                for weights in barycentric:
                    src = _sample(pixels, weights @ source_uv).astype(int)
                    dst = _sample(packed, weights @ target_uv).astype(int)
                    self.assertLessEqual(int(np.abs(src[:3] - dst[:3]).max()), 2)
                    self.assertEqual(src[3], dst[3])

    def test_invalid_similarity_options_fail_before_writing(self):
        for options in ({"similarity_threshold": -1}, {"similarity_threshold": 1.01},
                        {"similarity_threshold": float("nan")}, {"similarity_max_error": 256},
                        {"similarity_max_error": 2.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                repack(self.source, self.output, merge_similar=True, **options)
        self.assertFalse(self.output.exists())

    def test_solid_compression_preserves_colour_alpha_geometry_and_linear_sampling(self):
        source = _Document(self.source)
        pixels = _pixels(source).copy()
        pixels[:] = [90, 130, 180, 128]
        texture = self.directory / "solid.png"
        Image.fromarray(pixels).save(texture)
        report = repack(self.source, self.output, texture=texture, pad=2,
                        merge_similar=True, compress_solid=True, packing="dense")
        self.assertEqual(report["solid_compressed_patches"], 6)
        self.assertEqual(report["unique_patches"], 1)
        self.assertEqual(report["packed_size"], [5, 5])
        self.assertFalse(report["lossy"])
        self.assertGreater(report["solid_texels_saved_before_deduplication"], 0)
        self.assert_same_surfaces(source, _Document(self.output), pixels)

    def test_nonuniform_detail_is_not_collapsed_as_a_solid(self):
        source = _Document(self.source)
        pixels = _pixels(source).copy()
        pixels[:] = [90, 130, 180, 255]
        uv = source.floats(source.gltf.meshes[0].primitives[0].attributes.TEXCOORD_0)
        x, y = np.floor(uv[:4].mean(axis=0) * (pixels.shape[1], pixels.shape[0])).astype(int)
        pixels[y, x] = [220, 40, 55, 255]
        texture = self.directory / "detailed.png"
        Image.fromarray(pixels).save(texture)
        report = repack(self.source, self.output, texture=texture, pad=2,
                        merge_similar=True, compress_solid=True, packing="dense")
        self.assertLess(report["solid_compressed_patches"], 6)
        self.assertGreater(report["solid_compressed_patches"], 0)
        self.assert_same_surfaces(source, _Document(self.output), pixels)

    def test_dense_packing_stays_in_bounds_without_overlap_and_never_worsens_shelf_area(self):
        sets = [[(0, 9, 6), (1, 7, 12), (2, 1, 1), (3, 3, 11), (4, 5, 2)],
                [(i, int(w), int(h)) for i, (w, h) in enumerate(np.random.default_rng(9).integers(1, 30, (24, 2)))]]
        for rects in sets:
            for pot in (False, True):
                with self.subTest(rects=len(rects), pot=pot):
                    w, h, placements = _pack_dense(rects, pot=pot)
                    old_w, old_h, _ = _pack_rects_best(rects, pot=pot, square=False)
                    if pot:
                        old_w, old_h = 1 << (old_w - 1).bit_length(), 1 << (old_h - 1).bit_length()
                        self.assertTrue(all(n & (n - 1) == 0 for n in (w, h)))
                    self.assertLessEqual(w * h, old_w * old_h)
                    used = np.zeros((h, w), dtype=bool)
                    for i, rw, rh in rects:
                        x, y = placements[i]
                        self.assertGreaterEqual(min(x, y), 0)
                        self.assertLessEqual(x + rw, w)
                        self.assertLessEqual(y + rh, h)
                        self.assertFalse(used[y:y+rh, x:x+rw].any())
                        used[y:y+rh, x:x+rw] = True

    def test_shared_vertices_are_split_and_custom_attributes_morphs_survive(self):
        doc = _Document(self.source)
        primitive = doc.gltf.meshes[0].primitives[0]
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
        primitive.attributes.POSITION = doc.add_accessor(positions, primitive.attributes.POSITION)
        primitive.attributes.NORMAL = doc.add_accessor(positions + 1, primitive.attributes.NORMAL)
        primitive.attributes.TEXCOORD_0 = doc.add_accessor(np.array([[.1, .1], [.4, .1], [.1, .4], [.6, .6]], dtype="<f4"))
        primitive.attributes.TEXCOORD_1 = doc.add_accessor(np.zeros((4, 2), dtype="<f4"))
        primitive.attributes.COLOR_0 = doc.add_accessor(np.ones((4, 4), dtype="<f4"), primitive.attributes.COLOR_0)
        primitive.attributes._CUSTOM = doc.add_accessor(positions * 3, primitive.attributes.POSITION)
        primitive.indices = doc.add_accessor(np.array([[0], [1], [2], [0], [3], [1]], dtype="<u4"), indices=True)
        primitive.targets = [{"POSITION": doc.add_accessor(positions * .1, primitive.attributes.POSITION)}]
        _save(doc, self.source)
        before = _Document(self.source)
        report = repack(self.source, self.output)
        self.assertEqual(report["duplicated_vertices"], 2)
        after = _Document(self.output)
        self.assert_same_surfaces(before, after)
        old, new = before.gltf.meshes[0].primitives[0], after.gltf.meshes[0].primitives[0]
        old_i, new_i = before.accessor(old.indices), after.accessor(new.indices)
        np.testing.assert_array_equal(before.accessor(old.targets[0]["POSITION"])[old_i],
                                      after.accessor(new.targets[0]["POSITION"])[new_i])

    def test_skin_and_animation_binary_data_and_node_structure_survive(self):
        doc = _Document(self.source)
        g = doc.gltf
        primitive = g.meshes[0].primitives[0]
        count = g.accessors[primitive.attributes.POSITION].count
        joints = np.zeros((count, 4), dtype="<u2")
        weights = np.zeros((count, 4), dtype="<f4"); weights[:, 0] = 1
        view = doc.add_view(joints.tobytes(), 34962)
        g.accessors.append(Accessor(bufferView=view, componentType=5123, count=count, type="VEC4"))
        primitive.attributes.JOINTS_0 = len(g.accessors) - 1
        primitive.attributes.WEIGHTS_0 = doc.add_accessor(weights, primitive.attributes.COLOR_0)
        view = doc.add_view(np.eye(4, dtype="<f4").tobytes())
        g.accessors.append(Accessor(bufferView=view, componentType=5126, count=1, type="MAT4"))
        inverse = len(g.accessors) - 1
        g.nodes.append(Node(name="Bone"))
        g.skins.append(Skin(joints=[len(g.nodes) - 1], inverseBindMatrices=inverse))
        g.nodes[0].skin = 0
        view = doc.add_view(np.array([0., 1.], dtype="<f4").tobytes())
        g.accessors.append(Accessor(bufferView=view, componentType=5126, count=2, type="SCALAR", min=[0.], max=[1.]))
        times = len(g.accessors) - 1
        translations = doc.add_accessor(np.array([[0., 0., 0.], [1., 2., 3.]], dtype="<f4"), primitive.attributes.POSITION)
        g.animations.append(Animation(name="Move", samplers=[AnimationSampler(input=times, output=translations)],
                                      channels=[AnimationChannel(sampler=0, target=AnimationChannelTarget(node=len(g.nodes)-1, path="translation"))]))
        nodes = deepcopy(g.nodes)
        _save(doc, self.source)
        before = _Document(self.source)
        repack(self.source, self.output)
        after = _Document(self.output)
        self.assertEqual(nodes, after.gltf.nodes)
        self.assertEqual(after.gltf.animations[0].name, "Move")
        old_sampler, new_sampler = before.gltf.animations[0].samplers[0], after.gltf.animations[0].samplers[0]
        for field in ("input", "output"):
            np.testing.assert_array_equal(before.accessor(getattr(old_sampler, field)), after.accessor(getattr(new_sampler, field)))
        a = before.gltf.accessors[before.gltf.skins[0].inverseBindMatrices]
        b = after.gltf.accessors[after.gltf.skins[0].inverseBindMatrices]
        self.assertEqual(before.view(a.bufferView), after.view(b.bufferView))
        self.assert_same_surfaces(before, after)

    def test_outputs_cannot_overwrite_inputs_or_each_other(self):
        cases = [dict(output_model=self.source), dict(output_model=self.output, texture_out=self.source.with_suffix(".png")),
                 dict(output_model=self.output, report_out=self.output), dict(output_model=self.output, texture_out=self.output)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                repack(self.source, **kwargs)

    def test_unsupported_repeating_uvs_fail_before_writing_outputs(self):
        doc = _Document(self.source)
        primitive = doc.gltf.meshes[0].primitives[0]
        uv = doc.floats(primitive.attributes.TEXCOORD_0)
        uv[0] = (2, 2)
        primitive.attributes.TEXCOORD_0 = doc.add_accessor(uv.astype("<f4"))
        _save(doc, self.source)
        with self.assertRaisesRegex(ValueError, "0..1"):
            repack(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_accepts_explicit_pot_and_no_pot_choices(self):
        for flag, expected in (("--tex-pot", True), ("--no-tex-pot", False)):
            with patch("sys.stdout", new=io.StringIO()), patch("btp_vox.repack.repack", return_value={"source_size": [64,64], "packed_size": [32,32], "area_reduction_percent": 75}) as mocked:
                main(["--input", "editing.gltf", "--output", "runtime.gltf", flag])
                self.assertEqual(mocked.call_args.kwargs["pot"], expected)

    def test_cli_passes_similarity_controls(self):
        with patch("sys.stdout", new=io.StringIO()), patch("btp_vox.repack.repack", return_value={"source_size": [64,64], "packed_size": [32,32], "area_reduction_percent": 75}) as mocked:
            main(["--input", "editing.gltf", "--output", "runtime.gltf", "--merge-similar",
                  "--similarity-threshold", ".995", "--similarity-max-error", "4",
                  "--compress-solid-patches", "--pack", "dense"])
            self.assertTrue(mocked.call_args.kwargs["merge_similar"])
            self.assertEqual(mocked.call_args.kwargs["similarity_threshold"], .995)
            self.assertEqual(mocked.call_args.kwargs["similarity_max_error"], 4)
            self.assertTrue(mocked.call_args.kwargs["compress_solid"])
            self.assertEqual(mocked.call_args.kwargs["packing"], "dense")


class SimilarityMatcherTests(unittest.TestCase):
    @staticmethod
    def make_patch(pixels, pad=0):
        block = np.asarray(pixels, dtype=np.uint8).copy()
        core = block[pad:-pad, pad:-pad].copy() if pad else block.copy()
        return _Patch([], np.array([0, 0]), core, block)

    @staticmethod
    def solid(value):
        block = np.full((5, 8, 4), value, dtype=np.uint8)
        block[:, :, 3] = 255
        return block

    def assign(self, patches, threshold=.98, max_error=16, pad=0):
        return _assign_owners(patches, deduplicate=False, merge_similar=True,
                              threshold=threshold, max_error=max_error, pad=pad)

    def test_small_colour_changes_can_merge_and_threshold_can_reject(self):
        patches = [self.make_patch(self.solid(v)) for v in (100, 102)]
        self.assertEqual(len(self.assign(patches)), 1)
        self.assertEqual(patches[1].owner, 0)
        self.assertEqual(patches[1].mean_error, 2)
        self.assertEqual(patches[1].max_error, 2)
        self.assertEqual(len(self.assign(patches, threshold=.999)), 2)

    def test_alpha_differences_never_merge(self):
        first = self.solid(100)
        second = first.copy(); second[1, 1, 3] = 254
        patches = [self.make_patch(first), self.make_patch(second)]
        self.assertEqual(len(self.assign(patches, threshold=0, max_error=255)), 2)

    def test_peak_error_keeps_tiny_distinctive_details(self):
        first = self.solid(100)
        second = first.copy(); second[1, 1, 0] = 200
        patches = [self.make_patch(first), self.make_patch(second)]
        # This patch meets the mean threshold, but must fail the peak-error cap.
        self.assertLess(np.abs(first.astype(int) - second.astype(int)).mean(), 5.1)
        self.assertEqual(len(self.assign(patches)), 2)

    def test_merge_does_not_accumulate_error_through_chains(self):
        patches = [self.make_patch(self.solid(v)) for v in (100, 104, 108)]
        self.assertEqual(len(self.assign(patches)), 2)
        self.assertEqual([p.owner for p in patches], [0, 0, 2])

    def test_closest_eligible_representative_is_selected(self):
        patches = [self.make_patch(self.solid(v)) for v in (100, 110, 107)]
        self.assign(patches)
        self.assertEqual([p.owner for p in patches], [0, 1, 1])
        self.assertEqual(patches[2].mean_error, 3)

    def test_all_rotations_and_reflections_map_rectangular_texels_correctly(self):
        original = np.arange(3 * 5 * 4, dtype=np.uint8).reshape(3, 5, 4)
        original[:, :, 3] = 255
        for transform in _TRANSFORMS:
            with self.subTest(transform=transform):
                child = _transform_block(original, transform).copy()
                owner, patch = self.make_patch(original), self.make_patch(child)
                self.assertEqual(len(self.assign([owner, patch], threshold=1)), 1)
                for y in range(child.shape[0]):
                    for x in range(child.shape[1]):
                        mapped = np.floor(_mapped_pixels(patch, owner, np.array([x + .5, y + .5]))).astype(int)
                        np.testing.assert_array_equal(original[mapped[1], mapped[0]], child[y, x])

    def test_size_changes_require_no_resampling_and_remain_separate(self):
        first = self.solid(100)
        larger = np.repeat(np.repeat(first, 2, axis=0), 2, axis=1)
        self.assertEqual(len(self.assign([self.make_patch(first), self.make_patch(larger)])), 2)

    def test_gutter_alpha_is_also_protected(self):
        first = self.solid(100)
        second = first.copy(); second[0, 0, 3] = 0
        self.assertEqual(len(self.assign([self.make_patch(first, 1), self.make_patch(second, 1)], pad=1)), 2)


if __name__ == "__main__":
    unittest.main()
