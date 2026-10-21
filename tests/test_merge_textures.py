from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pygltflib import Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler, Node, Skin

from btp_vox.merge_textures import main, merge_textures
from btp_vox.pipeline import AtlasOptions, PipelineOptions, convert
from btp_vox.repack import _Document, _TRANSFORMS, _transform_block
from btp_vox.voxio import VoxModel, VoxNode, VoxScene


def _export(path, *, size=(3, 4, 5), scale=3, alpha=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    vox = (np.arange(np.prod(size)).reshape(size) % 254 + 1).astype(np.int32)
    palette = np.full((256, 4), 255, dtype=np.uint8)
    palette[:, :3] = np.arange(256)[:, None] * np.array([1, 17, 37]) % 256
    if alpha:
        palette[::2, 3] = 128
    scene = VoxScene([VoxModel("part", size, vox, (0, 0, 0), (0, 0, 0, 1))],
                     palette, [VoxNode("shp", "part", model_ids=[0])], [0])
    with patch("btp_vox.pipeline.load_scene", return_value=scene):
        convert("input.vox", path, output_format="glb" if path.suffix == ".glb" else "gltf",
                options=PipelineOptions(uv_mode="paint", export_uv2=True, export_vertex_color=True,
                                        atlas=AtlasOptions(texel_scale=scale, square=True, pot=True)))


def _save(document, path):
    document.compact()
    if path.suffix == ".glb":
        document.gltf.save_binary(str(path))
    else:
        document.gltf.save_json(str(path))


def _sample(pixels, uv):
    xy = np.floor(uv * (pixels.shape[1], pixels.shape[0])).astype(int)
    xy = np.clip(xy, (0, 0), (pixels.shape[1] - 1, pixels.shape[0] - 1))
    return pixels[xy[:, 1], xy[:, 0]]


class MergeTexturesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.first, self.second = self.directory / "first.gltf", self.directory / "second.gltf"
        self.output = self.directory / "output"
        _export(self.first)
        _export(self.second)

    def assert_models(self, source, destination, *, max_error=0):
        before, after = _Document(source), _Document(destination)
        self.assertEqual(before.gltf.nodes, after.gltf.nodes)
        self.assertEqual(before.gltf.scenes, after.gltf.scenes)
        self.assertEqual(before.gltf.materials, after.gltf.materials)
        self.assertEqual(before.gltf.samplers, after.gltf.samplers)
        weights = np.random.default_rng(42).dirichlet(np.ones(3), 80)
        # Also check close to UV seams, without sampling exactly on a boundary.
        weights = np.vstack((weights, [[.00001, .45003, .54996], [.12345, .00001, .87654]]))
        for oldmesh, newmesh in zip(before.gltf.meshes, after.gltf.meshes):
            for old, new in zip(oldmesh.primitives, newmesh.primitives):
                oi = before.accessor(old.indices).reshape(-1, 3)
                ni = after.accessor(new.indices).reshape(-1, 3)
                info = before.gltf.materials[old.material].pbrMetallicRoughness.baseColorTexture
                uv_name = f"TEXCOORD_{info.texCoord or 0}"
                for name, index in vars(old.attributes).items():
                    if index is not None and name != uv_name:
                        np.testing.assert_array_equal(before.accessor(index)[oi],
                                                      after.accessor(getattr(new.attributes, name))[ni])
                old_image = before.gltf.textures[info.index].source
                new_image = after.gltf.textures[info.index].source
                a = np.asarray(Image.open(io.BytesIO(before.image(old_image))).convert("RGBA"))
                b = np.asarray(Image.open(io.BytesIO(after.image(new_image))).convert("RGBA"))
                olduv = before.floats(getattr(old.attributes, uv_name))[oi]
                newuv = after.floats(getattr(new.attributes, uv_name))[ni]
                for ot, nt in zip(olduv, newuv):
                    ac, bc = _sample(a, weights @ ot), _sample(b, weights @ nt)
                    self.assertLessEqual(int(np.abs(ac[:, :3].astype(int) - bc[:, :3].astype(int)).max()), max_error)
                    np.testing.assert_array_equal(ac[:, 3], bc[:, 3])

    def test_multiple_sizes_glb_alpha_and_one_external_png(self):
        other = self.directory / "other.glb"
        _export(other, size=(2, 5, 4), scale=4, alpha=True)
        sources = [self.first, other]
        protected = [self.first, self.first.with_suffix(".png"), self.first.with_suffix(".bin"), other]
        hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in protected}
        report = merge_textures(sources, self.output, pad=2, pot=True, packing="dense")
        self.assertEqual(report["model_count"], 2)
        for value in report["packed_size"]:
            self.assertEqual(value & (value - 1), 0)
        for source, record in zip(sources, report["models"]):
            self.assert_models(source, Path(record["output"]))
            doc = _Document(Path(record["output"]))
            self.assertEqual(doc.gltf.images[0].uri, "shared_atlas.png")
            self.assertIsNone(doc.gltf.images[0].bufferView)
        self.assertEqual(len(list(self.output.glob("*.png"))), 1)
        for p, old in hashes.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).digest(), old)

    def test_cross_model_exact_and_rotation_matching_adjusts_uvs(self):
        for transform in _TRANSFORMS:
            with self.subTest(transform=transform):
                _export(self.second)
                doc = _Document(self.second)
                raw = np.asarray(Image.open(self.second.with_suffix(".png")).convert("RGBA"))
                Image.fromarray(_transform_block(raw, transform)).save(self.second.with_suffix(".png"))
                swap, flip_u, flip_v = transform
                for mesh in doc.gltf.meshes:
                    for primitive in mesh.primitives:
                        uv = doc.floats(primitive.attributes.TEXCOORD_0)
                        if swap:
                            uv = uv[:, ::-1].copy()
                        if flip_u:
                            uv[:, 0] = 1 - uv[:, 0]
                        if flip_v:
                            uv[:, 1] = 1 - uv[:, 1]
                        primitive.attributes.TEXCOORD_0 = doc.add_accessor(uv.astype("<f4"))
                _save(doc, self.second)
                report = merge_textures([self.first, self.second], self.output, pad=0,
                                        merge_similar=True, similarity_threshold=1, similarity_max_error=0)
                self.assertGreater(report["cross_model_merged_patches"], 0)
                if transform != (False, False, False):
                    self.assertGreater(report["transformed_merged_patches"], 0)
                self.assertFalse(report["lossy"])
                for source, record in zip([self.first, self.second], report["models"]):
                    self.assert_models(source, Path(record["output"]))

    def test_material_using_uv2_only_updates_that_uv_set(self):
        doc = _Document(self.second)
        doc.gltf.materials[0].pbrMetallicRoughness.baseColorTexture.texCoord = 1
        _save(doc, self.second)
        report = merge_textures([self.first, self.second], self.output, deduplicate=True)
        self.assertGreater(report["cross_model_merged_patches"], 0)
        self.assert_models(self.second, Path(report["models"][1]["output"]))

    def test_similar_colours_merge_but_alpha_must_match(self):
        raw = np.asarray(Image.open(self.second.with_suffix(".png")).convert("RGBA")).copy()
        raw[:, :, :3] = np.minimum(raw[:, :, :3].astype(int) + 1, 255)
        Image.fromarray(raw).save(self.second.with_suffix(".png"))
        report = merge_textures([self.first, self.second], self.output, pad=0,
                                merge_similar=True, similarity_threshold=.99, similarity_max_error=1)
        self.assertGreater(report["cross_model_merged_patches"], 0)
        self.assertTrue(report["lossy"])
        for source, record in zip([self.first, self.second], report["models"]):
            self.assert_models(source, Path(record["output"]), max_error=1)
        raw[:, :, 3] -= 1
        Image.fromarray(raw).save(self.second.with_suffix(".png"))
        report = merge_textures([self.first, self.second], self.output, pad=0,
                                merge_similar=True, similarity_threshold=.99, similarity_max_error=1)
        self.assertEqual(report["cross_model_merged_patches"], 0)

    def test_multiple_base_colour_images_in_one_model(self):
        doc = _Document(self.first)
        image = deepcopy(doc.gltf.images[0])
        image.uri, image.bufferView = "other.png", None
        doc.gltf.images.append(image)
        Image.fromarray(np.full((43, 59, 4), [12, 45, 67, 128], dtype=np.uint8)).save(self.directory / "other.png")
        texture = deepcopy(doc.gltf.textures[0]); texture.source = 1
        doc.gltf.textures.append(texture)
        material = deepcopy(doc.gltf.materials[0]); material.pbrMetallicRoughness.baseColorTexture.index = 1
        doc.gltf.materials.append(material)
        primitive = deepcopy(doc.gltf.meshes[0].primitives[0]); primitive.material = 1
        doc.gltf.meshes[0].primitives.append(primitive)
        _save(doc, self.first)
        report = merge_textures([self.first, self.second], self.output, pad=2, deduplicate=True)
        self.assertEqual(len(report["source_images"]), 3)
        self.assert_models(self.first, Path(report["models"][0]["output"]))
        new = _Document(Path(report["models"][0]["output"]))
        self.assertEqual({image.uri for image in new.gltf.images}, {"shared_atlas.png"})

    def test_skinned_animated_glb_keeps_weights_animation_and_nodes(self):
        doc = _Document(self.first)
        g = doc.gltf
        primitive = g.meshes[0].primitives[0]
        count = len(doc.accessor(primitive.attributes.POSITION))
        joints = np.zeros((count, 4), dtype="<u2")
        weights = np.zeros((count, 4), dtype="<f4"); weights[:, 0] = 1
        joint_id = doc.add_accessor(joints); g.accessors[joint_id].type = "VEC4"; g.accessors[joint_id].componentType = 5123
        weight_id = doc.add_accessor(weights); g.accessors[weight_id].type = "VEC4"
        primitive.attributes.JOINTS_0, primitive.attributes.WEIGHTS_0 = joint_id, weight_id
        g.nodes.append(Node(name="bone"))
        g.nodes[0].children = (g.nodes[0].children or []) + [len(g.nodes) - 1]
        inverse = doc.add_accessor(np.eye(4, dtype="<f4").reshape(1, 16))
        g.accessors[inverse].type = "MAT4"
        g.skins.append(Skin(joints=[len(g.nodes) - 1], inverseBindMatrices=inverse))
        mesh_node = next(node for node in g.nodes if node.mesh == 0); mesh_node.skin = 0
        times = doc.add_accessor(np.array([[0], [1]], dtype="<f4")); g.accessors[times].type = "SCALAR"
        translations = doc.add_accessor(np.array([[0, 0, 0], [1, 2, 3]], dtype="<f4")); g.accessors[translations].type = "VEC3"
        g.animations.append(Animation(name="move", samplers=[AnimationSampler(input=times, output=translations)],
                                     channels=[AnimationChannel(sampler=0, target=AnimationChannelTarget(node=len(g.nodes) - 1, path="translation"))]))
        source = self.directory / "animated.glb"
        _save(doc, source)
        report = merge_textures([source, self.second], self.output, deduplicate=True)
        target = Path(report["models"][0]["output"])
        self.assert_models(source, target)
        before, after = _Document(source), _Document(target)
        for old, new in zip(before.gltf.animations[0].samplers, after.gltf.animations[0].samplers):
            np.testing.assert_array_equal(before.accessor(old.input), after.accessor(new.input))
            np.testing.assert_array_equal(before.accessor(old.output), after.accessor(new.output))
        # MAT4 is copied as a buffer payload; compare its bytes without decoding.
        old = before.gltf.accessors[before.gltf.skins[0].inverseBindMatrices]
        new = after.gltf.accessors[after.gltf.skins[0].inverseBindMatrices]
        self.assertEqual(before.view(old.bufferView), after.view(new.bufferView))

    def test_same_names_external_texture_path_and_deterministic_output(self):
        other = self.directory / "sub" / "first.gltf"
        _export(other)
        atlas = self.directory / "shared images" / "paint atlas.png"
        report = merge_textures([self.first, other], self.output, texture_out=atlas, deduplicate=True, packing="dense")
        self.assertEqual([Path(m["output"]).name for m in report["models"]], ["first.gltf", "first_2.gltf"])
        png, blob = atlas.read_bytes(), (self.output / "first.bin").read_bytes()
        merge_textures([self.first, other], self.output, texture_out=atlas, deduplicate=True, packing="dense")
        self.assertEqual(png, atlas.read_bytes())
        self.assertEqual(blob, (self.output / "first.bin").read_bytes())
        self.assert_models(other, self.output / "first_2.gltf")
        self.assertEqual(json.loads((self.output / "first.gltf").read_text())["images"][0]["uri"],
                         "../shared%20images/paint%20atlas.png")

    def test_invalid_inputs_and_cross_model_overwrites_are_rejected_before_writing(self):
        cases = [dict(input_models=[]), dict(input_models=[self.first, self.first]),
                 dict(output_dir=self.directory), dict(texture_out=self.second.with_suffix(".png")),
                 dict(report_out=self.first.with_suffix(".bin")), dict(pad=-1),
                 dict(similarity_threshold=float("nan")), dict(similarity_max_error=256),
                 dict(texture_out=self.output / "wrong.jpg")]
        for changes in cases:
            kwargs = dict(input_models=[self.first, self.second], output_dir=self.output)
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                merge_textures(**kwargs)
            self.assertFalse(self.output.exists())
        doc = _Document(self.second)
        doc.gltf.extensionsUsed = ["KHR_draco_mesh_compression"]
        _save(doc, self.second)
        with self.assertRaisesRegex(ValueError, "extensions"):
            merge_textures([self.first, self.second], self.output)
        self.assertFalse(self.output.exists())

    def test_cli_and_disabled_merging(self):
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(main(["--input", str(self.first), str(self.second), "--output-dir", str(self.output),
                                   "--no-tex-pot", "--pack", "dense"]), 0)
        report = json.loads((self.output / "merge_report.json").read_text())
        self.assertEqual(report["patches"], report["unique_patches"])
        self.assertEqual(report["cross_model_merged_patches"], 0)
        self.assertFalse(report["pot"])


if __name__ == "__main__":
    unittest.main()
