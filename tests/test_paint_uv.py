from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from pygltflib import GLTF2

from btp_vox.atlas import build_atlas
from btp_vox.btp_mesher import MesherResult, build_quads
from btp_vox.cli import build_parser, main
from btp_vox.paint_uv import build_paint_atlas, unique_scene_models
from btp_vox.pipeline import (
    AtlasOptions, PipelineOptions, _build_model_meshes,
    _build_plat_cutout_quads_for_model, convert,
)
from btp_vox.voxio import VoxModel, VoxNode, VoxScene


def _scene(voxels: np.ndarray, name: str = "part") -> VoxScene:
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, :3] = np.arange(256)[:, None] * np.array([1, 17, 37]) % 256
    palette[:, 3] = 255
    model = VoxModel(name, voxels.shape, voxels, (0, 0, 0), (0, 0, 0, 1))
    return VoxScene([model], palette, [VoxNode("shp", name, model_ids=[0])], [0])


def _pattern(size=(3, 4, 5)) -> np.ndarray:
    return (np.arange(np.prod(size)).reshape(size) % 254 + 1).astype(np.int32)


def _corners(uv) -> np.ndarray:
    a = np.array([uv.u0, uv.v0])
    b = np.array([uv.u1, uv.v1])
    d = np.array([uv.u2, uv.v2])
    return np.array([a, b, b + d - a, d])


class PaintUVTests(unittest.TestCase):
    def assert_surface_colors(self, scene, mesher, result, scale) -> None:
        pixels = np.asarray(Image.open(io.BytesIO(result.texture_png)).convert("RGBA"))
        occupied = np.zeros(pixels.shape[:2], dtype=bool)
        for midx, quads in enumerate(mesher.quads_per_model):
            for qid, quad in enumerate(quads):
                points = _corners(result.quad_uvs[(midx, qid)]) * (result.width, result.height)
                np.testing.assert_allclose(points, np.round(points), atol=1e-6)
                x0, y0 = np.round(points.min(axis=0)).astype(int)
                x1, y1 = np.round(points.max(axis=0)).astype(int)
                self.assertFalse(occupied[y0:y1, x0:x1].any(), "Surface texels overlap")
                occupied[y0:y1, x0:x1] = True
                self.assertEqual((x1 - x0) * (y1 - y0), quad.size_u * quad.size_v * scale**2)
                vertices = np.asarray(quad.verts)
                for v in range(quad.size_v):
                    for u in range(quad.size_u):
                        a, b = (u + 0.5) / quad.size_u, (v + 0.5) / quad.size_v
                        # Read the independent source voxel immediately inside the face.
                        pos = vertices[0] + a * (vertices[1] - vertices[0]) + b * (vertices[3] - vertices[0])
                        cell = np.floor(pos - np.asarray(quad.normal) * 0.5).astype(int)
                        index = scene.models[midx].voxels[tuple(cell)]
                        expected = scene.palette_rgba[index - 1]
                        xy = np.floor(points[0] + a * (points[1] - points[0]) +
                                      b * (points[3] - points[0])).astype(int)
                        np.testing.assert_array_equal(pixels[xy[1], xy[0]], expected)

    def test_box_colors_density_and_unique_surface_texels(self):
        scene = _scene(_pattern())
        mesher = build_quads(scene)
        atlas, layout = build_paint_atlas(scene, mesher, texel_scale=3, pad=2)
        self.assertEqual(len(layout.charts), 1)
        self.assert_surface_colors(scene, mesher, atlas, 3)

    def test_box_net_glues_five_edges_without_mirroring(self):
        scene = _scene(_pattern())
        mesher = build_quads(scene)
        atlas, _ = build_paint_atlas(scene, mesher)
        maps = {}
        for qid, q in enumerate(mesher.quads_per_model[0]):
            points = _corners(atlas.quad_uvs[(0, qid)])
            maps[(q.axis, q.normal_sign)] = {tuple(v): uv for v, uv in zip(q.verts, points)}
        for face_a, face_b in [((0, -1), (1, 1)), ((1, 1), (0, 1)),
                               ((0, 1), (1, -1)), ((2, 1), (1, 1)), ((2, -1), (1, 1))]:
            shared = maps[face_a].keys() & maps[face_b].keys()
            self.assertEqual(len(shared), 2)
            for vertex in shared:
                np.testing.assert_allclose(maps[face_a][vertex], maps[face_b][vertex])
        # All outward faces have the same UV orientation relative to their normal.
        for qid, q in enumerate(mesher.quads_per_model[0]):
            uv = _corners(atlas.quad_uvs[(0, qid)])
            det = np.linalg.det(np.stack((uv[1] - uv[0], uv[3] - uv[0])))
            verts = np.asarray(q.verts)
            orientation = np.dot(np.cross(verts[1] - verts[0], verts[3] - verts[0]), q.normal)
            self.assertGreater(det * orientation, 0)

    def test_recesses_steps_disconnected_parts_and_cavities_do_not_overlap(self):
        for kind in ("step", "cavity", "disconnected", "checker"):
            vox = _pattern((5, 6, 7))
            if kind == "step":
                vox[2:, 2:, 3:] = 0
            elif kind == "cavity":
                vox[1:4, 1:5, 1:6] = 0
            elif kind == "disconnected":
                vox[2:4, :, :] = 0
            else:
                vox[np.indices(vox.shape).sum(axis=0) % 2 == 0] = 0
            with self.subTest(kind=kind):
                scene = _scene(vox)
                mesher = build_quads(scene)
                atlas, layout = build_paint_atlas(scene, mesher, texel_scale=2, pad=2)
                self.assertGreater(len(layout.charts), 1)
                self.assert_surface_colors(scene, mesher, atlas, 2)

    def test_same_plane_remains_continuous_across_mesher_quad_boundaries(self):
        vox = np.ones((3, 3, 1), dtype=np.int32)
        vox[1:, 1:, :] = 0
        scene = _scene(vox)
        mesher = build_quads(scene)
        atlas, _ = build_paint_atlas(scene, mesher)
        top = [(i, q) for i, q in enumerate(mesher.quads_per_model[0]) if q.axis == 2 and q.normal_sign == 1]
        self.assertGreater(len(top), 1)
        vertices = {}
        shared = 0
        for qid, q in top:
            for vertex, uv in zip(q.verts, _corners(atlas.quad_uvs[(0, qid)])):
                if vertex in vertices:
                    shared += 1
                    np.testing.assert_allclose(uv, vertices[vertex])
                vertices[vertex] = uv
        self.assertGreater(shared, 0)

    def test_shape_instances_have_independent_paint_regions_without_mutating_scene(self):
        original = _scene(_pattern(), "arm")
        original.nodes = [VoxNode("grp", "hero", children=[1, 2]),
                          VoxNode("shp", "left", model_ids=[0]),
                          VoxNode("shp", "right", model_ids=[0])]
        scene = unique_scene_models(original)
        self.assertEqual(len(original.models), 1)
        self.assertEqual(original.nodes[2].model_ids, [0])
        self.assertEqual(scene.nodes[1].model_ids, [0])
        self.assertEqual(scene.nodes[2].model_ids, [1])
        mesher = build_quads(scene)
        result, _ = build_paint_atlas(scene, mesher)
        self.assert_surface_colors(scene, mesher, result, 1)

    def test_padding_does_not_change_core_colors(self):
        scene = _scene(_pattern())
        mesher = build_quads(scene)
        atlas, _ = build_paint_atlas(scene, mesher, texel_scale=2, pad=4)
        self.assert_surface_colors(scene, mesher, atlas, 2)
        pixels = np.asarray(Image.open(io.BytesIO(atlas.texture_png)))
        self.assertTrue((pixels[:, :, 3] == 255).any())

    def test_cutout_keeps_transparency_and_rotated_basis(self):
        vox = _pattern((3, 4, 5))
        vox[1, 1, :] = 0
        scene = _scene(vox)
        for basis in ((0, 1, 1, 1, 2, 1), (2, -1, 0, 1, 1, -1), (1, -1, 2, -1, 0, 1)):
            quads = _build_plat_cutout_quads_for_model(scene, 0, mode="f", front_basis=basis)
            mesher = MesherResult([quads])
            result, _ = build_paint_atlas(scene, mesher, texel_scale=2, alpha="rgba")
            pixels = np.asarray(Image.open(io.BytesIO(result.texture_png)))
            for qid, q in enumerate(quads):
                points = _corners(result.quad_uvs[(0, qid)]) * (result.width, result.height)
                for v in range(q.size_v):
                    for u in range(q.size_u):
                        a, b = (u + 0.5) / q.size_u, (v + 0.5) / q.size_v
                        xy = np.floor(points[0] + a * (points[1] - points[0]) + b * (points[3] - points[0])).astype(int)
                        index = q.colors[v, u]
                        if index:
                            np.testing.assert_array_equal(pixels[xy[1], xy[0]], scene.palette_rgba[index - 1])
                        else:
                            self.assertEqual(pixels[xy[1], xy[0], 3], 0)

    def test_layout_is_deterministic_and_fixed_size_is_respected(self):
        scene = _scene(_pattern())
        mesher = build_quads(scene)
        first, layout = build_paint_atlas(scene, mesher, fixed_size=(128, 96), pot=True, square=True)
        second, _ = build_paint_atlas(scene, mesher, fixed_size=(128, 96))
        self.assertEqual((first.width, first.height), (128, 96))
        self.assertEqual(first, second)
        self.assertEqual(layout.width, 128)
        with self.assertRaisesRegex(ValueError, "too small"):
            build_paint_atlas(scene, mesher, fixed_size=(2, 2))

    def test_empty_scene_and_empty_model_export_valid_blank_atlas(self):
        for scene in (_scene(np.zeros((3, 4, 5), dtype=np.int32)), VoxScene([], np.zeros((256, 4), dtype=np.uint8), [], [])):
            atlas, layout = build_paint_atlas(scene, build_quads(scene))
            self.assertEqual((atlas.width, atlas.height), (1, 1))
            self.assertEqual(layout.charts, [])

    def test_svg_escapes_names_and_json_matches_png_even_with_flip_v(self):
        scene = _scene(_pattern(), 'head<&"')
        atlas, layout = build_paint_atlas(scene, build_quads(scene))
        with tempfile.TemporaryDirectory() as directory:
            svg, meta = Path(directory) / "guide.svg", Path(directory) / "islands.json"
            layout.write_template(svg, flip_v=True)
            layout.write_json(meta, flip_v=True)
            root = ET.fromstring(svg.read_text())
            self.assertEqual(root.get("width"), str(atlas.width))
            data = json.loads(meta.read_text())
            self.assertTrue(data["uv_flip_v"])
            self.assertEqual(data["charts"][0]["name"], 'head<&"')
            for q in data["charts"][0]["quads"]:
                expected = _corners(atlas.quad_uvs[(0, q["quad"])]) * (atlas.width, atlas.height)
                np.testing.assert_allclose(q["pixels"], expected)

    def test_paint_ignores_alias_compression_solid_and_inset_options(self):
        scene = _scene(_pattern(), "part@lrfk@tb")
        with tempfile.TemporaryDirectory() as directory, patch("btp_vox.pipeline.load_scene", return_value=scene):
            first, second = Path(directory) / "first.gltf", Path(directory) / "second.gltf"
            for path, atlas in [(first, AtlasOptions()), (second, AtlasOptions(
                    style="solid", reuse_subrects=True, compress_solid_quads=True,
                    face_alias_uv_remap=True, inset=0.5, layout="global"))]:
                convert("source.vox", path, output_format="gltf", options=PipelineOptions(uv_mode="paint", atlas=atlas))
            self.assertEqual(first.with_suffix(".bin").read_bytes(), second.with_suffix(".bin").read_bytes())
            self.assertEqual(first.with_suffix(".png").read_bytes(), second.with_suffix(".png").read_bytes())

    def test_mesh_geometry_is_identical_and_uv2_copy_follows_paint_uvs(self):
        scene = _scene(_pattern())
        mesher = build_quads(scene)
        paint, _ = build_paint_atlas(scene, mesher)
        atlas = build_atlas(scene, mesher)
        old, _ = _build_model_meshes(scene, mesher, atlas, scale=0.02, flip_v=False)
        new, _ = _build_model_meshes(scene, mesher, paint, scale=0.02, flip_v=True, export_uv2=True)
        for key in ("positions", "normals", "indices"):
            np.testing.assert_array_equal(old[0][key], new[0][key])
        np.testing.assert_array_equal(new[0]["texcoords"], new[0]["texcoords1"])

    def test_glb_embeds_texture_and_cutout_material_survives(self):
        scene = _scene(_pattern(), "part@t")
        with tempfile.TemporaryDirectory() as directory, patch("btp_vox.pipeline.load_scene", return_value=scene):
            path = Path(directory) / "part.glb"
            convert("source.vox", path, options=PipelineOptions(uv_mode="paint", export_uv2=True))
            doc = GLTF2().load(str(path))
            self.assertEqual(doc.materials[0].alphaMode, "MASK")
            self.assertIsNotNone(doc.images[0].bufferView)
            self.assertIsNotNone(doc.meshes[0].primitives[0].attributes.TEXCOORD_1)

    def test_cli_defaults_unchanged_and_template_requires_paint(self):
        args = build_parser().parse_args(["--input", "in.vox", "--output", "out.glb"])
        self.assertEqual(args.uv_mode, "atlas")
        self.assertEqual(args.tex_texel_scale, 1)
        self.assertTrue(args.tex_reuse_subrects)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            main(["--input", "in.vox", "--output", "out.glb", "--uv-template-out", "guide.svg"])
        with self.assertRaisesRegex(ValueError, "requires"):
            convert("does-not-exist.vox", "out.glb", uv_template_out="guide.svg")


if __name__ == "__main__":
    unittest.main()
