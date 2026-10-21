from __future__ import annotations

import itertools
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from btp_vox.pipeline import AtlasOptions, PipelineOptions, convert
from btp_vox.voxio import load_scene


def _i32(*values: int) -> bytes:
    return struct.pack("<" + "i" * len(values), *values)


def _dict(values: dict[str, str]) -> bytes:
    result = _i32(len(values))
    for key, value in values.items():
        for text in (key, value):
            encoded = text.encode("utf-8")
            result += _i32(len(encoded)) + encoded
    return result


def _chunk(kind: bytes, content: bytes, children: bytes = b"") -> bytes:
    return kind + _i32(len(content), len(children)) + content + children


def _trn(nid: int, name: str, child: int, translation: str, rotation: int = 4) -> bytes:
    return _chunk(b"nTRN", _i32(nid) + _dict({"_name": name}) +
                  _i32(child, -1, 0, 1) + _dict({"_t": translation, "_r": str(rotation)}))


def _grp(nid: int, *children: int) -> bytes:
    return _chunk(b"nGRP", _i32(nid) + _dict({}) + _i32(len(children), *children))


def _shp(nid: int, *models: int) -> bytes:
    return _chunk(b"nSHP", _i32(nid) + _dict({}) + _i32(len(models)) +
                  b"".join(_i32(mid) + _dict({}) for mid in models))


def _model(size: tuple[int, int, int], low: tuple[int, int, int], high: tuple[int, int, int]) -> bytes:
    cells = list(itertools.product(*(range(a, b) for a, b in zip(low, high))))
    return (_chunk(b"SIZE", _i32(*size)) +
            _chunk(b"XYZI", _i32(len(cells)) + b"".join(bytes((*p, 1)) for p in cells)))


def _matrix(rotation: np.ndarray, translation: tuple[int, int, int]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _sort_points(points: np.ndarray) -> np.ndarray:
    rounded = np.unique(np.round(points, 5), axis=0)
    return rounded[np.lexsort(rounded.T[::-1])]


def _exported_parts(path: Path) -> tuple[dict, list[tuple[list[str], np.ndarray]]]:
    gltf = json.loads(path.read_text())
    binary = (path.parent / gltf["buffers"][0]["uri"]).read_bytes()
    parts = []
    parents = {}
    for pid, node in enumerate(gltf["nodes"]):
        for cid in node.get("children", []):
            if cid in parents:
                raise AssertionError(f"Node {cid} has more than one parent")
            parents[cid] = pid

    def walk(nid: int, parent: np.ndarray, names: list[str]) -> None:
        node = gltf["nodes"][nid]
        # Independent quaternion-to-matrix formula; source expectations below
        # use explicit integer matrices, rather than exporter transform helpers.
        x, y, z, w = node.get("rotation", (0, 0, 0, 1))
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        local = np.eye(4)
        local[:3, :3] = np.eye(3) + 2 * w * cross + 2 * cross @ cross
        local[:3, 3] = node.get("translation", (0, 0, 0))
        world = parent @ local
        names = names + [node.get("name", "")]
        if "mesh" in node:
            prim = gltf["meshes"][node["mesh"]]["primitives"][0]
            accessor = gltf["accessors"][prim["attributes"]["POSITION"]]
            view = gltf["bufferViews"][accessor["bufferView"]]
            offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            points = np.frombuffer(binary, dtype="<f4", count=accessor["count"] * 3, offset=offset).reshape(-1, 3)
            parts.append((names, points @ world[:3, :3].T + world[:3, 3]))
        for child in node.get("children", []):
            walk(child, world, names)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4), [])
    return gltf, parts


class SceneTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "character.vox"
        self.model_data = [((5, 6, 9), (1, 2, 1), (4, 5, 7)),
                           ((3, 5, 4), (0, 1, 2), (2, 5, 4))]
        chunks = b"".join(_model(*model) for model in self.model_data)
        chunks += (_trn(0, "hero", 1, "10 -20 30", 17) + _grp(1, 2, 6, 10) +
                   _trn(2, "arm-a", 3, "3 4 5", 40) + _grp(3, 4) +
                   _trn(4, "tip", 5, "1 2 3", 17) + _shp(5, 0) +
                   _trn(6, "arm-b", 7, "-5 1 -2") + _grp(7, 8) +
                   _trn(8, "tip", 9, "2 -3 1", 40) + _shp(9, 0) +
                   _trn(10, "body", 11, "1 2 3") + _shp(11, 1))
        self.source.write_bytes(b"VOX " + _i32(200) + _chunk(b"MAIN", b"", chunks))

        identity = np.eye(3)
        rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        root = _matrix(rz, (10, -20, 30))
        self.world = [root @ _matrix(rx, (3, 4, 5)) @ _matrix(rz, (1, 2, 3)),
                      root @ _matrix(identity, (-5, 1, -2)) @ _matrix(rx, (2, -3, 1)),
                      root @ _matrix(identity, (1, 2, 3))]

    def _expected(self, handedness: str) -> list[np.ndarray]:
        basis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        if handedness == "left":
            basis[2] *= -1
        expected = []
        for mid, world in zip((0, 0, 1), self.world):
            size, low, high = self.model_data[mid]
            corners = np.array(list(itertools.product(*zip(low, high))))
            corners = corners - np.array(size) // 2
            points = corners @ world[:3, :3].T + world[:3, 3]
            expected.append(_sort_points(points @ basis.T * 0.02))
        return sorted(expected, key=lambda p: tuple(p.ravel()))

    def _convert(self, **kwargs) -> tuple[dict, list[tuple[list[str], np.ndarray]]]:
        path = self.directory / "character.gltf"
        convert(self.source, path, output_format="gltf", options=PipelineOptions(
            scale=0.02, atlas=AtlasOptions(reuse_subrects=False), **kwargs))
        return _exported_parts(path)

    def test_all_modes_preserve_world_geometry_for_every_pivot_and_handedness(self) -> None:
        modes = ({}, {"no_merge_nodes": True}, {"character_apart": True}, {"character_flat": True})
        uv_modes = ({"uv_mode": "atlas"}, {"uv_mode": "paint"},
                    {"uv_mode": "paint", "paint_unwrap": "surface"})
        for mode in (dict(m, **uv) for uv in uv_modes for m in modes):
            for pivot in ("corner", "center", "bottom_center"):
                for handedness in ("right", "left"):
                    with self.subTest(mode=mode, pivot=pivot, handedness=handedness):
                        _, parts = self._convert(pivot=pivot, handedness=handedness, **mode)
                        actual = sorted((_sort_points(p) for _, p in parts), key=lambda p: tuple(p.ravel()))
                        self.assertEqual(len(actual), 3)  # Same model instantiated in two arms.
                        for points, expected in zip(actual, self._expected(handedness)):
                            np.testing.assert_allclose(points, expected, atol=2e-6)

    def test_character_apart_retains_named_groups(self) -> None:
        _, parts = self._convert(pivot="bottom_center", character_apart=True)
        paths = [tuple(n for i, n in enumerate(names) if i == 0 or n != names[i - 1]) for names, _ in parts]
        self.assertCountEqual(paths, [("hero", "arm-a", "tip"), ("hero", "arm-b", "tip"), ("hero", "body")])

    def test_character_flat_explicitly_removes_part_groups(self) -> None:
        _, parts = self._convert(pivot="bottom_center", character_flat=True)
        self.assertTrue(all(len(names) == 2 and names[0] == "hero" for names, _ in parts))

    def test_no_merge_retains_source_node_links(self) -> None:
        gltf, _ = self._convert(pivot="bottom_center", no_merge_nodes=True)
        scene = load_scene(self.source)
        self.assertEqual(len(gltf["nodes"]), len(scene.nodes))
        self.assertEqual(gltf["scenes"][0]["nodes"], scene.root_node_ids)
        for node, source in zip(gltf["nodes"], scene.nodes):
            self.assertEqual(node.get("children", []), source.children)

    def test_loader_accumulates_rotated_parent_translation(self) -> None:
        scene = load_scene(self.source)
        # load_scene stores the last instance's per-model transform.
        np.testing.assert_allclose(scene.models[0].translation, self.world[1][:3, 3], atol=1e-6)

    def test_multi_model_shape_preserves_each_model_origin(self) -> None:
        chunks = b"".join(_model(*model) for model in self.model_data)
        chunks += _trn(0, "combined", 1, "0 0 0") + _shp(1, 0, 1)
        self.source.write_bytes(b"VOX " + _i32(200) + _chunk(b"MAIN", b"", chunks))
        _, parts = self._convert(pivot="bottom_center", no_merge_nodes=True)
        self.assertEqual(len(parts), 1)
        expected = []
        for size, low, high in self.model_data:
            corners = np.array(list(itertools.product(*zip(low, high)))) - np.array(size) // 2
            expected.extend(corners[:, [0, 2, 1]] * [0.02, 0.02, -0.02])
        np.testing.assert_allclose(_sort_points(parts[0][1]), _sort_points(np.array(expected)), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
