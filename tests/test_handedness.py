from __future__ import annotations

import unittest
from math import sqrt

import numpy as np

from btp_vox.cli import build_parser
from btp_vox.pipeline import _quat_rotate_vec, _to_y_up, _to_y_up_nodes


def _mesh() -> dict:
    return {
        "positions": np.asarray(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 3.0],
                [1.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        ),
        "normals": np.asarray([[0.0, -1.0, 0.0]] * 3, dtype=np.float32),
        "indices": np.asarray([0, 1, 2], dtype=np.uint32),
        "translation": (4.0, 5.0, 6.0),
        "rotation": (0.0, 0.0, 0.0, 1.0),
    }


class HandednessTests(unittest.TestCase):
    def test_cli_defaults_to_right_handed_output(self) -> None:
        args = build_parser().parse_args(["--input", "in.vox", "--output", "out.gltf"])
        self.assertEqual(args.handedness, "right")

    def test_right_handed_mesh_uses_pure_z_up_to_y_up_rotation(self) -> None:
        result = _to_y_up([_mesh()], handedness="right")[0]

        np.testing.assert_allclose(result["positions"][0], [1.0, 3.0, -2.0])
        np.testing.assert_allclose(result["normals"][0], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(result["indices"], [0, 1, 2])
        np.testing.assert_allclose(result["translation"], [4.0, 6.0, -5.0])

    def test_left_handed_mesh_preserves_legacy_z_mirror_and_winding(self) -> None:
        result = _to_y_up([_mesh()], handedness="left")[0]

        np.testing.assert_allclose(result["positions"][0], [1.0, 3.0, 2.0])
        np.testing.assert_allclose(result["normals"][0], [0.0, 0.0, -1.0])
        np.testing.assert_array_equal(result["indices"], [0, 2, 1])
        np.testing.assert_allclose(result["translation"], [4.0, 6.0, 5.0])

    def test_node_translations_follow_selected_handedness(self) -> None:
        nodes = [{"name": "node", "translation": (4.0, 5.0, 6.0)}]

        right = _to_y_up_nodes(nodes, [], handedness="right")[0]
        left = _to_y_up_nodes(nodes, [], handedness="left")[0]

        np.testing.assert_allclose(right["translation"], [4.0, 6.0, -5.0])
        np.testing.assert_allclose(left["translation"], [4.0, 6.0, 5.0])

    def test_node_rotations_follow_selected_handedness(self) -> None:
        quarter_turn_around_source_z = (0.0, 0.0, sqrt(0.5), sqrt(0.5))
        nodes = [{"name": "node", "rotation": quarter_turn_around_source_z}]

        right = _to_y_up_nodes(nodes, [], handedness="right")[0]
        left = _to_y_up_nodes(nodes, [], handedness="left")[0]

        np.testing.assert_allclose(
            _quat_rotate_vec(right["rotation"], (1.0, 0.0, 0.0)),
            [0.0, 0.0, -1.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            _quat_rotate_vec(left["rotation"], (1.0, 0.0, 0.0)),
            [0.0, 0.0, 1.0],
            atol=1e-6,
        )

    def test_invalid_handedness_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "handedness"):
            _to_y_up([_mesh()], handedness="invalid")


if __name__ == "__main__":
    unittest.main()
