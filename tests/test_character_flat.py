from __future__ import annotations

import unittest
from math import sqrt

import numpy as np

from btp_vox.pipeline import (
    _collapse_auto_wrappers,
    _collapse_same_name_single_child,
    _flatten_character_roots,
    _quat_to_mat3,
)


def _world_matrices(nodes: list[dict], roots: list[int]) -> dict[int, np.ndarray]:
    matrices = {}

    def walk(nid: int, parent: np.ndarray) -> None:
        node = nodes[nid]
        local = np.eye(4)
        local[:3, :3] = _quat_to_mat3(node.get("rotation", (0, 0, 0, 1)))
        local[:3, 3] = node.get("translation", (0, 0, 0))
        matrices[nid] = parent @ local
        for child in node.get("children", []):
            walk(child, matrices[nid])

    for root in roots:
        walk(root, np.eye(4))
    return matrices


class CharacterFlatTests(unittest.TestCase):
    def test_flatten_detaches_old_parents_and_preserves_world_transforms(self) -> None:
        turn = (0, 0, sqrt(0.5), sqrt(0.5))
        nodes = [
            {"name": "hero", "translation": (10, 20, 30), "rotation": turn, "children": [1, 3]},
            {"name": "right-hand", "translation": (2, 3, 4), "rotation": turn, "children": [2]},
            {"name": "left-hand", "mesh": 0, "translation": (1, 0, 0)},
            {"name": "body", "translation": (0, 0, 5), "children": [4]},
            {"name": "body", "mesh": 1, "translation": (0, 1, 0), "children": [5]},
            {"name": "trn_5", "translation": (3, 0, 0), "children": [6]},
            {"name": "attachment", "mesh": 2, "rotation": turn},
        ]
        original = _world_matrices(nodes, [0])
        nodes, roots = _flatten_character_roots(nodes, [0])
        nodes, roots = _collapse_auto_wrappers(nodes, roots)
        nodes, roots = _collapse_same_name_single_child(nodes, roots)

        self.assertEqual(roots, [0])
        self.assertCountEqual(nodes[0]["children"], [2, 4, 6])
        # Blender checks every node, including nodes outside the active scene.
        parents = {}
        for pid, node in enumerate(nodes):
            for child in node.get("children", []):
                self.assertNotIn(child, parents, f"Node {child} has multiple parents")
                parents[child] = pid
        self.assertEqual(parents, {2: 0, 4: 0, 6: 0})
        flattened = _world_matrices(nodes, roots)
        for nid in (2, 4, 6):
            np.testing.assert_allclose(flattened[nid], original[nid], atol=1e-5)

    def test_separate_characters_can_share_a_mesh_without_sharing_nodes(self) -> None:
        nodes = [
            {"name": "hero-a", "children": [1]},
            {"name": "arm-a", "children": [2]},
            {"name": "hand-a", "mesh": 0},
            {"name": "hero-b", "translation": (10, 0, 0), "children": [4]},
            {"name": "arm-b", "children": [5]},
            {"name": "hand-b", "mesh": 0},
        ]
        nodes, roots = _flatten_character_roots(nodes, [0, 3])

        self.assertEqual(roots, [0, 3])
        self.assertEqual(nodes[0]["children"], [2])
        self.assertEqual(nodes[3]["children"], [5])
        self.assertEqual(nodes[1]["children"], [])
        self.assertEqual(nodes[4]["children"], [])
        self.assertEqual(nodes[2]["mesh"], nodes[5]["mesh"])


if __name__ == "__main__":
    unittest.main()
