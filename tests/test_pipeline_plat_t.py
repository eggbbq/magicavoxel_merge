from __future__ import annotations

import unittest

import numpy as np

from btp_vox.pipeline import _compute_plat_base_half_height
from btp_vox.voxio import VoxModel, VoxScene


def _model(name: str, voxels: np.ndarray) -> VoxModel:
    return VoxModel(
        name=name,
        size=tuple(int(value) for value in voxels.shape),
        voxels=voxels,
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )


def _scene(*models: VoxModel) -> VoxScene:
    return VoxScene(
        models=list(models),
        palette_rgba=np.zeros((256, 4), dtype=np.uint8),
        nodes=[],
        root_node_ids=[],
    )


class PlatTopOffsetTests(unittest.TestCase):
    def test_standalone_at_t_uses_model_height_fallback(self) -> None:
        voxels = np.zeros((4, 4, 1), dtype=np.int32)
        voxels[:, :, 0] = 205
        scene = _scene(_model("grass@t", voxels))

        offset = _compute_plat_base_half_height(scene, 0, 0.02)

        self.assertAlmostEqual(offset, 0.01)

    def test_at_t_uses_occupied_height_of_separate_base_model(self) -> None:
        plate = np.ones((4, 4, 1), dtype=np.int32)
        base = np.zeros((4, 4, 8), dtype=np.int32)
        base[:, :, 2:5] = 99
        scene = _scene(_model("grass@t", plate), _model("grass", base))

        offset = _compute_plat_base_half_height(scene, 0, 0.02)

        self.assertAlmostEqual(offset, 0.03)

    def test_legacy_plat_t_still_uses_separate_base_model(self) -> None:
        plate = np.ones((4, 4, 1), dtype=np.int32)
        base = np.zeros((4, 4, 8), dtype=np.int32)
        base[:, :, 1:5] = 77
        scene = _scene(_model("grass-plat-t", plate), _model("grass", base))

        offset = _compute_plat_base_half_height(scene, 0, 0.02)

        self.assertAlmostEqual(offset, 0.04)

    def test_other_cutouts_are_not_used_as_base_models(self) -> None:
        first_plate = np.full((4, 4, 1), 220, dtype=np.int32)
        second_plate = np.full((4, 4, 6), 180, dtype=np.int32)
        scene = _scene(_model("grass@t", first_plate), _model("grass@cutout", second_plate))

        offset = _compute_plat_base_half_height(scene, 0, 0.02)

        self.assertAlmostEqual(offset, 0.01)


if __name__ == "__main__":
    unittest.main()
