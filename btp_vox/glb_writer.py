"""Wrapper utilities for writing glTF/GLB scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from magicavoxel_merge.glb import write_glb_scene


def write_meshes(
    output_glb: str | Path,
    *,
    meshes: Iterable[dict[str, np.ndarray]],
    texture_png: bytes | None,
    texture_path: str | None,
    name_prefix: str | None = None,
) -> None:
    """Write meshes to a GLB, embedding or referencing the texture."""

    write_glb_scene(
        str(output_glb),
        meshes=list(meshes),
        texture_png=texture_png,
        texture_uri=texture_path,
        name_prefix=name_prefix,
    )
