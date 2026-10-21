"""Pack multiple glTF/GLB models into one shared colour texture.

Models remain separate glTF files. The existing repacker supplies patch
extraction, exact/similar matching, rotation/mirroring and UV remapping.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote

import numpy as np
from PIL import Image

from .atlas import _image_to_bytes
from .repack import (
    _Document, _collect_jobs, _layout_patches, _optimization_report, _patches,
    _rewrite_jobs, _validate_options, _write_document,
)


def merge_textures(
    input_models: Sequence[str | Path], output_dir: str | Path, *,
    texture_out: str | Path | None = None, report_out: str | Path | None = None,
    pad: int = 2, pot: bool = False, deduplicate: bool = False,
    merge_similar: bool = False, similarity_threshold: float = 0.98,
    similarity_max_error: int = 16, compress_solid: bool = False,
    packing: str = "shelf",
) -> dict:
    """Write independent models sharing one PNG, preserving all input files.

    Supports multiple base-colour images per model. Other texture channels,
    transformed/repeating UVs and compressed meshes use the same restrictions
    as the single-model repacker. Input order determines stable tie-breaking.
    """
    _validate_options(pad=pad, packing=packing, similarity_threshold=similarity_threshold,
                      similarity_max_error=similarity_max_error)
    if isinstance(input_models, (str, Path)) or not input_models:
        raise ValueError("Provide a non-empty list of input .gltf/.glb models")
    sources = [Path(path).resolve() for path in input_models]
    if len(set(sources)) != len(sources):
        raise ValueError("Input models must be distinct")
    if any(path.suffix.lower() not in (".gltf", ".glb") for path in sources):
        raise ValueError("Input models must be .gltf or .glb")
    directory = Path(output_dir).resolve()
    atlas_path = Path(texture_out).resolve() if texture_out else directory / "shared_atlas.png"
    report_path = Path(report_out).resolve() if report_out else directory / "merge_report.json"
    if atlas_path.suffix.lower() != ".png":
        raise ValueError("Shared texture output must be .png")

    # Distinguish files with the same stem, including case-only differences on
    # case-insensitive filesystems, without conflating their meshes/materials.
    destinations, names = [], set()
    for source in sources:
        stem, suffix = source.stem, 1
        name = stem
        while name.casefold() in names:
            suffix += 1
            name = f"{stem}_{suffix}"
        names.add(name.casefold())
        destinations.append(directory / f"{name}.gltf")

    documents = [_Document(source) for source in sources]
    protected = set(sources)
    for source, document in zip(sources, documents):
        for resource in [*document.gltf.buffers, *document.gltf.images]:
            if resource.uri and not resource.uri.startswith("data:"):
                protected.add((source.parent / unquote(resource.uri)).resolve())
    writes = [atlas_path, report_path]
    for destination in destinations:
        writes.extend((destination.resolve(), destination.with_suffix(".bin").resolve()))
    if len(set(writes)) != len(writes):
        raise ValueError("Model, buffer, texture and report output paths must be distinct")
    if protected & set(writes):
        raise ValueError("An output would overwrite an input model, buffer or painted image")

    # Prepare every document before writing. Patch ownership uses one global
    # index space, which allows reuse across both models and source images.
    groups, patches, patch_models, image_sets = [], [], [], []
    image_records, source_areas = [], {}
    model_records = [{"input": str(source), "output": str(destination),
                      "patches": 0, "triangles": 0, "duplicated_vertices": 0}
                     for source, destination in zip(sources, destinations)]
    for midx, document in enumerate(documents):
        jobs_by_image = _collect_jobs(document)
        image_sets.append(set(jobs_by_image))
        for image_id, jobs in jobs_by_image.items():
            raw = document.image(image_id)
            original = Image.open(io.BytesIO(raw)).convert("RGBA")
            pixels = np.asarray(original)
            source_areas[hashlib.sha256(raw).digest()] = original.width * original.height
            image_records.append({"model": midx, "image": image_id, "size": list(original.size)})
            job_patches = []
            for _, _, positions, uv, triangles, wrap_s, wrap_t in jobs:
                current = _patches(positions, uv, triangles, pixels, pad, wrap_s, wrap_t)
                patches.extend(current)
                patch_models.extend([midx] * len(current))
                job_patches.append(current)
                model_records[midx]["patches"] += len(current)
                model_records[midx]["triangles"] += len(triangles)
            groups.append((midx, jobs, job_patches, original.size))
    if not patches:
        raise ValueError("Input models do not contain any base-colour textured triangles")

    width, height, rects, packed, solid_before = _layout_patches(
        patches, pad=pad, pot=pot, deduplicate=deduplicate, merge_similar=merge_similar,
        similarity_threshold=similarity_threshold, similarity_max_error=similarity_max_error,
        compress_solid=compress_solid, packing=packing)
    for midx, jobs, job_patches, source_size in groups:
        model_records[midx]["duplicated_vertices"] += _rewrite_jobs(
            documents[midx], jobs, job_patches, patches,
            source_size=source_size, atlas_size=(width, height))

    report = {
        "models": model_records, "model_count": len(documents),
        "texture": str(atlas_path), "packed_size": [width, height], "pot": pot,
        "source_images": image_records, "source_image_count": len(source_areas),
        "source_area_pixels": sum(source_areas.values()),
        "area_reduction_percent": round((1 - width * height / sum(source_areas.values())) * 100, 2),
        "patches": len(patches), "unique_patches": len(rects), "padding": pad,
        "deduplicate": deduplicate, "packing": packing, "resampled": False,
        "cross_model_merged_patches": sum(p.owner != i and patch_models[p.owner] != patch_models[i]
                                           for i, p in enumerate(patches)),
        "rect_fill_percent": round(sum(w * h for _, w, h in rects) / (width * height) * 100, 2),
    }
    report.update(_optimization_report(
        patches, compress_solid=compress_solid, solid_texels_before=solid_before,
        merge_similar=merge_similar, similarity_threshold=similarity_threshold,
        similarity_max_error=similarity_max_error))
    for match in report.get("similarity_matches", []):
        match["model"] = patch_models[match["patch"]]
        match["owner_model"] = patch_models[match["owner"]]

    texture_bytes = _image_to_bytes(Image.fromarray(packed))
    for document, destination, image_ids in zip(documents, destinations, image_sets):
        _write_document(document, destination, image_ids, texture_bytes,
                        atlas_path, write_texture=False)
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_path.write_bytes(texture_bytes)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, metavar="MODEL", help="Input .gltf/.glb files, in stable order")
    parser.add_argument("--output-dir", required=True, help="New independent glTF models and shared PNG")
    parser.add_argument("--tex-out", help="Shared PNG path; defaults to OUTPUT_DIR/shared_atlas.png")
    parser.add_argument("--tex-pad", type=int, default=2)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--tex-pot", dest="tex_pot", action="store_true")
    size_group.add_argument("--no-tex-pot", dest="tex_pot", action="store_false")
    parser.set_defaults(tex_pot=False)
    parser.add_argument("--deduplicate", action="store_true", help="Reuse exactly identical patches")
    parser.add_argument("--merge-similar", action="store_true", help="Reuse similar patches, including rotations/mirrors; may change colours")
    parser.add_argument("--similarity-threshold", type=float, default=0.98)
    parser.add_argument("--similarity-max-error", type=int, default=16)
    parser.add_argument("--compress-solid-patches", action="store_true")
    parser.add_argument("--pack", choices=("shelf", "dense"), default="shelf")
    parser.add_argument("--report-out", help="JSON report; defaults to OUTPUT_DIR/merge_report.json")
    args = parser.parse_args(argv)
    try:
        report = merge_textures(
            args.input, args.output_dir, texture_out=args.tex_out, report_out=args.report_out,
            pad=args.tex_pad, pot=args.tex_pot, deduplicate=args.deduplicate,
            merge_similar=args.merge_similar, similarity_threshold=args.similarity_threshold,
            similarity_max_error=args.similarity_max_error,
            compress_solid=args.compress_solid_patches, packing=args.pack)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Merged {report['model_count']} models -> {report['texture']} "
          f"({report['packed_size'][0]} x {report['packed_size'][1]}); "
          f"{report['patches']} patches -> {report['unique_patches']}, "
          f"{report['cross_model_merged_patches']} shared across models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
