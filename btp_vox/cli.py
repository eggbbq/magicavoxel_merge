"""Command line interface for the BTP voxel pipeline."""

from __future__ import annotations

import argparse

from .pipeline import AtlasOptions, PipelineOptions, convert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btp-vox")
    parser.add_argument("input", help="Path to MagicaVoxel .vox file")
    parser.add_argument("output", help="Path to output .glb file")

    parser.add_argument("--texture-out", help="Write atlas PNG to this path (otherwise embed into GLB)")
    parser.add_argument("--uv-json-out", help="Write per-model UV rectangles to this JSON file")
    parser.add_argument("--debug-transforms-out", help="Write per-model transform diagnostics JSON")
    parser.add_argument("--print-nodes", action="store_true", help="Print VOX scene graph nodes to stderr")
    parser.add_argument("--format", choices=("glb", "gltf"), default="glb", help="Output format")

    parser.add_argument(
        "--texture-alpha",
        choices=("auto", "rgba", "rgb"),
        default="auto",
        help="Atlas texture alpha mode: auto keeps alpha only if needed; rgba forces alpha; rgb strips alpha",
    )

    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied to geometry")
    parser.add_argument("--center", action="store_true", help="Center each model at origin")
    parser.add_argument("--center-bounds", action="store_true", help="Center using bounding box midpoint")
    parser.add_argument("--pivot", choices=("corner", "bottom_center", "center"), default="corner", help="Model pivot")
    parser.add_argument("--weld", action="store_true", help="Enable vertex welding")
    parser.add_argument("--flip-v", action="store_true", help="Flip V texture coordinate")
    parser.add_argument("--bake-translation", action="store_true", help="Bake translation into vertex positions")

    parser.add_argument("--atlas-pad", type=int, default=2, help="Atlas padding in texels")
    parser.add_argument("--atlas-inset", type=float, default=1.0, help="Atlas inset in texels")
    parser.add_argument("--atlas-texel-scale", type=int, default=1, help="Texel scale applied to each quad")
    parser.add_argument("--atlas-style", choices=("baked", "solid"), default="baked", help="Atlas style")
    parser.add_argument("--atlas-layout", choices=("by-model", "global"), default="by-model", help="Atlas layout")
    parser.add_argument("--atlas-square", action="store_true", help="Force atlas to be square")
    parser.add_argument("--atlas-pot", action="store_true", help="Force atlas dimensions to power-of-two")
    parser.add_argument("--atlas-tight-blocks", action="store_true", help="Tight-pack per-model blocks (by-model layout)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    atlas_opts = AtlasOptions(
        pad=args.atlas_pad,
        inset=args.atlas_inset,
        texel_scale=args.atlas_texel_scale,
        layout=args.atlas_layout,
        square=args.atlas_square,
        pot=args.atlas_pot,
        tight_blocks=args.atlas_tight_blocks,
        style=args.atlas_style,
    )
    pipeline_opts = PipelineOptions(
        scale=args.scale,
        pivot=args.pivot,
        center=args.center,
        center_bounds=args.center_bounds,
        weld=args.weld,
        flip_v=args.flip_v,
        bake_translation=args.bake_translation,
        texture_alpha=str(args.texture_alpha),
        atlas=atlas_opts,
    )

    convert(
        args.input,
        args.output,
        texture_out=args.texture_out,
        uv_json_out=args.uv_json_out,
        debug_transforms_out=args.debug_transforms_out,
        print_nodes=bool(args.print_nodes),
        output_format=str(args.format),
        options=pipeline_opts,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
