"""Command line interface for the BTP voxel pipeline."""

from __future__ import annotations

import argparse

from .pipeline import AtlasOptions, PipelineOptions, convert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btp-vox")
    parser.add_argument("--input", required=True, help="Path to MagicaVoxel .vox file")
    parser.add_argument("--output", required=True, help="Path to output .glb file")
    
    parser.add_argument("--debug-transforms-out", help="Write per-model transform diagnostics JSON")
    parser.add_argument("--print-nodes", action="store_true", help="Print VOX scene graph nodes to stderr")

    parser.add_argument("--format", choices=("glb", "gltf"), default="glb", help="Output format")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied to geometry")
    parser.add_argument("--pivot", choices=("corner", "bottom_center", "center"), default="corner", help="Model pivot")
    
    parser.add_argument("--uv-out", dest="uv_json_out", help="Write per-model UV rectangles to this JSON file")
    parser.add_argument("--uv-flip-v", dest="flip_v", action="store_true", help="Flip V texture coordinate")

    parser.add_argument("--tex-fmt", dest="texture_alpha", choices=("auto", "rgba", "rgb"), default="auto", help="Atlas texture alpha mode: auto keeps alpha only if needed; rgba forces alpha; rgb strips alpha")
    parser.add_argument("--tex-out", dest="texture_out", help="Write atlas PNG to this path (otherwise embed into GLB)")
    parser.add_argument("--tex-pad", dest="atlas_pad", type=int, default=2, help="Atlas padding in texels")
    parser.add_argument("--tex-inset", dest="atlas_inset", type=float, default=1.0, help="Atlas inset in texels")
    parser.add_argument("--tex-texel-scale", dest="atlas_texel_scale", type=int, default=1, help="Texel scale applied to each quad")
    parser.add_argument("--tex-style", dest="atlas_style", choices=("baked", "solid"), default="baked", help="Atlas style")
    parser.add_argument("--tex-layout", dest="atlas_layout", choices=("by-model", "global"), default="by-model", help="Atlas layout")
    parser.add_argument("--tex-square", dest="atlas_square", action="store_true", help="Force atlas to be square")
    parser.add_argument("--tex-pot", dest="atlas_pot", action="store_true", help="Force atlas dimensions to power-of-two")
    parser.add_argument("--tex-tight-blocks", dest="atlas_tight_blocks", action="store_true", help="Tight-pack per-model blocks (by-model layout)")

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
        flip_v=args.flip_v,
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
