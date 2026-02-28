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
    parser.add_argument("--cull", default="", help="Cull faces by letters in MagicaVoxel model space: t=+Z, b=-Z, l=-X, r=+X, f=+Y, k=-Y")
    parser.add_argument("--plat-top-cutout", action="store_true", help="Plat top cutout mode: one top quad per model + alpha clip")
    parser.add_argument("--plat-cutoff", type=float, default=0.5, help="Alpha cutoff for --plat-top-cutout (glTF alphaMode=MASK)")
    parser.add_argument(
        "--plat-suffix",
        nargs="?",
        const="-cutout",
        default="-cutout",
        help="Models with names ending in this suffix use plat cutout quad + alpha clip (default: -cutout)",
    )
    
    parser.add_argument("--uv-out", help="Write per-model UV rectangles to this JSON file")
    parser.add_argument("--uv-flip-v", action="store_true", help="Flip V texture coordinate")
    parser.add_argument("--uv2", action="store_true", help="Export secondary UV set (TEXCOORD_1) by copying TEXCOORD_0 (for lightmap baking)")
    parser.add_argument("--uv2-mode", choices=("copy", "lightmap"), default="copy", help="How to generate UV2 (TEXCOORD_1): copy duplicates UV0; lightmap generates non-overlapping UVs")
    parser.add_argument("--vertex-color", action="store_true", help="Export vertex colors as COLOR_0 (filled with white)")

    parser.add_argument("--tex-fmt", choices=("auto", "rgba", "rgb"), default="auto", help="Atlas texture alpha mode: auto keeps alpha only if needed; rgba forces alpha; rgb strips alpha")
    parser.add_argument("--tex-out", help="Write atlas PNG to this path (otherwise embed into GLB)")
    parser.add_argument("--tex-pad", type=int, default=2, help="Atlas padding in texels")
    parser.add_argument("--tex-inset", type=float, default=1.0, help="Atlas inset in texels")
    parser.add_argument("--tex-texel-scale", type=int, default=1, help="Texel scale applied to each quad")
    parser.add_argument("--tex-style", choices=("baked", "solid"), default="baked", help="Atlas style")
    parser.add_argument("--tex-layout", choices=("by-model", "global"), default="by-model", help="Atlas layout")
    parser.add_argument("--tex-square", action="store_true", help="Force atlas to be square")
    parser.add_argument("--tex-pot", action="store_true", help="Force atlas dimensions to power-of-two")
    parser.add_argument("--tex-tight-blocks", action="store_true", help="Tight-pack per-model blocks (by-model layout)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    atlas_opts = AtlasOptions(
        pad=args.tex_pad,
        inset=args.tex_inset,
        texel_scale=args.tex_texel_scale,
        layout=args.tex_layout,
        square=args.tex_square,
        pot=args.tex_pot,
        tight_blocks=args.tex_tight_blocks,
        style=args.tex_style,
    )
    pipeline_opts = PipelineOptions(
        scale=args.scale,
        pivot=args.pivot,
        flip_v=args.uv_flip_v,
        export_uv2=bool(args.uv2),
        uv2_mode=str(args.uv2_mode),
        export_vertex_color=bool(args.vertex_color),
        cull=str(args.cull),
        plat_cutout=bool(args.plat_top_cutout),
        plat_cutoff=float(args.plat_cutoff),
        plat_suffix=str(args.plat_suffix),
        texture_alpha=str(args.tex_fmt),
        atlas=atlas_opts,
    )

    convert(
        args.input,
        args.output,
        texture_out=args.tex_out,
        uv_json_out=args.uv_out,
        debug_transforms_out=args.debug_transforms_out,
        print_nodes=bool(args.print_nodes),
        output_format=str(args.format),
        options=pipeline_opts,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
