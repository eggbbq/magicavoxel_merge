import argparse

from .pipeline import vox_to_glb


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="magicavoxel-merge")
    parser.add_argument("input", help="Path to .vox file")
    parser.add_argument("output", help="Path to output .glb file")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--center-bounds", action="store_true")
    parser.add_argument("--weld", action="store_true")
    parser.add_argument("--cull-mv-faces", default=None)
    parser.add_argument("--mode", choices=("palette", "atlas"), default="palette")
    parser.add_argument("--axis", choices=("y_up", "identity"), default="y_up")
    parser.add_argument("--mv-zup", dest="axis", action="store_const", const="y_up", help=argparse.SUPPRESS)
    parser.add_argument("--merge-strategy", choices=("greedy", "maxrect"), default="greedy")
    parser.add_argument("--atlas-pad", type=int, default=2)
    parser.add_argument("--atlas-inset", type=float, default=1.5)
    parser.add_argument("--atlas-style", choices=("solid", "baked"), default="solid")
    parser.add_argument("--atlas-texel-scale", type=int, default=1)
    parser.add_argument("--atlas-layout", choices=("global", "by-model"), default="global")
    parser.add_argument("--handedness", choices=("right", "left"), default="right")
    parser.add_argument("--avg-normals-attr", choices=("none", "color", "tangent"), default="none")
    parser.add_argument("--flip-v", action="store_true")
    parser.add_argument("--forward", choices=("posZ", "negZ"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--facing", choices=("+z", "-z"), default=None, help=argparse.SUPPRESS)
    atlas_grp = parser.add_mutually_exclusive_group()
    atlas_grp.add_argument("--atlas-square", dest="atlas_square", action="store_true")
    atlas_grp.add_argument("--no-atlas-square", dest="atlas_square", action="store_false")
    parser.set_defaults(atlas_square=True)

    baked_dedup_grp = parser.add_mutually_exclusive_group()
    baked_dedup_grp.add_argument("--baked-dedup", dest="baked_dedup", action="store_true")
    baked_dedup_grp.add_argument("--no-baked-dedup", dest="baked_dedup", action="store_false")
    parser.set_defaults(baked_dedup=True)

    parser.add_argument("--texture-out", default=None)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--preserve-transforms", dest="preserve_transforms", action="store_true")
    grp.add_argument("--no-preserve-transforms", dest="preserve_transforms", action="store_false")
    parser.set_defaults(preserve_transforms=True)
    args = parser.parse_args(argv)

    if args.center and args.center_bounds:
        raise SystemExit("--center and --center-bounds are mutually exclusive")

    handedness = args.handedness

    vox_to_glb(
        args.input,
        args.output,
        axis=args.axis,
        merge_strategy=args.merge_strategy,
        scale=args.scale,
        center=args.center,
        center_bounds=args.center_bounds,
        weld=args.weld,
        cull_mv_faces=args.cull_mv_faces,
        atlas_pad=args.atlas_pad,
        atlas_inset=args.atlas_inset,
        atlas_style=args.atlas_style,
        baked_dedup=args.baked_dedup,
        atlas_texel_scale=args.atlas_texel_scale,
        atlas_layout=args.atlas_layout,
        atlas_square=args.atlas_square,
        handedness=handedness,
        texture_out=args.texture_out,
        preserve_transforms=args.preserve_transforms,
        avg_normals_attr=args.avg_normals_attr,
        flip_v=args.flip_v,
        mode=args.mode,
    )
    return 0
