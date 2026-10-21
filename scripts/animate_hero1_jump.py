"""Add Jump: blender -b hero1.blend --python scripts/animate_hero1_jump.py."""

import math

import bpy
from mathutils import Quaternion, Vector


scene = bpy.context.scene
rig = bpy.data.objects.get("Hero_WalkRig")
hero = bpy.data.objects.get("hero")
if rig is None or hero is None:
    raise RuntimeError("Expected the existing hero rig")
if bpy.data.actions.get("Hero_Jump") is not None:
    raise RuntimeError("Hero_Jump already exists")
for name in ("Hero_Walk_InPlace", "Hero_Idle"):
    action = bpy.data.actions.get(name)
    if action is None:
        raise RuntimeError(f"Missing existing action: {name}")
    action.use_fake_user = True
rig.animation_data.action = None


def reset_pose():
    for pose in rig.pose.bones:
        pose.rotation_mode = "QUATERNION"
        pose.rotation_quaternion = (1, 0, 0, 0)
        pose.location = (0, 0, 0)
        pose.scale = (1, 1, 1)


def rotate(name, *rotations):
    pose = rig.pose.bones[name]
    world = Quaternion((1, 0, 0, 0))
    for axis, degrees in rotations:
        world = world @ Quaternion(axis, math.radians(degrees))
    rest = pose.bone.matrix_local.to_quaternion()
    pose.rotation_quaternion = rest.inverted() @ world @ rest


legs = [obj for obj in hero.children if obj.type == "MESH" and "leg" in obj.name]
reset_pose()
bpy.context.view_layer.update()
graph = bpy.context.evaluated_depsgraph_get()
floor = min((obj.matrix_world @ vertex.co).z for obj in legs
            for vertex in obj.evaluated_get(graph).data.vertices)

# Frame, leg compression, torso pitch, arm swing, air leg spread.
# Short legs compress during preparation and impact, matching this rigid rig.
poses = [
    (1, 1.0, 0, 0, 0),
    (4, 0.96, 2, 8, 0),
    (8, 0.72, 12, 28, 0),
    (10, 0.80, 8, 10, 0),
    (12, 1.0, -4, -100, 0),
    (18, 0.94, -3, -145, 8),
    (24, 0.92, 0, -155, 12),
    (30, 0.98, 4, -105, 7),
    (36, 1.0, 7, -35, 0),
    (40, 0.75, 11, 20, 0),
    (46, 1.0, 0, 0, 0),
    (49, 1.0, 0, 0, 0),
]


def parameters(frame):
    for first, second in zip(poses, poses[1:]):
        if first[0] <= frame <= second[0]:
            t = (frame - first[0]) / (second[0] - first[0])
            t = t * t * (3 - 2 * t)
            return tuple(a + (b - a) * t for a, b in zip(first[1:], second[1:]))
    raise ValueError(frame)


for frame in range(1, 50):
    scene.frame_set(frame)
    reset_pose()
    compress, pitch, arms, spread = parameters(frame)
    rotate("body", ((0, 1, 0), pitch))
    rotate("head", ((0, 1, 0), -pitch * 0.6))
    rotate("arm.L", ((0, 1, 0), arms), ((1, 0, 0), -4))
    rotate("arm.R", ((0, 1, 0), arms), ((1, 0, 0), 4))
    rotate("leg.L", ((0, 1, 0), spread))
    rotate("leg.R", ((0, 1, 0), -spread))
    for name in ("leg.L", "leg.R"):
        rig.pose.bones[name].scale.y = compress

    for pose in rig.pose.bones:
        for channel in ("rotation_quaternion", "location", "scale"):
            pose.keyframe_insert(channel, frame=frame, group=pose.name)
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    foot_z = min((obj.matrix_world @ vertex.co).z for obj in legs
                 for vertex in obj.evaluated_get(graph).data.vertices)
    t = (frame - 12) / 24
    height = 0.55 * 4 * t * (1 - t) if 12 <= frame <= 36 else 0
    hips = rig.pose.bones["hips"]
    # Ground the feet throughout preparation/landing; use a parabola in flight.
    local_offset = rig.matrix_world.to_3x3().inverted() @ Vector((0, 0, floor + height - foot_z))
    hips.location = hips.bone.matrix_local.to_quaternion().inverted() @ local_offset
    hips.keyframe_insert("location", frame=frame, group="hips")

jump = rig.animation_data.action
jump.name = "Hero_Jump"
jump.use_fake_user = True
jump["loop_frames"] = 48
jump["description"] = "Preparation, takeoff (12), apex (24), landing (36), impact and recovery."
for layer in jump.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for curve in bag.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "BEZIER"
                    key.handle_left_type = key.handle_right_type = "AUTO_CLAMPED"
                # Hold the recovered pose when used as a single jump clip.
                curve.extrapolation = "CONSTANT"

scene.render.fps, scene.render.fps_base = 24, 1
scene.frame_start, scene.frame_end = 1, 48
scene.frame_set(1)
scene.timeline_markers.clear()
for name, frame in (("Prepare", 8), ("Takeoff", 12), ("Apex", 24),
                    ("Landing", 36), ("Impact", 40), ("Recover", 46)):
    scene.timeline_markers.new(name, frame=frame)
rig["animation_notes"] = "Walk: 1-24. Idle: 1-72. Jump: 1-48. All 24 fps; select in Action Editor."
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("JUMP_SAVED:", bpy.data.filepath, "Hero_Jump (48 frames); Walk and Idle retained")
