"""Add Idle alongside Walk: blender -b hero1.blend --python scripts/animate_hero1_idle.py."""

import math

import bpy
from mathutils import Quaternion


scene = bpy.context.scene
rig = bpy.data.objects.get("Hero_WalkRig")
if rig is None or rig.type != "ARMATURE":
    raise RuntimeError("Hero_WalkRig is required; create the walk rig first")
if bpy.data.actions.get("Hero_Idle") is not None:
    raise RuntimeError("Hero_Idle already exists")
walk = bpy.data.actions.get("Hero_Walk_InPlace")
if walk is None:
    raise RuntimeError("Expected the existing walk animation")
walk.use_fake_user = True
rig.animation_data.action = None


def rotate(name, *rotations):
    pose = rig.pose.bones[name]
    world_rotation = Quaternion((1, 0, 0, 0))
    for axis, degrees in rotations:
        world_rotation = world_rotation @ Quaternion(axis, math.radians(degrees))
    rest = pose.bone.matrix_local.to_quaternion()
    pose.rotation_quaternion = rest.inverted() @ world_rotation @ rest


for frame in range(1, 74, 6):
    scene.frame_set(frame)
    phase = 2 * math.pi * (frame - 1) / 72
    breathe = 0.5 - 0.5 * math.cos(phase)
    sway = math.sin(phase)
    for pose in rig.pose.bones:
        pose.rotation_mode = "QUATERNION"
        pose.rotation_quaternion = (1, 0, 0, 0)
        pose.location = (0, 0, 0)
        pose.scale = (1, 1, 1)

    # Feet and hips remain planted. Only the upper body breathes and sways.
    rotate("body", ((0, 0, 1), 0.7 * sway), ((0, 1, 0), -0.6 * breathe))
    rig.pose.bones["body"].scale = (1 + 0.004 * breathe, 1 + 0.012 * breathe, 1)
    rotate("head", ((0, 0, 1), -1.5 * sway), ((0, 1, 0), 1.2 * breathe))
    rotate("arm.L", ((1, 0, 0), -4 - breathe), ((0, 1, 0), 1.5 * sway))
    rotate("arm.R", ((1, 0, 0), 4 + breathe), ((0, 1, 0), -1.5 * sway))

    # Key every transform so selecting Idle resets the previous Walk pose.
    for pose in rig.pose.bones:
        for channel in ("rotation_quaternion", "location", "scale"):
            pose.keyframe_insert(channel, frame=frame, group=pose.name)

idle = rig.animation_data.action
idle.name = "Hero_Idle"
idle.use_fake_user = True
for layer in idle.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for curve in bag.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "BEZIER"
                    key.handle_left_type = key.handle_right_type = "AUTO_CLAMPED"
                curve.modifiers.new("CYCLES")

scene.render.fps, scene.render.fps_base = 24, 1
scene.frame_start, scene.frame_end = 1, 72
scene.frame_set(1)
scene.timeline_markers.clear()
scene.timeline_markers.new("Idle - breathe in", frame=1)
scene.timeline_markers.new("Idle - breathe out", frame=37)
scene.timeline_markers.new("Idle loop seam", frame=73)
idle["loop_frames"] = 72
walk["loop_frames"] = 24
rig["animation_notes"] = "Hero_Idle: frames 1-72 (3s). Hero_Walk_InPlace: frames 1-24 (1s). Both at 24 fps."

# Open in Animation with an Action Editor, so both clips are easy to select.
for window in bpy.context.window_manager.windows:
    workspace = bpy.data.workspaces.get("Animation")
    if workspace:
        window.workspace = workspace
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.overlay.show_overlays = False
        elif area.type == "DOPESHEET_EDITOR":
            area.spaces.active.mode = "ACTION"
            area.spaces.active.show_region_ui = False
            area.spaces.active.dopesheet.show_only_selected = False

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("IDLE_SAVED:", bpy.data.filepath, "Hero_Idle (72 frames), Walk retained")
