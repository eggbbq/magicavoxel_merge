"""Run with Blender: blender -b hero1.blend --python scripts/animate_hero1.py.

Add -- --preview to render eight preview PNGs in hero1_walk_preview/.
The original file should be backed up before running this script.
"""

from pathlib import Path
import argparse
import math
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector


parser = argparse.ArgumentParser()
parser.add_argument("--preview", action="store_true")
parser.add_argument("--output", type=Path)
options = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
save_path = str(options.output.resolve()) if options.output else bpy.data.filepath
scene = bpy.context.scene
hero = bpy.data.objects.get("hero")
if hero is None or bpy.data.objects.get("Hero_WalkRig") is not None:
    raise RuntimeError("Expected an unanimated hero with no Hero_WalkRig")
parts = [obj for obj in hero.children if obj.type == "MESH"]
if len(parts) != 6 or any(obj.animation_data for obj in parts):
    raise RuntimeError("Expected six unanimated character parts")


def bounds(obj):
    points = [obj.matrix_local @ Vector(corner) for corner in obj.bound_box]
    return Vector(tuple(min(p[i] for p in points) for i in range(3))), Vector(
        tuple(max(p[i] for p in points) for i in range(3))
    )


head = next(obj for obj in parts if obj.name == "head")
body = next(obj for obj in parts if obj.name == "body")
legs = sorted((obj for obj in parts if "leg" in obj.name), key=lambda obj: bounds(obj)[0].y)
arms = sorted((obj for obj in parts if "hand" in obj.name), key=lambda obj: bounds(obj)[0].y)
rest_floor = min((hero.matrix_world @ Vector(corner)).z for obj in legs
                 for corner in [obj.matrix_local @ Vector(v) for v in obj.bound_box])

rig_data = bpy.data.armatures.new("Hero_WalkSkeleton")
rig = bpy.data.objects.new("Hero_WalkRig", rig_data)
hero.users_collection[0].objects.link(rig)
rig.parent = hero
rig.matrix_parent_inverse = Matrix.Identity(4)
rig.matrix_basis = Matrix.Identity(4)
rig.show_in_front = True
rig_data.display_type = "STICK"
bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")

body_min, body_max = bounds(body)
hip = Vector(((body_min.x + body_max.x) / 2, (body_min.y + body_max.y) / 2, body_min.z))


def bone(name, start, end, parent=None, deform=True):
    item = rig_data.edit_bones.new(name)
    item.head, item.tail = start, end
    item.use_deform = deform
    if parent:
        item.parent = rig_data.edit_bones[parent]
    return item


bone("hips", hip, hip + Vector((0, 0, 0.08)), deform=False)
bone("body", hip, Vector((hip.x, hip.y, body_max.z)), "hips")
part_bones = {body: "body", head: "head"}
head_min, head_max = bounds(head)
bone("head", Vector(((head_min.x + head_max.x) / 2, hip.y, head_min.z)),
     Vector(((head_min.x + head_max.x) / 2, hip.y, head_max.z)), "body")
for kind, objects, parent in (("leg", legs, "hips"), ("arm", arms, "body")):
    for side, obj in zip(("L", "R"), objects):
        low, high = bounds(obj)
        center = (low + high) / 2
        name = f"{kind}.{side}"
        bone(name, Vector((center.x, center.y, high.z)),
             Vector((center.x, center.y, low.z)), parent)
        part_bones[obj] = name
bpy.ops.object.mode_set(mode="OBJECT")

# Each voxel part remains rigid, with all its vertices weighted to one bone.
for obj, name in part_bones.items():
    # Imported instances share mesh data, including deform weights. Give each
    # animated limb its own weights so opposite swings cannot cancel each other.
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    group = obj.vertex_groups.new(name=name)
    group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    modifier = obj.modifiers.new("Walk rig", "ARMATURE")
    modifier.object = rig
    modifier.use_deform_preserve_volume = True


def rotate_world_axis(name, axis, angle):
    pose = rig.pose.bones[name]
    rest = pose.bone.matrix_local.to_quaternion()
    pose.rotation_mode = "QUATERNION"
    pose.rotation_quaternion = rest.inverted() @ Quaternion(axis, angle) @ rest


def translate_world_axis(name, value):
    pose = rig.pose.bones[name]
    pose.location = pose.bone.matrix_local.to_quaternion().inverted() @ Vector(value)


scene.render.fps = 24
scene.render.fps_base = 1
scene.frame_start, scene.frame_end = 1, 24
scene.timeline_markers.new("Left contact", frame=1)
scene.timeline_markers.new("Passing", frame=7)
scene.timeline_markers.new("Right contact", frame=13)
scene.timeline_markers.new("Passing", frame=19)
scene.timeline_markers.new("Loop seam", frame=25)

for frame in range(1, 26, 2):
    scene.frame_set(frame)
    phase = 2 * math.pi * (frame - 1) / 24
    swing = math.cos(phase)
    rotate_world_axis("leg.L", (0, 1, 0), -math.radians(26) * swing)
    rotate_world_axis("leg.R", (0, 1, 0), math.radians(26) * swing)
    rotate_world_axis("arm.L", (0, 1, 0), math.radians(22) * swing)
    rotate_world_axis("arm.R", (0, 1, 0), -math.radians(22) * swing)
    rotate_world_axis("body", (0, 0, 1), math.radians(3) * swing)
    rotate_world_axis("head", (0, 0, 1), -math.radians(2) * swing)
    translate_world_axis("leg.L", (0, 0, 0.035 * max(0, -math.sin(phase))))
    translate_world_axis("leg.R", (0, 0, 0.035 * max(0, math.sin(phase))))
    translate_world_axis("hips", (0, 0.009 * swing, 0))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    foot_z = min((evaluated.matrix_world @ vertex.co).z for obj in legs
                 for evaluated in [obj.evaluated_get(depsgraph)] for vertex in evaluated.data.vertices)
    # Ground the lower foot; the other leg lifts during its forward swing.
    translate_world_axis("hips", (0, 0.009 * swing, rest_floor - foot_z))
    for name in ("hips", "leg.L", "leg.R", "arm.L", "arm.R", "body", "head"):
        pose = rig.pose.bones[name]
        if name != "hips":
            pose.keyframe_insert("rotation_quaternion", frame=frame, group=name)
        if name in ("hips", "leg.L", "leg.R"):
            pose.keyframe_insert("location", frame=frame, group=name)

action = rig.animation_data.action
action.name = "Hero_Walk_InPlace"
action.use_fake_user = True
for layer in action.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for curve in bag.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "BEZIER"
                    key.handle_left_type = key.handle_right_type = "AUTO_CLAMPED"
                curve.modifiers.new("CYCLES")
rig["animation_notes"] = "Walk in place; 24 fps; play frames 1-24. Frame 25 duplicates frame 1."
rig["forward_axis"] = "+X; limbs swing around Y"
scene.frame_set(1)

# Make the character convenient to inspect when the file is opened.
center = hero.matrix_world @ Vector((hip.x, hip.y, (head_max.z + min(bounds(o)[0].z for o in legs)) / 2))
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.shading.type = "MATERIAL"
            space.region_3d.view_location = center
            space.region_3d.view_distance = 2.7
            space.region_3d.view_rotation = Vector((3, -4, 2)).to_track_quat("Z", "Y")
bpy.ops.wm.save_as_mainfile(filepath=save_path)
print("WALK_SAVED:", bpy.data.filepath)

if options.preview:
    # Preview staging is temporary and is not saved into the user's scene.
    output = Path(bpy.data.filepath).parent / "hero1_walk_preview"
    output.mkdir(exist_ok=True)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 384
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("Walk Preview World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.12, 0.15, 0.20, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.65
    camera_data = bpy.data.cameras.new("Walk Preview Camera")
    camera = bpy.data.objects.new("Walk Preview Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((3.4, -4.4, 2.1))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type, camera_data.ortho_scale = "ORTHO", 1.85
    scene.camera = camera
    for name, offset, energy, size in (
        ("Preview Key", (3, -4, 5), 350, 4),
        ("Preview Fill", (-2, -1, 3), 160, 3),
        ("Preview Rim", (1, 3, 4), 250, 3),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy, light_data.shape, light_data.size = energy, "DISK", size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = center + Vector(offset)
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.mesh.primitive_plane_add(size=200, location=(center.x, center.y, rest_floor - 0.002))
    ground = bpy.context.object
    material = bpy.data.materials.new("Preview Ground")
    material.diffuse_color = (0.22, 0.27, 0.34, 1)
    ground.data.materials.append(material)
    for frame in range(1, 25, 3):
        scene.frame_set(frame)
        scene.render.filepath = str(output / f"walk_{frame:02d}.png")
        bpy.ops.render.render(write_still=True)
    print("PREVIEW_SAVED:", output)
