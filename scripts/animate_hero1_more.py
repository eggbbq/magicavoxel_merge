"""Add Sleep, Run, and ChopTree: blender -b hero1.blend --python scripts/animate_hero1_more.py."""

from pathlib import Path
import math
import shutil

import bpy
from mathutils import Matrix, Quaternion, Vector


scene = bpy.context.scene
hero = bpy.data.objects["hero"]
rig = bpy.data.objects["Hero_WalkRig"]
new_names = ("Hero_Sleep", "Hero_Run", "Hero_ChopTree")
if any(bpy.data.actions.get(name) for name in new_names):
    raise RuntimeError("One of the requested actions already exists")
backup = Path(bpy.data.filepath).with_name("hero1_before_more_actions.blend")
if not backup.exists():
    shutil.copy2(bpy.data.filepath, backup)
parts = [obj for obj in hero.children if obj.type == "MESH"]
legs = [obj for obj in parts if "leg" in obj.name]


def action_curves(action):
    return [curve for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags for curve in bag.fcurves]


original_signatures = {
    action.name: [(curve.data_path, curve.array_index, [tuple(key.co) for key in curve.keyframe_points])
                  for curve in action_curves(action)]
    for action in bpy.data.actions
}
rig["chop_props"] = 0.0
rig["sleep_lids"] = 0.0
rig["tree_hit"] = 0.0
for name in original_signatures:
    action = bpy.data.actions[name]
    action.use_fake_user = True
    rig.animation_data.action = action
    for prop in ("chop_props", "sleep_lids", "tree_hit"):
        rig.keyframe_insert(f'["{prop}"]', frame=1, group="Props")
rig.animation_data.action = None


def reset():
    for pose in rig.pose.bones:
        pose.rotation_mode = "QUATERNION"
        pose.rotation_quaternion = (1, 0, 0, 0)
        pose.location = (0, 0, 0)
        pose.scale = (1, 1, 1)


def rotate(name, *rotations):
    world = Quaternion((1, 0, 0, 0))
    for axis, degrees in rotations:
        world = world @ Quaternion(axis, math.radians(degrees))
    pose = rig.pose.bones[name]
    rest = pose.bone.matrix_local.to_quaternion()
    pose.rotation_quaternion = rest.inverted() @ world @ rest


def shift(name, vector):
    pose = rig.pose.bones[name]
    pose.location = pose.bone.matrix_local.to_quaternion().inverted() @ Vector(vector)


def ground_parts(objects, height=0):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    low = min((obj.matrix_world @ vertex.co).z for obj in objects
              for vertex in obj.evaluated_get(graph).data.vertices)
    offset = rig.matrix_world.to_3x3().inverted() @ Vector((0, 0, floor + height - low))
    pose = rig.pose.bones["hips"]
    pose.location += pose.bone.matrix_local.to_quaternion().inverted() @ offset


reset()
bpy.context.view_layer.update()
graph = bpy.context.evaluated_depsgraph_get()
floor = min((obj.matrix_world @ vertex.co).z for obj in legs
            for vertex in obj.evaluated_get(graph).data.vertices)


def key_pose(frame):
    for pose in rig.pose.bones:
        for channel in ("rotation_quaternion", "location", "scale"):
            pose.keyframe_insert(channel, frame=frame, group=pose.name)
    for prop in ("chop_props", "sleep_lids", "tree_hit"):
        rig.keyframe_insert(f'["{prop}"]', frame=frame, group="Props")


def finish(name, length, description):
    action = rig.animation_data.action
    action.name, action.use_fake_user = name, True
    action["loop_frames"] = length
    action["description"] = description
    for curve in action_curves(action):
        points = list(curve.keyframe_points)
        if max(key.co.y for key in points) - min(key.co.y for key in points) < 1e-6:
            value = points[0].co.y
            curve.keyframe_points.clear()
            curve.keyframe_points.insert(1, value)
        for key in curve.keyframe_points:
            key.interpolation = "BEZIER"
            key.handle_left_type = key.handle_right_type = "AUTO_CLAMPED"
            key.select_control_point = False
        curve.modifiers.new("CYCLES")
        curve.update()
    print("ACTION_CREATED:", name, "frames", length,
          "keys", sum(len(curve.keyframe_points) for curve in action_curves(action)))
    return action


# Sleeping face-up: the hips rotate the whole skeleton, with a gentle breath.
for frame in (1, 25, 49, 73, 97):
    scene.frame_set(frame)
    reset()
    phase = 2 * math.pi * (frame - 1) / 96
    breath = 0.5 - 0.5 * math.cos(phase)
    rotate("hips", ((0, 1, 0), -90))
    rotate("body", ((0, 1, 0), 0.7 * breath))
    rig.pose.bones["body"].scale.y = 1 + 0.008 * breath
    rotate("head", ((0, 0, 1), 2), ((0, 1, 0), -0.8 * breath))
    rotate("arm.L", ((1, 0, 0), -12 - breath))
    rotate("arm.R", ((1, 0, 0), 12 + breath))
    rig["sleep_lids"], rig["chop_props"], rig["tree_hit"] = 1, 0, 0
    ground_parts(parts)
    key_pose(frame)
sleep = finish("Hero_Sleep", 96, "Face-up sleeping loop, closed eyes and gentle breathing.")

# A quick in-place run: wide opposite swings, lifted swing foot and two flight phases.
rig.animation_data.action = None
for frame in range(1, 18, 2):
    scene.frame_set(frame)
    reset()
    phase = 2 * math.pi * (frame - 1) / 16
    swing = math.cos(phase)
    rotate("body", ((0, 1, 0), 9), ((0, 0, 1), 1.2 * swing))
    rotate("head", ((0, 1, 0), -5), ((0, 0, 1), -0.6 * swing))
    rotate("leg.L", ((0, 1, 0), -30 * swing))
    rotate("leg.R", ((0, 1, 0), 30 * swing))
    rotate("arm.L", ((0, 1, 0), 32 * swing), ((1, 0, 0), -8))
    rotate("arm.R", ((0, 1, 0), -32 * swing), ((1, 0, 0), 8))
    shift("leg.L", (0, 0, 0.045 * max(0, -math.sin(phase))))
    shift("leg.R", (0, 0, 0.045 * max(0, math.sin(phase))))
    shift("hips", (0, 0.0024 * swing, 0))
    rig["sleep_lids"], rig["chop_props"], rig["tree_hit"] = 0, 0, 0
    ground_parts(legs, 0.012 * math.sin(phase) ** 2)
    key_pose(frame)
run = finish("Hero_Run", 16, "Fast in-place run; 16-frame stride with two short flight phases.")

# A one-handed chop: windup, quick strike, recoil, and return to the ready pose.
rig.animation_data.action = None
chop_poses = (
    (1, -65, 2, 0, 1.0, 0),
    (7, -120, -8, -5, 1.0, 0),
    (11, -155, -13, -8, 1.0, 0),
    (15, -135, -8, -4, 1.0, 0),
    (19, -55, 12, 4, 0.94, 1),
    (21, -60, 8, 2, 0.96, 0.25),
    (27, -80, 0, 0, 1.0, 0),
    (37, -65, 2, 0, 1.0, 0),
)
for frame, arm, pitch, twist, compress, hit in chop_poses:
    scene.frame_set(frame)
    reset()
    rotate("body", ((0, 1, 0), pitch), ((0, 0, 1), twist))
    rotate("head", ((0, 1, 0), -pitch * 0.45), ((0, 0, 1), -twist * 0.5))
    rotate("arm.R", ((0, 1, 0), arm), ((1, 0, 0), 3))
    rotate("arm.L", ((0, 1, 0), -25 - 0.15 * abs(arm)), ((1, 0, 0), -12))
    for name in ("leg.L", "leg.R"):
        rig.pose.bones[name].scale.y = compress
    rig["sleep_lids"], rig["chop_props"], rig["tree_hit"] = 0, 1, hit
    ground_parts(legs)
    key_pose(frame)
chop = finish("Hero_ChopTree", 36, "Ready, windup, axe strike at frame 19, recoil, and recovery.")


def material(name, srgb, metallic=0):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    shader = value.node_tree.nodes.get("Principled BSDF")
    linear = tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)
    shader.inputs["Base Color"].default_value = (*linear, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = 0.65
    value.diffuse_color = (*linear, 1)
    return value


wood = material("Axe Handle", (0.39, 0.22, 0.10))
steel = material("Axe Steel", (0.54, 0.61, 0.69), 0.65)
edge = material("Axe Edge", (0.79, 0.85, 0.89), 0.75)
skin = material("Sleep Eyelids", (0.831, 0.682, 0.667))
lash = material("Closed Eyes", (0.133, 0.137, 0.137))
collection = bpy.data.collections.new("Action Props")
scene.collection.children.link(collection)


def visibility(obj, prop):
    for path in ("hide_viewport", "hide_render"):
        driver = obj.driver_add(path).driver
        driver.expression = "enabled < 0.5"
        variable = driver.variables.new()
        variable.name, variable.type = "enabled", "SINGLE_PROP"
        variable.targets[0].id = rig
        variable.targets[0].data_path = f'["{prop}"]'


def box(name, center, size, mat, bone=None, prop=None, transform=None):
    center, size = Vector(center), Vector(size)
    vertices = [center + Vector((x * size.x / 2, y * size.y / 2, z * size.z / 2))
                for x, y, z in ((-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),
                               (1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1))]
    if transform is not None:
        vertices = [transform @ vertex for vertex in vertices]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], ((0,4,6,2),(1,3,7,5),(0,1,5,4),
                                     (2,6,7,3),(0,2,3,1),(4,5,7,6)))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.parent = hero
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.data.materials.append(mat)
    if bone:
        group = obj.vertex_groups.new(name=bone)
        group.add(list(range(8)), 1, "REPLACE")
        modifier = obj.modifiers.new("Follow character bone", "ARMATURE")
        modifier.object = rig
    if prop:
        visibility(obj, prop)
    return obj


grip = rig.data.bones["arm.R"].tail_local
box("Axe_Handle", grip + Vector((0, 0, -0.10)), (0.035, 0.035, 0.40), wood, "arm.R", "chop_props")
blade = box("Axe_Head", grip + Vector((0.07, 0, -0.26)),
            (0.21, 0.045, 0.14), steel, "arm.R", "chop_props")
box("Axe_BladeEdge", grip + Vector((0.185, 0, -0.26)),
    (0.025, 0.05, 0.17), edge, "arm.R", "chop_props")

# Cover the two eye patches with skin and a thin closed-eye line in Sleep only.
head = bpy.data.objects["head"]
for index, y in enumerate((-0.08, 0.08)):
    box(f"Sleep_Eyelid_{index}", (0.1705, y, 0.24), (0.001, 0.12, 0.04),
        skin, "head", "sleep_lids", head.matrix_local)
    box(f"Sleep_EyeLine_{index}", (0.1712, y, 0.24), (0.001, 0.08, 0.006),
        lash, "head", "sleep_lids", head.matrix_local)

# Place the preview trunk at the evaluated blade edge on the strike frame.
scene.frame_set(19)
bpy.context.view_layer.update()
graph = bpy.context.evaluated_depsgraph_get()
edge_obj = bpy.data.objects["Axe_BladeEdge"]
blade_points = [hero.matrix_world.inverted() @ (edge_obj.matrix_world @ v.co)
                for v in edge_obj.evaluated_get(graph).data.vertices]
hit_center = sum(blade_points, Vector()) / len(blade_points)
floor_local = (hero.matrix_world.inverted() @ Vector((hero.location.x, hero.location.y, floor))).z
tree_root = bpy.data.objects.new("Chop_PreviewTree", None)
collection.objects.link(tree_root)
tree_root.parent = hero
tree_root.location = (hit_center.x + 0.12, hit_center.y, floor_local)
tree_root["purpose"] = "Preview prop for Hero_ChopTree; automatically hidden in other actions."
driver = tree_root.driver_add("rotation_euler", 1).driver
driver.expression = "0.035 * hit"
variable = driver.variables.new()
variable.name, variable.type = "hit", "SINGLE_PROP"
variable.targets[0].id = rig
variable.targets[0].data_path = '["tree_hit"]'
bark = material("Tree Bark", (0.32, 0.20, 0.12))
leaves = material("Tree Leaves", (0.24, 0.46, 0.24))
leaves_top = material("Tree Leaves Top", (0.34, 0.56, 0.28))
for name, center, size, mat in (
    ("Tree_Trunk", (0, 0, 0.58), (0.25, 0.25, 1.16), bark),
    ("Tree_Crown", (0, 0, 1.31), (0.76, 0.68, 0.54), leaves),
    ("Tree_CrownTop", (-0.05, 0.02, 1.66), (0.53, 0.52, 0.30), leaves_top),
):
    obj = box(name, center, size, mat, prop="chop_props")
    obj.parent = tree_root
    obj.matrix_parent_inverse = Matrix.Identity(4)

# Verify existing transform keys survived the added prop-visibility channels.
for name, signature in original_signatures.items():
    current = [(curve.data_path, curve.array_index, [tuple(key.co) for key in curve.keyframe_points])
               for curve in action_curves(bpy.data.actions[name]) if not curve.data_path.startswith('["')]
    assert current == signature, name

assert len(rig.data.bones) == 7
scene.render.fps, scene.render.fps_base = 24, 1
scene.frame_start, scene.frame_end = 1, 36
scene.frame_set(1)
scene.timeline_markers.clear()
for name, frame in (("Ready", 1), ("Windup", 11), ("Strike", 19), ("Recovery", 27)):
    scene.timeline_markers.new(name, frame=frame)
rig["animation_notes"] = "24 fps. Walk 1-24; Idle 1-72; Jump 1-48; Sleep 1-96; Run 1-16; ChopTree 1-36."
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.shading.type = "MATERIAL"
            space.overlay.show_overlays = False
            space.region_3d.view_location = hero.matrix_world @ Vector((0.35, 0, 0.10))
            space.region_3d.view_distance = 4.3
            space.region_3d.view_rotation = Vector((3, -4, 2)).to_track_quat("Z", "Y")
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("MORE_ACTIONS_SAVED:", bpy.data.filepath)
