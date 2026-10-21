"""Soften only the existing Run action; run with Blender -b hero1_actions.blend."""

import ast
import math
from pathlib import Path
import shutil

import bpy
from mathutils import Quaternion, Vector

scene = bpy.context.scene
rig = bpy.data.objects["Hero_WalkRig"]
hero = bpy.data.objects["hero"]
parts = [obj for obj in hero.children if obj.type == "MESH"
         and obj.name in ("head", "body", "left-hand", "left-hand.001",
                          "left-leg", "left-leg.001")]
legs = [obj for obj in parts if "leg" in obj.name]
action = bpy.data.actions["Hero_Run"]
rig.animation_data.action = action
rig.animation_data.action_slot = action.slots[0]
backup = Path(bpy.data.filepath).with_name("hero1_before_soft_run.blend")
if not backup.exists():
    shutil.copy2(bpy.data.filepath, backup)

# Reuse the pose helpers and the Run recipe, without rebuilding other actions/props.
source = Path(__file__).with_name("animate_hero1_more.py").read_text()
tree = ast.parse(source)
helpers = {"action_curves", "reset", "rotate", "shift", "ground_parts", "key_pose"}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in helpers:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(__file__), "exec"))


def signature(clip):
    return [(c.data_path, c.array_index,
             [(tuple(k.co), tuple(k.handle_left), tuple(k.handle_right))
              for k in c.keyframe_points]) for c in action_curves(clip)]


others = {a.name: signature(a) for a in bpy.data.actions if a != action}


def sample(frame):
    scene.frame_set(int(frame), subframe=frame % 1)
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    body = bpy.data.objects["body"]
    values = [body.matrix_world @ v.co for v in body.evaluated_get(graph).data.vertices]
    center = sum(values, Vector()) / len(values)
    floor_height = min((o.matrix_world @ v.co).z for o in legs
                       for v in o.evaluated_get(graph).data.vertices)
    return center, floor_height


before = [sample(1 + i / 8) for i in range(129)]
old_bounce = max(p.z for p, _ in before) - min(p.z for p, _ in before)
floor = min(z for _, z in before)
for curve in action_curves(action):
    curve.keyframe_points.clear()
    for modifier in list(curve.modifiers):
        curve.modifiers.remove(modifier)

for node in tree.body:
    if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
        if ast.unparse(node.iter) == "range(1, 18, 2)":
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(__file__), "exec"))
            break
else:
    raise RuntimeError("Run recipe not found")

for curve in action_curves(action):
    keys = list(curve.keyframe_points)
    if max(k.co.y for k in keys) - min(k.co.y for k in keys) < 1e-6:
        value = keys[0].co.y
        curve.keyframe_points.clear()
        curve.keyframe_points.insert(1, value)
    for key in curve.keyframe_points:
        key.interpolation = "BEZIER"
        key.handle_left_type = key.handle_right_type = "AUTO_CLAMPED"
        key.select_control_point = False
    curve.modifiers.new("CYCLES")
    curve.update()

after = [sample(1 + i / 8) for i in range(129)]
new_bounce = max(p.z for p, _ in after) - min(p.z for p, _ in after)
assert new_bounce < old_bounce * .5, (old_bounce, new_bounce)
assert min(z for _, z in after) >= floor - .002, "Feet penetrate ground"
assert (after[0][0] - after[-1][0]).length < 1e-5
assert all(signature(bpy.data.actions[name]) == sig for name, sig in others.items())
assert all(abs(c.evaluate(1) - c.evaluate(17)) < 1e-5 for c in action_curves(action))
action["description"] = "Steady in-place run; reduced bounce and sway, symmetric flight, 16-frame loop."
scene.frame_start, scene.frame_end = 1, 16
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("SOFT_RUN_OK", "body_vertical_range", old_bounce, "->", new_bounce,
      "feet_min", min(z for _, z in after), "floor", floor,
      "keys", sum(len(c.keyframe_points) for c in action_curves(action)))
