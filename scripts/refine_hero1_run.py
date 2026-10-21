"""Fast, airborne Run with smooth periodic motion. Blender -b hero1_actions.blend --python this_file."""

import ast
import math
from pathlib import Path
import shutil

import bpy
from mathutils import Quaternion, Vector

scene = bpy.context.scene
rig = bpy.data.objects["Hero_WalkRig"]
hero = bpy.data.objects["hero"]
legs = [o for o in hero.children if o.type == "MESH" and "leg" in o.name]
action = bpy.data.actions["Hero_Run"]
rig.animation_data.action = action
rig.animation_data.action_slot = action.slots[0]
backup = Path(bpy.data.filepath).with_name("hero1_before_airborne_run.blend")
if not backup.exists():
    shutil.copy2(bpy.data.filepath, backup)

source = Path(__file__).with_name("animate_hero1_more.py").read_text()
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "action_curves", "reset", "rotate", "shift", "ground_parts", "key_pose"
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(__file__), "exec"))


def signature(clip):
    return [(c.data_path, c.array_index,
             [(tuple(k.co), tuple(k.handle_left), tuple(k.handle_right))
              for k in c.keyframe_points]) for c in action_curves(clip)]


others = {a.name: signature(a) for a in bpy.data.actions if a != action}


def foot_height():
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    return min((o.matrix_world @ v.co).z for o in legs
               for v in o.evaluated_get(graph).data.vertices)


scene.frame_set(1)
floor = foot_height()
rig.animation_data.action = None
reset()
rotate("leg.L", ((0, 1, 0), -44))
rotate("leg.R", ((0, 1, 0), 44))
ground_parts(legs)
base = rig.data.bones["hips"].matrix_local.to_quaternion() @ rig.pose.bones["hips"].location
rig.animation_data.action = action
rig.animation_data.action_slot = action.slots[0]
for c in action_curves(action):
    c.keyframe_points.clear()
    for mod in list(c.modifiers):
        c.modifiers.remove(mod)

PERIOD = 16
DATES = [1 + PERIOD * i / 8 for i in range(9)]


def pose(frame):
    reset()
    phase = 2 * math.pi * (frame - 1) / PERIOD
    swing, sine = math.cos(phase), math.sin(phase)
    rotate("body", ((0, 1, 0), 11), ((0, 0, 1), 1.2 * sine))
    rotate("head", ((0, 1, 0), -6), ((0, 0, 1), -1.2 * sine))
    rotate("leg.L", ((0, 1, 0), -44 * swing))
    rotate("leg.R", ((0, 1, 0), 44 * swing))
    rotate("arm.L", ((0, 1, 0), 48 * swing), ((1, 0, 0), -8))
    rotate("arm.R", ((0, 1, 0), -48 * swing), ((1, 0, 0), 8))
    for name, recovery in (("leg.L", max(0, -sine) ** 2),
                           ("leg.R", max(0, sine) ** 2)):
        shift(name, (0, 0, .055 * recovery))
        rig.pose.bones[name].scale.y = 1 - .14 * recovery
    # One smooth rise/fall per step; no per-frame minimum-foot corrections.
    shift("hips", base + Vector((0, 0, .095 * sine ** 2)))
    rig["sleep_lids"], rig["chop_props"], rig["tree_hit"] = 0, 0, 0


for frame in DATES:
    pose(frame)
    key_pose(frame)


def values(frame):
    pose(frame)
    return {(c.data_path, c.array_index):
            (rig.path_resolve(c.data_path)[c.array_index]
             if c.data_path.startswith("pose.") else rig.path_resolve(c.data_path))
            for c in action_curves(action)}


# Explicit periodic tangents keep velocity continuous at every pose and at the seam.
epsilon = .002
derivatives = {}
for frame in DATES:
    left, right = values(frame - epsilon), values(frame + epsilon)
    derivatives[frame] = {key: (right[key] - left[key]) / (2 * epsilon) for key in left}
for c in action_curves(action):
    points = list(c.keyframe_points)
    if max(k.co.y for k in points) - min(k.co.y for k in points) < 1e-6:
        value = points[0].co.y
        c.keyframe_points.clear()
        c.keyframe_points.insert(1, value)
    else:
        for k in c.keyframe_points:
            frame, value = k.co
            slope = derivatives[frame][(c.data_path, c.array_index)]
            width = (PERIOD / 8) / 3
            k.handle_left_type = k.handle_right_type = "FREE"
            k.handle_left = (frame - width, value - slope * width)
            k.handle_right = (frame + width, value + slope * width)
    for k in c.keyframe_points:
        k.interpolation = "BEZIER"
        k.select_control_point = False
    c.modifiers.new("CYCLES")
    c.update()


def sample(frame):
    scene.frame_set(int(frame), subframe=frame % 1)
    bpy.context.view_layer.update()
    return foot_height(), rig.pose.bones["hips"].matrix.translation.copy()


samples = [sample(1 + i / 16) for i in range(PERIOD * 16 + 1)]
assert min(z for z, _ in samples) > floor - .002, "Ground penetration"
assert all(abs(sample(f)[0] - floor) < 1e-5 for f in (1, 1 + PERIOD / 2, 1 + PERIOD))
assert all(sample(f)[0] > floor + .05 for f in (1 + PERIOD / 4, 1 + 3 * PERIOD / 4)), "Missing flight"
assert (samples[0][1] - samples[-1][1]).length < 1e-5
curves = action_curves(action)
assert all(abs(c.evaluate(1) - c.evaluate(1 + PERIOD)) < 1e-5 for c in curves)
velocity_error = max(abs((c.evaluate(1.001)-c.evaluate(1))/.001
                         - (c.evaluate(1 + PERIOD)-c.evaluate(1 + PERIOD - .001))/.001) for c in curves)
assert velocity_error < .001, velocity_error
assert all(signature(bpy.data.actions[n]) == sig for n, sig in others.items())
action["loop_frames"] = PERIOD
action["description"] = "Airborne run; 16-frame stride, smooth symmetric bounce and continuous periodic tangents."
scene.frame_start, scene.frame_end = 1, PERIOD
scene.render.fps = 24
scene.render.fps_base = 1
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("AIRBORNE_RUN_OK", "period", PERIOD, "bounce", max(v.z for _, v in samples)-min(v.z for _, v in samples),
      "clearance", max(z for z, _ in samples)-floor, "velocity_error", velocity_error,
      "keys", sum(len(c.keyframe_points) for c in curves))
