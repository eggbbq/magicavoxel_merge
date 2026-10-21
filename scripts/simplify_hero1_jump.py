"""Reduce Jump keys while checking mesh motion: blender -b hero1.blend --python scripts/simplify_hero1_jump.py."""

from pathlib import Path
import shutil

import bpy


scene = bpy.context.scene
rig = bpy.data.objects["Hero_WalkRig"]
jump = bpy.data.actions["Hero_Jump"]
rig.animation_data.action = jump
backup = Path(bpy.data.filepath).with_name("hero1_before_simplify.blend")
if not backup.exists():
    shutil.copy2(bpy.data.filepath, backup)


def curves(action):
    return [curve for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags for curve in bag.fcurves]


def signature(action):
    return [(curve.data_path, curve.array_index,
             [(tuple(key.co), tuple(key.handle_left), tuple(key.handle_right))
              for key in curve.keyframe_points]) for curve in curves(action)]


other_actions = {name: signature(bpy.data.actions[name])
                 for name in ("Hero_Walk_InPlace", "Hero_Idle")}
parts = [obj for obj in bpy.data.objects["hero"].children if obj.type == "MESH"]
samples = [1 + index * 0.5 for index in range(97)]


def snapshot(frame):
    scene.frame_set(int(frame), subframe=frame - int(frame))
    graph = bpy.context.evaluated_depsgraph_get()
    return {obj.name: [obj.matrix_world @ vertex.co
                       for vertex in obj.evaluated_get(graph).data.vertices]
            for obj in parts}


def lowest_foot(data):
    return min(vertex.z for name, vertices in data.items() if "leg" in name
               for vertex in vertices)


reference = {frame: snapshot(frame) for frame in samples}
source = jump.copy()
original_curves = curves(source)
target_curves = curves(jump)
before = sum(len(curve.keyframe_points) for curve in target_curves)
key_frames = {1, 4, 8, 10, 12, 18, 24, 30, 36, 40, 46, 49}


def rebuild():
    for original, target in zip(original_curves, target_curves):
        values = [original.evaluate(frame) for frame in samples]
        constant = max(values) - min(values) < 1e-6
        frames = [1] if constant else sorted(key_frames)
        target.keyframe_points.clear()
        target.keyframe_points.add(len(frames))
        for index, (frame, key) in enumerate(zip(frames, target.keyframe_points)):
            value = original.evaluate(frame)
            slope = (original.evaluate(frame + 0.01) - original.evaluate(frame - 0.01)) / 0.02
            left_span = (frame - frames[index - 1]) / 3 if index else 1
            right_span = (frames[index + 1] - frame) / 3 if index + 1 < len(frames) else 1
            key.co = (frame, value)
            key.interpolation = "BEZIER"
            key.handle_left_type = key.handle_right_type = "FREE"
            key.handle_left = (frame - left_span, value - slope * left_span)
            key.handle_right = (frame + right_span, value + slope * right_span)
            key.select_control_point = False
            key.select_left_handle = key.select_right_handle = False
        target.update()
    jump.update_tag()


# Add a supporting pose only if mesh deviation or ground contact requires it.
for attempt in range(38):
    rebuild()
    errors = []
    max_motion_error = max_ground_error = 0
    for frame in samples:
        current = snapshot(frame)
        motion = max((a - b).length for name in reference[frame]
                     for a, b in zip(reference[frame][name], current[name]))
        ground = abs(lowest_foot(current) - lowest_foot(reference[frame])) if frame <= 12 or frame >= 36 else 0
        max_motion_error = max(max_motion_error, motion)
        max_ground_error = max(max_ground_error, ground)
        errors.append((max(motion / 0.006, ground / 0.002), frame))
    score, worst_frame = max(errors)
    if score <= 1:
        break
    available = [frame for frame in range(2, 49) if frame not in key_frames]
    if not available:
        raise RuntimeError("Could not preserve the original motion within tolerance")
    key_frames.add(min(available, key=lambda frame: abs(frame - worst_frame)))
else:
    raise RuntimeError("Key reduction failed validation; file has not been saved")

assert all(signature(bpy.data.actions[name]) == state for name, state in other_actions.items())
assert all(max((a - b).length for a, b in zip(reference[frame][name], snapshot(frame)[name])) < 1e-5
           for frame in (1, 8, 12, 24, 36, 40, 46, 49) for name in reference[frame])
after = sum(len(curve.keyframe_points) for curve in target_curves)
assert after < before
summary = (f"Jump keys: {before} -> {after}; {len(key_frames)} pose frames; "
           f"max mesh deviation {max_motion_error:.6f}; ground deviation {max_ground_error:.6f}")
jump["key_reduction"] = summary
bpy.data.actions.remove(source)
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("SIMPLIFIED:", summary)
print("POSE_FRAMES:", sorted(key_frames))
