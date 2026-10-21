"""Slow the Run cadence while preserving all poses and Bezier curve shapes."""

from pathlib import Path
import shutil

import bpy

scene = bpy.context.scene
rig = bpy.data.objects["Hero_WalkRig"]
action = bpy.data.actions["Hero_Run"]
curves = [c for layer in action.layers for strip in layer.strips
          for bag in strip.channelbags for c in bag.fcurves]
old_period = int(action["loop_frames"])
period = 16
factor = period / old_period
backup = Path(bpy.data.filepath).with_name("hero1_before_run_retime.blend")
if not backup.exists():
    shutil.copy2(bpy.data.filepath, backup)
references = [[c.evaluate(1 + old_period * i / 64) for i in range(65)] for c in curves]
for curve in curves:
    for key in curve.keyframe_points:
        co, left, right = key.co.copy(), key.handle_left.copy(), key.handle_right.copy()
        key.co = (1 + (co.x - 1) * factor, co.y)
        key.handle_left = (1 + (left.x - 1) * factor, left.y)
        key.handle_right = (1 + (right.x - 1) * factor, right.y)
    curve.update()
error = max(abs(curve.evaluate(1 + period * i / 64) - values[i])
            for curve, values in zip(curves, references) for i in range(65))
assert error < 1e-5, error
assert all(abs(c.evaluate(1) - c.evaluate(period + 1)) < 1e-5 for c in curves)
rig.animation_data.action = action
rig.animation_data.action_slot = action.slots[0]
action["loop_frames"] = period
action["description"] = "Airborne run; comfortable 16-frame cadence, smooth symmetric bounce and continuous periodic tangents."
scene.frame_start, scene.frame_end = 1, period
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("RETIME_OK", old_period, "->", period, "pose_error", error)
