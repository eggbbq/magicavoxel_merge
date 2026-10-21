"""Separate authored arm length from width, preserving wrists and tool motion.

Four unweighted scale-compensation bones cancel longitudinal scaling before
the next joint rotates. This avoids inherited shear and exports as glTF TRS.
Simple property drivers preserve the existing sparse scale interpolation;
the game exporter samples their evaluated transforms into ordinary animation.
Run apply() once in the current authoring scene, then save/export separately.
"""
from pathlib import Path
import json
import math
import bpy

ROOT = Path(__file__).resolve().parent.parent
MARKER = 'hero_arm_width_fixed_v1'


def curves(action):
    return [c for l in action.layers for s in l.strips
            for b in s.channelbags for c in b.fcurves]


def select(rig, action, frame):
    rig.animation_data.action = action
    rig.animation_data.action_slot = action.slots[0]
    bpy.context.scene.frame_set(math.floor(frame), subframe=frame % 1)
    bpy.context.view_layer.update()


def snapshot(rig):
    return {p.name: {'head': p.head.copy(), 'tail': p.tail.copy(),
                     'matrix': p.matrix.copy()} for p in rig.pose.bones}


def constant(curve, value=1):
    curve.keyframe_points.clear()
    curve.keyframe_points.insert(1, value)
    curve.update()


def driver(bone, index, expression, paths):
    d = bone.driver_add('scale', index).driver
    d.type = 'SCRIPTED'
    for name, path in paths.items():
        v = d.variables.new()
        v.name = name
        v.type = 'SINGLE_PROP'
        v.targets[0].id = bone.id_data
        v.targets[0].data_path = path
    d.expression = expression


def apply():
    scene = bpy.context.scene
    rig = bpy.data.objects['Hero']
    assert not scene.get(MARKER), 'Arm-width correction already applied.'
    assert len(rig.data.bones) == 14
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    backup = ROOT / 'hero1_before_arm_thickness_fix.blend'
    if not backup.exists():
        bpy.ops.wm.save_as_mainfile(filepath=str(backup), copy=True)
    actions = list(bpy.data.actions)
    originals = {}
    for a in actions:
        times = {1 + a['loop_frames'] * i / 64 for i in range(65)}
        times.update(float(k.co.x) for c in curves(a) for k in c.keyframe_points)
        originals[a.name] = {}
        for f in sorted(times):
            select(rig, a, f)
            for side in ('L', 'R'):
                for segment in ('upper_arm', 'forearm'):
                    s = rig.pose.bones[segment + '.' + side].scale
                    assert max(s) - min(s) < 2e-6, (a.name, f, segment, tuple(s))
            originals[a.name][f] = snapshot(rig)

    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for side in ('L', 'R'):
        for parent, child, name in (('upper_arm', 'forearm', 'elbow_scale'),
                                    ('forearm', 'hand', 'wrist_scale')):
            p = rig.data.edit_bones[parent + '.' + side]
            c = rig.data.edit_bones[child + '.' + side]
            h = rig.data.edit_bones.new(name + '.' + side)
            h.head = p.tail
            h.tail = p.tail + (p.tail - p.head).normalized() * .02
            h.roll = p.roll
            h.parent = p
            h.use_connect = False
            h.use_deform = False
            c.use_connect = False
            c.parent = h
    bpy.ops.object.mode_set(mode='OBJECT')

    for side in ('L', 'R'):
        lower = rig.pose.bones['forearm.' + side]
        hand = rig.pose.bones['hand.' + side]
        lower['authored_length_ratio'] = 1.0
        for axis in 'xyz':
            hand['authored_palm_scale_' + axis] = 1.0
        for name in ('elbow_scale', 'wrist_scale'):
            p = rig.pose.bones[name + '.' + side]
            p.rotation_mode = 'QUATERNION'
            p['purpose'] = 'Cancel arm length scaling before the next joint rotates'
    # Relocate the original curves without resampling their keys or handles.
    for a in actions:
        for side in ('L', 'R'):
            for c in curves(a):
                if c.data_path == f'pose.bones["upper_arm.{side}"].scale' and c.array_index != 1:
                    constant(c)
                elif c.data_path == f'pose.bones["forearm.{side}"].scale':
                    if c.array_index == 1:
                        c.data_path = f'pose.bones["forearm.{side}"]["authored_length_ratio"]'
                        c.array_index = 0
                    else:
                        constant(c)
                elif c.data_path == f'pose.bones["hand.{side}"].scale':
                    axis = 'xyz'[c.array_index]
                    c.data_path = f'pose.bones["hand.{side}"]["authored_palm_scale_{axis}"]'
                    c.array_index = 0
    for side in ('L', 'R'):
        u = f'pose.bones["upper_arm.{side}"].scale[1]'
        f = f'pose.bones["forearm.{side}"]["authored_length_ratio"]'
        driver(rig.pose.bones['elbow_scale.' + side], 1, '1.0 / upper', {'upper': u})
        driver(rig.pose.bones['forearm.' + side], 1, 'upper * ratio', {'upper': u, 'ratio': f})
        driver(rig.pose.bones['wrist_scale.' + side], 1, '1.0 / (upper * ratio)', {'upper': u, 'ratio': f})
        for i, axis in enumerate('xyz'):
            h = f'pose.bones["hand.{side}"]["authored_palm_scale_{axis}"]'
            driver(rig.pose.bones['hand.' + side], i, 'upper * ratio * palm',
                   {'upper': u, 'ratio': f, 'palm': h})
    rig.update_tag()
    report = {'bones': len(rig.data.bones), 'clips': {}, 'samples': 0}
    for a in actions:
        stats = {'joint_position_error_m': 0.0, 'palm_tool_matrix_error': 0.0,
                 'arm_local_width_error': 0.0, 'body_matrix_error': 0.0,
                 'samples': len(originals[a.name])}
        for frame, original in originals[a.name].items():
            select(rig, a, frame)
            now = snapshot(rig)
            for n, old in original.items():
                stats['joint_position_error_m'] = max(stats['joint_position_error_m'],
                    (old['head']-now[n]['head']).length, (old['tail']-now[n]['tail']).length)
                if n.startswith('hand.') or n == 'tool.R':
                    key = 'palm_tool_matrix_error'
                elif n.startswith(('upper_arm.', 'forearm.')):
                    continue
                else:
                    key = 'body_matrix_error'
                stats[key] = max(stats[key], max(abs(old['matrix'][i][j]-now[n]['matrix'][i][j])
                                                 for i in range(4) for j in range(4)))
            for side in ('L', 'R'):
                for name in ('upper_arm', 'forearm'):
                    p = rig.pose.bones[name+'.'+side]
                    stats['arm_local_width_error'] = max(stats['arm_local_width_error'],
                                                         abs(p.scale.x-1), abs(p.scale.z-1))
            report['samples'] += 1
        report['clips'][a.name] = stats
        assert stats['joint_position_error_m'] < 1e-5, (a.name, stats)
        # Tiny original rest-axis differences can accumulate in tool's large
        # counter-scale; actual joint/palm/tool positions must remain within 10um.
        assert stats['palm_tool_matrix_error'] < 1e-4, (a.name, stats)
        assert stats['body_matrix_error'] < 1e-7, (a.name, stats)
        assert stats['arm_local_width_error'] < 1e-7, (a.name, stats)
    scene[MARKER] = True
    (ROOT/'hero1_arm_thickness_validation.json').write_text(json.dumps(report, indent=2))
    print('ARM_WIDTH_FIXED', json.dumps(report))
    return report


if __name__ == '__main__':
    apply()
