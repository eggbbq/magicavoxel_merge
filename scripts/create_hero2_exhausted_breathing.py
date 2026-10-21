"""Add a sparse, planted, bowed panting loop without editing approved clips."""
from pathlib import Path
import copy
import json
import math
import sys

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import hero1_work_actions as work
import relax_hero1_wrists as wrists
from retarget_voxel_character import action_fingerprint

NAME = 'exhausted_breathing'
LENGTH = 144
# Fast intake, longer deflation. Two breaths in three seconds, nine main poses.
# phase, torso pitch, additional head bow, relaxed upper-arm forward offset
ROWS = [(0, 43, 22, .025), (.08, 38, 20, .015), (.16, 33, 16, -.015),
        (.32, 39, 20, .010), (.5, 43, 22, .025),
        (.58, 38, 20, .015), (.66, 33, 16, -.015),
        (.82, 39, 20, .010), (1, 43, 22, .025)]


def pose(row):
    r = work.rig()
    work.reset()
    _, pitch, head, sway = row
    # Constant lower-body pose: breathing moves the torso, never the feet.
    thigh, knee = -9, 18
    a, b = math.radians(thigh), math.radians(thigh + knee)
    height = .16 * math.cos(a) + .14 * math.cos(b) + .02 * abs(math.sin(b)) + .002
    hip = r.pose.bones['hips']
    hip.location = hip.bone.matrix_local.to_quaternion().inverted() @ Vector((0, 0, height - .3))
    for side in ('L', 'R'):
        work.rotate('thigh.' + side, thigh)
        work.rotate('shin.' + side, knee)
        r.pose.bones['forearm.' + side]['authored_length_ratio'] = 1.
        for axis in 'xyz':
            r.pose.bones['hand.' + side]['authored_palm_scale_' + axis] = 1.
    work.rotate('body', pitch)
    work.rotate('head', head)
    bpy.context.view_layer.update()
    for side, extra, elbow in (('L', .012, 8), ('R', -.008, 6)):
        # World-space gravity keeps the arms limp as the parent torso breathes.
        work.set_world_direction(r.pose.bones['upper_arm.' + side], (sway + extra, 0, -1))
        r.pose.bones['forearm.' + side].rotation_quaternion = Quaternion((0, 0, 1), math.radians(-elbow))
        r.pose.bones['hand.' + side].rotation_quaternion = (1, 0, 0, 0)
    bpy.context.view_layer.update()
    return {p.name: {'location': tuple(p.location),
                     'rotation_quaternion': tuple(p.rotation_quaternion),
                     'scale': tuple(p.scale)} for p in r.pose.bones}


def build():
    r, s = work.rig(), bpy.context.scene
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    preserved = {a.name: action_fingerprint(a) for a in bpy.data.actions if a.name != NAME}
    old = bpy.data.actions.get(NAME)
    r.animation_data.action = None
    drivers = list(r.animation_data.drivers)
    driven = {(c.data_path, c.array_index) for c in drivers}
    try:
        for c in drivers:
            c.mute = True
        samples = [pose(row) for row in ROWS]
    finally:
        for c in drivers:
            c.mute = False
    for i in range(1, len(samples)):
        for bone in samples[i]:
            q = Quaternion(samples[i][bone]['rotation_quaternion'])
            if q.dot(Quaternion(samples[i-1][bone]['rotation_quaternion'])) < 0:
                q.negate()
            samples[i][bone]['rotation_quaternion'] = tuple(q)
    samples[-1] = copy.deepcopy(samples[0])
    action = bpy.data.actions.new(NAME + '_build')
    action.use_fake_user = True
    action.use_frame_range = True
    action.frame_start, action.frame_end = 1, LENGTH + 1
    action['loop_frames'] = LENGTH
    action['looping'] = True
    action['description'] = '虚弱喘息：低头弓背，双臂松垂；快速吸气、缓慢呼气，双脚站稳'
    action['author_pose_times'] = len(ROWS)
    action['breaths_per_loop'] = 2
    r.animation_data.action = action
    wrists.write_samples(r, action, [1 + row[0] * LENGTH for row in ROWS], samples, list(r.pose.bones.keys()))
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for c in list(bag.fcurves):
                    if (c.data_path, c.array_index) in driven:
                        bag.fcurves.remove(c)
    for side in ('L', 'R'):
        for bone, keys in (('forearm.', ['authored_length_ratio']),
                           ('hand.', ['authored_palm_scale_' + axis for axis in 'xyz'])):
            p = r.pose.bones[bone + side]
            for key in keys:
                p[key] = 1.
                p.keyframe_insert('["' + key + '"]', frame=1, group=p.name)
    if old:
        bpy.data.actions.remove(old)
    action.name = NAME
    action.asset_mark()
    assert all(action_fingerprint(bpy.data.actions[n]) == h for n, h in preserved.items())
    work.set_action(NAME, 1)
    import hero_action_preview as preview
    preview.sync_scene(s, force=True)
    result = {'name': NAME, 'frames': LENGTH, 'fps': s.render.fps,
              'duration': LENGTH / s.render.fps, 'looping': True,
              'author_pose_times': len(ROWS), 'breaths_per_loop': 2,
              'keys': sum(len(c.keyframe_points) for c in work.curves(action)),
              'preserved_actions': list(preserved), 'preserved_fingerprints': preserved,
              'mesh_and_rig_unchanged': True}
    stats = json.loads(s.get('animation_clips', '{}'))
    stats[NAME] = {k: v for k, v in result.items() if k not in ('preserved_actions', 'preserved_fingerprints')}
    s['animation_clips'] = json.dumps(stats)
    (ROOT / 'hero2_exhausted_breathing_report.json').write_text(json.dumps(result, indent=2))
    print('EXHAUSTED_BREATHING_CREATED', json.dumps({k: v for k, v in result.items() if k != 'preserved_fingerprints'}))
    return result


if __name__ == '__main__':
    build()
