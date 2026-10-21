"""Check visible panting, gravity-hanging arms, contact and preserved clips."""
import json
import math
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree
import hero1_work_actions as work
import hero_action_preview as preview
from retarget_voxel_character import action_fingerprint
from create_hero2_exhausted_breathing import NAME, LENGTH, ROOT


def validate():
    r, m = work.rig(), work.mesh()
    original = json.loads((ROOT / 'hero2_exhausted_breathing_report.json').read_text())
    assert all(action_fingerprint(bpy.data.actions[n]) == h for n, h in original['preserved_fingerprints'].items())
    def faces(prefix):
        ids = {v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(prefix) for g in v.groups)}
        return [tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    obstacles = {n: faces(n) for n in ('body', 'head')}
    arms = faces(('forearm.', 'hand.'))
    foot_ids = [v.index for v in m.data.vertices if v.co.z < .021 and any(m.vertex_groups[g.group].name.startswith('shin.') for g in v.groups)]
    pitches, heads, shoulders, foot_z, palm_drop = [], [], [], [], []
    first = None
    foot_motion = 0.
    for i in range(LENGTH * 4 + 1):
        work.set_action(NAME, 1 + i * .25)
        preview.update_preview_objects(bpy.context.scene, r)
        bpy.context.view_layer.update()
        vs = [v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
        if first is None:
            first = vs
        foot_motion = max(foot_motion, max((vs[k] - first[k]).length for k in foot_ids))
        foot_z.extend(vs[k].z for k in foot_ids)
        for label, fs in obstacles.items():
            assert not BVHTree.FromPolygons(vs, fs).overlap(BVHTree.FromPolygons(vs, arms)), ('Arm collision', label, i)
        for label, values in (('body', pitches), ('head', heads)):
            p = r.pose.bones[label]
            direction = (p.tail - p.head).normalized()
            values.append(math.degrees(math.atan2(direction.x, direction.z)))
        shoulders.append(r.pose.bones['upper_arm.L'].head.z)
        for side in ('L', 'R'):
            for segment in ('upper_arm.', 'forearm.'):
                p = r.pose.bones[segment + side]
                assert (p.tail - p.head).normalized().z < -.97, ('Arm does not hang', i, p.name)
                assert max(abs(c - 1) for c in p.scale) < 1e-5, ('Arm scale changed', i, p.name)
            elbow = r.pose.bones['forearm.' + side].rotation_quaternion.to_euler('XYZ')
            assert -15 < math.degrees(elbow.z) < -1 and abs(elbow.x) + abs(elbow.y) < .001
            hand = r.pose.bones['hand.' + side]
            assert hand.rotation_quaternion.angle < .001, ('Wrist bends', i, side)
            palm_drop.append(r.pose.bones['upper_arm.' + side].head.z - (hand.head.z + hand.tail.z) * .5)
        assert not any(o.get('preview_only') and not o.hide_viewport for o in bpy.context.scene.objects)
    breaths = sum(pitches[i] < pitches[i-1] and pitches[i] <= pitches[i+1] for i in range(1, len(pitches)-1))
    assert breaths == 2, breaths
    assert min(pitches) > 30 and max(pitches) - min(pitches) > 9
    assert min(heads) > 47
    assert max(shoulders) - min(shoulders) > .05
    assert foot_motion < 1e-6 and min(foot_z) > -.001
    # Shoulder to palm-centre reach is .32 m; a relaxed elbow shortens it slightly.
    assert min(palm_drop) > .315, min(palm_drop)
    seam = max((a - b).length for a, b in zip(first, vs))
    assert seam < 1e-6
    result = {'action': NAME, 'samples': LENGTH * 4 + 1, 'breaths_per_loop': breaths,
              'body_pitch_degrees': [min(pitches), max(pitches)],
              'head_pitch_degrees': [min(heads), max(heads)],
              'shoulder_height_change_m': max(shoulders) - min(shoulders),
              'foot_motion_m': foot_motion, 'min_foot_z': min(foot_z),
              'minimum_palm_drop_from_shoulder_m': min(palm_drop),
              'loop_seam_m': seam, 'forearm_head_body_collision': False,
              'wrist_rotation_degrees': 0., 'preserved_actions': len(original['preserved_fingerprints'])}
    (ROOT / 'hero2_exhausted_breathing_validation.json').write_text(json.dumps(result, indent=2))
    print('EXHAUSTED_BREATHING_VALIDATED', json.dumps(result))
    work.set_action(NAME, 1)
    preview.sync_scene(bpy.context.scene, force=True)
    return result


if __name__ == '__main__':
    validate()
