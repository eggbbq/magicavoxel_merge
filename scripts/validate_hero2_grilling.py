"""Check hand-held skewer alignment, turning, fire placement and body clearance."""
import json,math
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import hero1_work_actions as work
import hero_action_preview as preview
from retarget_voxel_character import action_fingerprint
from create_hero2_grilling import ROOT,NAME,LENGTH


def validate():
    r,m=work.rig(),work.mesh();s=bpy.context.scene
    report=json.loads((ROOT/'hero2_grilling_report.json').read_text())
    assert all(action_fingerprint(bpy.data.actions[n])==h for n,h in report['preserved_fingerprints'].items())
    obj=bpy.data.objects['PREVIEW_Skewer']
    def faces(prefixes):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(prefixes) for g in v.groups)}
        return [tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    armfaces=faces(('forearm.','hand.'));obstacles={n:faces(n) for n in ('body','head')}
    foot_ids=[v.index for v in m.data.vertices if v.co.z<.021 and any(m.vertex_groups[g.group].name.startswith('shin.') for g in v.groups)]
    reference=None;foot_motion=grip_error=0.;food_z=[];left_z=[];rolls=[];floor=1.;over_fire=0;max_swing=max_twist=0.
    for i in range(LENGTH*4+1):
        f=1+i*.25;work.set_action(NAME,f);preview.update_preview_objects(s,r);bpy.context.view_layer.update()
        vs=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
        if reference is None:reference=vs
        foot_motion=max(foot_motion,max((vs[k]-reference[k]).length for k in foot_ids));floor=min(floor,min(v.z for v in vs))
        ps=[obj.matrix_world@v.co for v in obj.data.vertices]
        skewer_bvh=BVHTree.FromPolygons(ps,[tuple(p.vertices) for p in obj.data.polygons])
        for name,fs in obstacles.items():
            body_bvh=BVHTree.FromPolygons(vs,fs)
            assert not body_bvh.overlap(BVHTree.FromPolygons(vs,armfaces)),('Arm collision',name,f)
            assert not body_bvh.overlap(skewer_bvh),('Skewer collision',name,f)
        palm=r.pose.bones['hand.R'];centre=(palm.head+palm.tail)*.5
        socket=obj.matrix_world@Vector((0,0,0));grip_error=max(grip_error,(socket-centre).length)
        food=obj.matrix_world@Vector((0,.345,0));food_z.append(food.z)
        if f<=1+.47*LENGTH:
            assert abs(food.x-.72)<.16 and abs(food.y-.10)<.12 and food.z>.36,('Food not over fire',f,tuple(food))
            over_fire+=1
        left=r.pose.bones['hand.L'];left_z.append((left.head.z+left.tail.z)*.5)
        rotation=obj.matrix_world.to_quaternion();axis=rotation@Vector((0,1,0));normal=rotation@Vector((0,0,1))
        # Project the cube's marked face against world-up around the shaft.
        up=Vector((0,0,1));up=(up-axis*up.dot(axis)).normalized()
        roll=math.degrees(math.atan2(axis.dot(up.cross(normal)),up.dot(normal)))
        if rolls:
            roll=rolls[-1]+(roll-rolls[-1]+180)%360-180
        rolls.append(roll)
        for side in ('L','R'):
            hinge=r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ')
            assert -145<math.degrees(hinge.z)<0 and abs(hinge.x)+abs(hinge.y)<.001,('Reverse elbow',f,side)
            q=r.pose.bones['hand.'+side].rotation_quaternion.normalized();swing,twist=q.to_swing_twist('Y')
            max_swing=max(max_swing,math.degrees(swing.angle));max_twist=max(max_twist,abs(math.degrees(twist)))
            assert swing.angle<math.radians(35.1) and abs(twist)<math.radians(25.1),('Wrist twisted',f,side)
            for n in ('upper_arm.','forearm.'):
                scale=r.pose.bones[n+side].scale
                assert abs(scale.x-1)<1e-6 and abs(scale.z-1)<1e-6,('Arm thickens',f,n,side)
        assert {o.name for o in s.objects if o.get('preview_only') and not o.hide_viewport}=={'PREVIEW_Skewer','PREVIEW_Campfire'}
    assert foot_motion<1e-6 and floor>-.001
    assert grip_error<.00061,grip_error
    assert max(food_z)-min(food_z)>.20,('No inspection lift',food_z)
    assert max(left_z)-min(left_z)>.09,('No visible fanning',left_z)
    assert max(rolls)-min(rolls)>50,('No visible skewer turning',rolls)
    seam=max((a-b).length for a,b in zip(reference,vs));assert seam<1e-5,seam
    result={'samples':LENGTH*4+1,'preserved_actions':len(report['preserved_fingerprints']),
            'foot_motion_m':foot_motion,'min_ground_z':floor,'grip_error_m':grip_error,
            'food_height_range_m':[min(food_z),max(food_z)],'left_hand_height_range_m':[min(left_z),max(left_z)],
            'skewer_turn_degrees':max(rolls)-min(rolls),'food_over_fire_samples':over_fire,
            'max_wrist_swing_degrees':max_swing,'max_wrist_twist_degrees':max_twist,
            'loop_seam_m':seam,'forearm_and_skewer_body_head_collision':False}
    (ROOT/'hero2_grilling_validation.json').write_text(json.dumps(result,indent=2));print('GRILLING_VALIDATED',json.dumps(result))
    work.set_action(NAME,1);preview.sync_scene(s,force=True)
    return result


if __name__=='__main__':validate()
