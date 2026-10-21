"""Validate an overarm throw, strong full-body motion and detached preview flight."""
from pathlib import Path
import json
import math
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import hero1_work_actions as work
import hero_action_preview as preview
from validate_hero1_till_preview import shaft_touches_palm


def validate():
    r,m=work.rig(),work.mesh();s=bpy.context.scene;prop=bpy.data.objects['PREVIEW_Javelin']
    a=bpy.data.actions['javelin_throw'];length=int(a['loop_frames']);release=float(a['release_frame'])
    def faces(prefixes):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(prefixes) for g in v.groups)}
        return [tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    armfaces=faces(('forearm.','hand.'));obstacles={n:faces((n,)) for n in ('body','head')}
    propfaces=[tuple(p.vertices) for p in prop.data.polygons]
    feet_ids={side:[v.index for v in m.data.vertices if v.co.z<.021 and any(m.vertex_groups[g.group].name=='shin.'+side for g in v.groups)] for side in ('L','R')}
    report=dict(samples=length*4+1,release_frame=release,right_hand_x_m=[100,-100],right_hand_z_m=[100,-100],
                body_pitch_degrees=[100,-100],body_yaw_degrees=[100,-100],hips_yaw_degrees=[100,-100],
                knee_flex_degrees=[100,-100],elbow_flex_degrees=[100,-100],max_wrist_swing_degrees=0,
                max_wrist_twist_degrees=0,min_character_z_m=100,max_held_spear_socket_error_m=0,
                left_foot_max_height_m=0,right_foot_max_height_m=0,max_lowest_foot_z_m=-100)
    hip_x=[];hand_positions={};left_plant=[];start=None;late_palms=[];late_elbows=[]
    for k in range(length*4+1):
        f=1+k*.25;work.set_action('javelin_throw',f);preview.update_preview_objects(s,r);bpy.context.view_layer.update()
        vs=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
        if start is None:start=vs
        for label,fs in obstacles.items():
            assert not BVHTree.FromPolygons(vs,fs).overlap(BVHTree.FromPolygons(vs,armfaces)),('Arm hits '+label,f)
            if f<release:
                ps=[prop.matrix_world@v.co for v in prop.data.vertices]
                assert not BVHTree.FromPolygons(vs,fs).overlap(BVHTree.FromPolygons(ps,propfaces)),('Spear hits '+label,f)
        right=r.pose.bones['hand.R'];centre=(right.head+right.tail)*.5;hand_positions[f]=centre.copy()
        report['right_hand_x_m']=[min(report['right_hand_x_m'][0],centre.x),max(report['right_hand_x_m'][1],centre.x)]
        report['right_hand_z_m']=[min(report['right_hand_z_m'][0],centre.z),max(report['right_hand_z_m'][1],centre.z)]
        if f<release:
            err=(prop.matrix_world.translation-r.matrix_world@r.pose.bones['tool.R'].head).length
            report['max_held_spear_socket_error_m']=max(report['max_held_spear_socket_error_m'],err)
            assert shaft_touches_palm(r.pose.bones['tool.R'],right,shaft_min=-.09,shaft_max=.09),('Hand leaves spear grip',f)
        for side in ('L','R'):
            q=r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ');angle=math.degrees(q.z)
            assert -145<angle<0 and max(abs(q.x),abs(q.y))<.001,('Elbow reverses/twists',side,f,tuple(q))
            report['elbow_flex_degrees']=[min(report['elbow_flex_degrees'][0],angle),max(report['elbow_flex_degrees'][1],angle)]
            if f>=77:
                palm=r.pose.bones['hand.'+side];centre=(palm.head+palm.tail)*.5
                late_palms.append(centre.z);late_elbows.append(abs(angle))
                assert centre.z<.53 and abs(angle)<10,('Recovery arm remains raised/bent',side,f,centre.z,angle)
            for segment in ('upper_arm','forearm','hand'):
                assert max(abs(v-1) for v in r.pose.bones[segment+'.'+side].scale)<1e-6,('Throw stretches arm',side,f)
            sw,tw=r.pose.bones['hand.'+side].rotation_quaternion.normalized().to_swing_twist('Y')
            report['max_wrist_swing_degrees']=max(report['max_wrist_swing_degrees'],math.degrees(sw.angle))
            report['max_wrist_twist_degrees']=max(report['max_wrist_twist_degrees'],abs(math.degrees(tw)))
            for segment in ('upper_arm','forearm'):
                scale=r.pose.bones[segment+'.'+side].scale
                assert abs(scale.x-1)<1e-6 and abs(scale.z-1)<1e-6,('Arm width changes',side,f)
            angle=math.degrees(r.pose.bones['shin.'+side].rotation_quaternion.angle)
            report['knee_flex_degrees']=[min(report['knee_flex_degrees'][0],angle),max(report['knee_flex_degrees'][1],angle)]
        for name in ('body','hips'):
            p=r.pose.bones[name];q=p.matrix.to_quaternion()@p.bone.matrix_local.to_quaternion().inverted()
            yaw=math.degrees(q.to_euler('XYZ').z)
            report[name+'_yaw_degrees']=[min(report[name+'_yaw_degrees'][0],yaw),max(report[name+'_yaw_degrees'][1],yaw)]
            if name=='body':
                up=(p.tail-p.head).normalized();pitch=math.degrees(math.atan2(math.hypot(up.x,up.y),up.z))*(1 if up.x>=0 else -1)
                report['body_pitch_degrees']=[min(report['body_pitch_degrees'][0],pitch),max(report['body_pitch_degrees'][1],pitch)]
        hip_x.append(r.pose.bones['hips'].head.x)
        lows={side:min(vs[i].z for i in feet_ids[side]) for side in ('L','R')}
        report['left_foot_max_height_m']=max(report['left_foot_max_height_m'],lows['L'])
        report['right_foot_max_height_m']=max(report['right_foot_max_height_m'],lows['R'])
        report['max_lowest_foot_z_m']=max(report['max_lowest_foot_z_m'],min(lows.values()))
        report['min_character_z_m']=min(report['min_character_z_m'],min(v.z for v in vs))
        if 39<=f<=59:left_plant.append(r.pose.bones['shin.L'].tail.copy())
        if f>=float(a['standing_frame']):
            assert (r.pose.bones['hips'].head-r.data.bones['hips'].head_local).length<1e-6,('Standing root displaced',f)
            for side in ('L','R'):
                for segment in ('upper_arm','forearm'):
                    p=r.pose.bones[segment+'.'+side]
                    assert (p.tail-p.head).normalized().dot(Vector((0,0,-1)))>.99,('Standing arm not hanging straight',side,segment,f)
                assert r.pose.bones['shin.'+side].rotation_quaternion.angle<1e-5,('Standing knee remains bent',side,f)
                assert r.pose.bones['hand.'+side].rotation_quaternion.angle<1e-5,('Standing wrist not neutral',side,f)
    seam=max((p-q).length for p,q in zip(start,vs))
    assert not a['looping'] and a['ends_relaxed'],('Throw must finish relaxed as a single action')
    assert seam<1e-6,('Throw does not return to the natural starting stance',seam)
    assert report['right_hand_x_m'][0]<-.3 and report['right_hand_x_m'][1]>.72,report
    assert hand_positions[39.].z>.95 and hand_positions[release].x>.7 and hand_positions[release].z>.7,report
    assert report['body_pitch_degrees'][0]<-18 and report['body_pitch_degrees'][1]>35,report
    assert report['body_yaw_degrees'][1]-report['body_yaw_degrees'][0]>95,report
    assert report['hips_yaw_degrees'][1]-report['hips_yaw_degrees'][0]>45 and max(hip_x)-min(hip_x)>.20,report
    assert report['left_foot_max_height_m']>.09 and report['right_foot_max_height_m']>.12,report
    assert report['min_character_z_m']>-.001 and report['max_lowest_foot_z_m']<.015,report
    assert report['max_wrist_swing_degrees']<35.1 and report['max_wrist_twist_degrees']<25.1,report
    assert report['max_held_spear_socket_error_m']<1e-5,report
    slide=max(math.hypot(p.x-left_plant[0].x,p.y-left_plant[0].y) for p in left_plant)
    assert slide<.02,('Plant foot slides',slide)
    work.set_action('javelin_throw',release);preview.update_preview_objects(s,r);bpy.context.view_layer.update();origin=prop.matrix_world.translation.copy()
    work.set_action('javelin_throw',release+4);preview.update_preview_objects(s,r);bpy.context.view_layer.update();fly=prop.matrix_world.translation.copy()
    assert (fly-origin).dot(Vector(prop['release_direction']))>.5 and (fly-r.pose.bones['tool.R'].head).length>.3,('Spear never leaves hand',origin,fly)
    for name,f,hidden in [('javelin_throw',70,True),('javelin_throw',105,True),('javelin_throw',length+1,True),('idle',1,True),('javelin_throw',1,False)]:
        work.set_action(name,f);preview.update_preview_objects(s,r);assert prop.hide_viewport==hidden and prop.hide_render==hidden,('Spear visibility',name,f)
    report.update(start_end_pose_difference_m=seam,plant_foot_slide_m=slide,flight_distance_after_four_frames_m=(fly-origin).length,
                  recovery_palm_height_m=[min(late_palms),max(late_palms)],recovery_max_elbow_flex_degrees=max(late_elbows))
    (Path(bpy.data.filepath).parent/'hero1_javelin_throw_validation.json').write_text(json.dumps(report,indent=2))
    print('JAVELIN_THROW_VALIDATED',json.dumps(report));return report


if __name__=='__main__':validate()
