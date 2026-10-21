"""Validate scythe grips, sweep, ground clearance, hinges and whole-body motion."""
from pathlib import Path
import json
import math
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import hero1_work_actions as work
from validate_hero1_till_preview import shaft_touches_palm


def validate():
    r,m=work.rig(),work.mesh();prop=bpy.data.objects['PREVIEW_Scythe'];prop.hide_viewport=False
    a=bpy.data.actions['harvest'];length=int(a['loop_frames'])
    def faces(prefixes):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(prefixes) for g in v.groups)}
        return [tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    armfaces=faces(('forearm.','hand.'));sidefaces={s:faces(('upper_arm.'+s,'forearm.'+s,'hand.'+s)) for s in ('L','R')}
    obs={n:faces((n,)) for n in ('body','head')};pf=[tuple(p.vertices) for p in prop.data.polygons]
    report=dict(samples=length*4+1,min_tool_z=100,min_character_z=100,hand_spacing_m=[100,-100],
                max_socket_error_m=0,max_hand_perpendicular_error_m=0,elbow_flex_degrees=[100,-100],
                max_wrist_swing_degrees=0,max_wrist_twist_degrees=0,cut_blade_height_m=[100,-100],
                max_cut_blade_height_span_m=0,heading_degrees=[100,-100],hips_yaw_degrees=[100,-100],
                knee_flex_degrees=[100,-100],min_stance_width_m=100,max_return_blade_height_m=-100)
    start=None;first_feet=None;max_foot_slide=0;hip_sway=[]
    for k in range(length*4+1):
        f=1+k*.25;work.set_action('harvest',f)
        vs=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
        ps=[prop.matrix_world@v.co for v in prop.data.vertices];blade=ps[16:]
        if start is None:start=vs
        report['min_tool_z']=min(report['min_tool_z'],min(v.z for v in ps))
        report['min_character_z']=min(report['min_character_z'],min(v.z for v in vs))
        for n,fs in obs.items():
            assert not BVHTree.FromPolygons(vs,fs).overlap(BVHTree.FromPolygons(vs,armfaces)),('Arm hits '+n,f)
            assert not BVHTree.FromPolygons(vs,fs).overlap(BVHTree.FromPolygons(ps,pf)),('Scythe hits '+n,f)
        assert not BVHTree.FromPolygons(vs,sidefaces['L']).overlap(BVHTree.FromPolygons(vs,sidefaces['R'])),('Arms intersect',f)
        p=r.pose.bones['tool.R'];axis=(p.tail-p.head).normalized()
        report['max_socket_error_m']=max(report['max_socket_error_m'],(prop.matrix_world.translation-r.matrix_world@p.head).length)
        palms={s:(r.pose.bones['hand.'+s].head+r.pose.bones['hand.'+s].tail)*.5 for s in ('L','R')}
        delta=palms['R']-palms['L'];spacing=delta.dot(axis)
        report['hand_spacing_m']=[min(report['hand_spacing_m'][0],spacing),max(report['hand_spacing_m'][1],spacing)]
        report['max_hand_perpendicular_error_m']=max(report['max_hand_perpendicular_error_m'],(delta-axis*spacing).length)
        assert .16<spacing<.225,('Hands swap or slide',f,spacing)
        for side in ('L','R'):
            assert shaft_touches_palm(p,r.pose.bones['hand.'+side],shaft_min=-.22,shaft_max=.98),('Hand leaves handle',side,f)
            q=r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ');angle=math.degrees(q.z)
            assert -145<angle<0 and max(abs(q.x),abs(q.y))<.001,('Elbow reverses/twists',f,side,tuple(q))
            report['elbow_flex_degrees']=[min(report['elbow_flex_degrees'][0],angle),max(report['elbow_flex_degrees'][1],angle)]
            sw,tw=r.pose.bones['hand.'+side].rotation_quaternion.normalized().to_swing_twist('Y')
            report['max_wrist_swing_degrees']=max(report['max_wrist_swing_degrees'],math.degrees(sw.angle))
            report['max_wrist_twist_degrees']=max(report['max_wrist_twist_degrees'],abs(math.degrees(tw)))
            for segment in ('upper_arm','forearm'):
                scale=r.pose.bones[segment+'.'+side].scale
                assert abs(scale.x-1)<1e-6 and abs(scale.z-1)<1e-6,('Arm width changes',side,f)
            knee=math.degrees(r.pose.bones['shin.'+side].rotation_quaternion.angle)
            report['knee_flex_degrees']=[min(report['knee_flex_degrees'][0],knee),max(report['knee_flex_degrees'][1],knee)]
        feet=[r.pose.bones['shin.'+side].tail.copy() for side in ('L','R')]
        if first_feet is None:first_feet=feet
        max_foot_slide=max(max_foot_slide,max(Vector((p.x-q.x,p.y-q.y)).length for p,q in zip(feet,first_feet)))
        report['min_stance_width_m']=min(report['min_stance_width_m'],(feet[1]-feet[0]).length)
        hip=r.pose.bones['hips'];world_rotation=hip.matrix.to_quaternion()@hip.bone.matrix_local.to_quaternion().inverted()
        yaw=math.degrees(world_rotation.to_euler('XYZ').z);hip_sway.append(hip.head.y)
        report['hips_yaw_degrees']=[min(report['hips_yaw_degrees'][0],yaw),max(report['hips_yaw_degrees'][1],yaw)]
        heading=math.degrees(math.atan2(axis.y,axis.x))
        report['heading_degrees']=[min(report['heading_degrees'][0],heading),max(report['heading_degrees'][1],heading)]
        if a['cut_start_frame']<=f<=a['cut_end_frame']:
            lo,hi=min(p.z for p in blade),max(p.z for p in blade)
            report['cut_blade_height_m']=[min(report['cut_blade_height_m'][0],lo),max(report['cut_blade_height_m'][1],hi)]
            report['max_cut_blade_height_span_m']=max(report['max_cut_blade_height_span_m'],hi-lo)
        if .68<=(f-1)/length<=.88:report['max_return_blade_height_m']=max(report['max_return_blade_height_m'],min(p.z for p in blade))
    seam=max((p-q).length for p,q in zip(start,vs));assert seam<1e-6,('Loop seam',seam)
    assert report['min_tool_z']>.01 and report['min_character_z']>-.001,report
    assert .02<report['cut_blade_height_m'][0] and report['cut_blade_height_m'][1]<.26,report
    assert report['max_cut_blade_height_span_m']<.05 and report['max_return_blade_height_m']>.35,report
    assert report['heading_degrees'][1]-report['heading_degrees'][0]>95,report
    assert report['max_wrist_swing_degrees']<35.1 and report['max_wrist_twist_degrees']<25.1,report
    assert report['max_socket_error_m']<1e-6 and report['max_hand_perpendicular_error_m']<.03,report
    assert report['min_stance_width_m']>.33 and report['hips_yaw_degrees'][1]-report['hips_yaw_degrees'][0]>24,report
    assert 5<report['knee_flex_degrees'][0]<report['knee_flex_degrees'][1]<46 and max_foot_slide<.005,report
    assert max(hip_sway)-min(hip_sway)>.04,report
    report.update(loop_pose_error_m=seam,max_foot_planar_motion_m=max_foot_slide,hip_sway_m=max(hip_sway)-min(hip_sway))
    (Path(bpy.data.filepath).parent/'hero1_harvest_validation.json').write_text(json.dumps(report,indent=2))
    print('HARVEST_VALIDATED',json.dumps(report));return report


if __name__=='__main__':validate()
