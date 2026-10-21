"""Check axe grip, arc, clearance and restrained joints in the chop preview."""
from pathlib import Path
import json
import math
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from validate_hero1_till_preview import shaft_touches_palm
import hero1_work_actions as work


def validate():
    r,m=work.rig(),work.mesh();prop=bpy.data.objects['PREVIEW_Axe']
    action=bpy.data.actions['chop'];length=int(action['loop_frames'])
    contact_start=float(action['contact_hold_start']);contact_end=float(action['contact_hold_end'])
    trunk=bpy.data.objects['PREVIEW_Trunk'];trunk.hide_viewport=False
    front=float(trunk['contact_front_x'])
    prop.hide_viewport=False
    armids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(('forearm.','hand.')) for g in v.groups)}
    armfaces=[tuple(p.vertices) for p in m.data.polygons if all(i in armids for i in p.vertices)]
    side_faces={}
    for side in ('L','R'):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(('upper_arm.','forearm.','hand.')) and m.vertex_groups[g.group].name.endswith('.'+side) for g in v.groups)}
        side_faces[side]=[tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    obstacles={}
    for name in ('body','head'):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name==name for g in v.groups)}
        obstacles[name]=[tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    report={'samples':length*4+1,'palm_z_range':[100,-100], 'axe_head_z_range':[100,-100],
            'axe_head_y_range':[100,-100],
            'shaft_heading_range_degrees':[1000,-1000], 'max_shaft_tilt_degrees':0,
            'max_socket_error':0,'max_grip_offset':0,'min_axe_z':100,
            'max_wrist_swing_degrees':0,'max_wrist_twist_degrees':0,
            'elbow_flex_range_degrees':[100,-100],'max_arm_width_error':0}
    report.update(grip_spacing_range_m=[100,-100],min_hand_order_in_body_m=100)
    feet=None;max_foot_motion=0;max_foot_xy_motion=0;min_stance=100;max_knee=0;min_knee=100
    knees=None;max_knee_motion=0;hip_y_values=[];hip_yaw_values=[]
    foot_floor=[100,-100]
    foot_ids={side:[v.index for v in m.data.vertices if v.co.z<.021 and any(m.vertex_groups[g.group].name=='shin.'+side for g in v.groups)] for side in ('L','R')}
    contact_points=[];contact_heads=[];maximum_depth=-100
    for k in range(length*4+1):
        f=1+k*.25;work.set_action('chop',f)
        vs=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
        toolpoints=[prop.matrix_world@v.co for v in prop.data.vertices]
        toolfaces=[tuple(p.vertices) for p in prop.data.polygons]
        for name,faces in obstacles.items():
            assert not BVHTree.FromPolygons(vs,faces).overlap(BVHTree.FromPolygons(vs,armfaces)),('Forearm/hand hits '+name,f)
            assert not BVHTree.FromPolygons(vs,faces).overlap(BVHTree.FromPolygons(toolpoints,toolfaces)),('Axe hits '+name,f)
        left,right=r.pose.bones['hand.L'],r.pose.bones['hand.R']
        assert not BVHTree.FromPolygons(vs,side_faces['L']).overlap(BVHTree.FromPolygons(vs,side_faces['R'])),('Arms cross/intersect',f)
        tool=r.pose.bones['tool.R'];axis=(tool.tail-tool.head).normalized()
        delta=(right.head+right.tail-left.head-left.tail)*.5
        spacing=delta.dot(axis)
        report['grip_spacing_range_m']=[min(report['grip_spacing_range_m'][0],spacing),max(report['grip_spacing_range_m'][1],spacing)]
        body=r.pose.bones['body']
        body_rotation=body.matrix.to_quaternion()@body.bone.matrix_local.to_quaternion().inverted()
        hand_order=delta.dot(body_rotation@Vector((0,1,0)))
        report['min_hand_order_in_body_m']=min(report['min_hand_order_in_body_m'],hand_order)
        assert .165<spacing<.215 and hand_order>.12,('Hands swap or cross',f,spacing,hand_order)
        report['max_shaft_tilt_degrees']=max(report['max_shaft_tilt_degrees'],abs(math.degrees(math.asin(max(-1,min(1,axis.z))))))
        heading=math.degrees(math.atan2(axis.y,axis.x))
        if k:
            heading += 360*round((previous_heading-heading)/360)
        previous_heading=heading
        report['shaft_heading_range_degrees']=[min(report['shaft_heading_range_degrees'][0],heading),max(report['shaft_heading_range_degrees'][1],heading)]
        for p in (left,right):
            centre=(p.head+p.tail)*.5
            delta=centre-tool.head
            offset=(delta-axis*delta.dot(axis)).length
            report['max_grip_offset']=max(report['max_grip_offset'],offset)
            assert shaft_touches_palm(tool,p,shaft_min=-.22,shaft_max=.51),('Hand leaves axe handle',p.name,f,offset)
            report['palm_z_range']=[min(report['palm_z_range'][0],centre.z),max(report['palm_z_range'][1],centre.z)]
            sw,tw=p.rotation_quaternion.normalized().to_swing_twist('Y')
            report['max_wrist_swing_degrees']=max(report['max_wrist_swing_degrees'],math.degrees(sw.angle))
            report['max_wrist_twist_degrees']=max(report['max_wrist_twist_degrees'],abs(math.degrees(tw)))
        for side in ('L','R'):
            q=r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ')
            flex=math.degrees(q.z)
            report['elbow_flex_range_degrees']=[min(report['elbow_flex_range_degrees'][0],flex),max(report['elbow_flex_range_degrees'][1],flex)]
            assert -145<flex<0 and max(abs(q.x),abs(q.y))<.001,('Elbow reverses/twists',f,side,tuple(q))
            for segment in ('upper_arm','forearm'):
                p=r.pose.bones[segment+'.'+side]
                report['max_arm_width_error']=max(report['max_arm_width_error'],abs(p.scale.x-1),abs(p.scale.z-1))
            bend=math.degrees(r.pose.bones['shin.'+side].rotation_quaternion.angle)
            max_knee=max(max_knee,bend);min_knee=min(min_knee,bend)
        current_feet=[r.pose.bones['shin.'+side].tail.copy() for side in ('L','R')]
        if feet is None:feet=current_feet
        max_foot_motion=max(max_foot_motion,max((p-q).length for p,q in zip(feet,current_feet)))
        max_foot_xy_motion=max(max_foot_xy_motion,max(math.hypot(p.x-q.x,p.y-q.y) for p,q in zip(feet,current_feet)))
        min_stance=min(min_stance,abs(current_feet[1].y-current_feet[0].y))
        current_knees=[r.pose.bones['shin.'+side].head.copy() for side in ('L','R')]
        if knees is None:knees=current_knees
        max_knee_motion=max(max_knee_motion,max((p-q).length for p,q in zip(knees,current_knees)))
        hip=r.pose.bones['hips'];hip_y_values.append(hip.head.y)
        rotation=hip.matrix.to_quaternion()@hip.bone.matrix_local.to_quaternion().inverted()
        hip_yaw_values.append(math.degrees(rotation.to_euler('XYZ').z))
        for side in ('L','R'):
            low=min(vs[i].z for i in foot_ids[side])
            foot_floor=[min(foot_floor[0],low),max(foot_floor[1],low)]
        report['max_socket_error']=max(report['max_socket_error'],(prop.matrix_world.translation-r.matrix_world@tool.head).length)
        zs=[(prop.matrix_world@v.co).z for v in prop.data.vertices]
        report['min_axe_z']=min(report['min_axe_z'],min(zs))
        head=prop.matrix_world@Vector((-.10,.48,0))
        report['axe_head_z_range']=[min(report['axe_head_z_range'][0],head.z),max(report['axe_head_z_range'][1],head.z)]
        report['axe_head_y_range']=[min(report['axe_head_y_range'][0],head.y),max(report['axe_head_y_range'][1],head.y)]
        edge=prop.matrix_world@Vector((-.195,.55,0))
        maximum_depth=max(maximum_depth,edge.x-front)
        if contact_start+.05<=f<=contact_end-.05:
            contact_points.append(edge);contact_heads.append(tool.matrix.copy())
    assert .49<report['palm_z_range'][0]<.54 and report['palm_z_range'][1]<.63,report
    assert report['max_shaft_tilt_degrees']<1,report
    assert report['shaft_heading_range_degrees'][1]-report['shaft_heading_range_degrees'][0]>155,report
    assert report['axe_head_z_range'][1]-report['axe_head_z_range'][0]<.18,report
    assert report['min_axe_z']>0 and report['max_socket_error']<1e-5,report
    assert report['max_arm_width_error']<1e-6,report
    assert report['max_wrist_swing_degrees']<35.1 and report['max_wrist_twist_degrees']<25.1,report
    assert min_stance>.33 and 5<min_knee<=max_knee<45,(min_stance,min_knee,max_knee)
    assert max_foot_xy_motion<.004,('Feet slide',max_foot_xy_motion)
    assert -.001<foot_floor[0] and foot_floor[1]<.010,('Foot sinks/floats',foot_floor)
    assert max_knee_motion>.012 and max_knee-min_knee>10,('Legs do not follow',max_knee_motion,min_knee,max_knee)
    assert max(hip_y_values)-min(hip_y_values)>.03 and max(hip_yaw_values)-min(hip_yaw_values)>18,('No weight transfer')
    assert contact_points
    hold_motion=max((p-contact_points[0]).length for p in contact_points)
    assert hold_motion<.002,('Axe keeps sweeping during embedded pause',hold_motion)
    assert all(.005<p.x-front<.04 and -.12<p.y<.14 for p in contact_points),('Edge misses front trunk',contact_points)
    assert maximum_depth<.04,('Axe passes through trunk',maximum_depth)
    contact=contact_points[len(contact_points)//2]
    work.set_action('chop',float(action.get('extraction_end_frame',1+.78*length)))
    extracted=prop.matrix_world@Vector((-.195,.55,0))
    assert extracted.x<front-.02 and contact.x-extracted.x>.05,('Axe is not pulled out',contact,extracted)
    report.update(contact_pause_seconds=(contact_end-contact_start)/48,contact_pause_motion_m=hold_motion,
                  max_trunk_depth_m=maximum_depth,extraction_distance_m=contact.x-extracted.x,
                  stance_width_m=min_stance,knee_bend_degrees=[min_knee,max_knee],max_foot_motion_m=max_foot_motion,
                  max_foot_planar_motion_m=max_foot_xy_motion,foot_floor_range_m=foot_floor,
                  max_knee_motion_m=max_knee_motion,hip_side_shift_m=max(hip_y_values)-min(hip_y_values),
                  hip_yaw_range_degrees=[min(hip_yaw_values),max(hip_yaw_values)])
    assert .18<report['contact_pause_seconds']<.25,report
    report['strike_seconds']=(float(action['swing_end_frame'])-float(action['swing_start_frame']))/48
    report['extraction_seconds']=(float(action['extraction_end_frame'])-contact_end)/48
    assert report['strike_seconds']<=.09 and report['extraction_seconds']<=.09,report
    (Path(bpy.data.filepath).parent/'hero1_chop_preview_validation.json').write_text(json.dumps(report,indent=2))
    print('CHOP_PREVIEW_VALIDATED',json.dumps(report))
    return report


if __name__=='__main__':validate()
