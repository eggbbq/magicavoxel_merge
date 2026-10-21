"""Validate the authored hoe preview; leave the selected till Action at frame one."""
import json
import math
import bpy
from pathlib import Path
from mathutils.bvhtree import BVHTree
import hero1_work_actions as work
from mathutils import Vector

def shaft_touches_palm(tool,palm,shaft_min=-.16,shaft_max=.60):
    inverse=palm.matrix.inverted()
    origin=inverse@tool.head
    direction=inverse.to_3x3()@(tool.matrix.to_3x3()@Vector((0,1,0)))
    # The palm is a 4-cm cube; expand its bounds by the wooden shaft radius.
    lower=(-.033,-.013,-.033);upper=(.033,.053,.033)
    lo,hi=shaft_min,shaft_max
    for i in range(3):
        if abs(direction[i])<1e-8:
            if not lower[i]<=origin[i]<=upper[i]:return False
        else:
            a,b=(lower[i]-origin[i])/direction[i],(upper[i]-origin[i])/direction[i]
            lo=max(lo,min(a,b));hi=min(hi,max(a,b))
            if hi<lo:return False
    return True

def validate():
    r,m=work.rig(),work.mesh();prop=bpy.data.objects['PREVIEW_Hoe']
    prop.hide_viewport=False
    armids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(('forearm.','hand.')) for g in v.groups)}
    af=[tuple(p.vertices) for p in m.data.polygons if all(i in armids for i in p.vertices)]
    obstacles={}
    for label in ('body','head'):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name==label for g in v.groups)}
        obstacles[label]=[tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    grip=0;socket_error=0;floor=100;tool_peak=-100;hand_range=[100,-100];floor_character=100
    upper_ranges={side:[100,-100] for side in ('L','R')}
    elbow_ranges={side:[100,-100] for side in ('L','R')}
    elbow_twist=0
    arm_scale_range=[100,-100];palm_scale_range=[100,-100];grip_spacing=[100,-100];peak_pose=None
    for k in range(289):
        work.set_action('till',1+k*.25)
        vs=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
        for label,faces in obstacles.items():
            pairs=BVHTree.FromPolygons(vs,faces).overlap(BVHTree.FromPolygons(vs,af))
            assert not pairs,('Forearm or hand intersects',label,1+k*.25,len(pairs))
        right,left=r.pose.bones['hand.R'],r.pose.bones['hand.L']
        base=(right.head+right.tail)*.5;delta=(left.head+left.tail)*.5-base
        tool=r.pose.bones['tool.R'];axis=(tool.tail-tool.head).normalized()
        along=delta.dot(axis)
        grip=max(grip,(delta-axis*along).length,max(0,.04-along,along-.08))
        grip_spacing=[min(grip_spacing[0],along),max(grip_spacing[1],along)]
        socket_error=max(socket_error,(prop.matrix_world.translation-r.matrix_world@tool.head).length)
        assert shaft_touches_palm(tool,left) and shaft_touches_palm(tool,right),('Hand leaves the wooden handle',1+k*.25)
        zs=[(prop.matrix_world@v.co).z for v in prop.data.vertices]
        floor=min(floor,min(zs));tool_peak=max(tool_peak,max(zs))
        floor_character=min(floor_character,min(v.z for v in vs))
        hand_range=[min(hand_range[0],base.z),max(hand_range[1],base.z)]
        for palm in (left,right):
            palm_scale=palm.length/palm.bone.length
            palm_scale_range=[min(palm_scale_range[0],palm_scale),max(palm_scale_range[1],palm_scale)]
            assert abs(palm_scale-1)<.05,('Palm visibly changes size',palm_scale)
        for side in ('L','R'):
            hinge=r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ')
            flex=math.degrees(hinge.z)
            elbow_ranges[side]=[min(elbow_ranges[side][0],flex),max(elbow_ranges[side][1],flex)]
            elbow_twist=max(elbow_twist,abs(hinge.x),abs(hinge.y))
            assert -145<flex<0,('Elbow hyperextends or reverses bend',side,1+k*.25,flex)
            assert max(abs(hinge.x),abs(hinge.y))<.001,('Elbow twists off hinge axis',side,1+k*.25)
            upper=r.pose.bones['upper_arm.'+side]
            factor=upper.length/upper.bone.length
            arm_scale_range=[min(arm_scale_range[0],factor),max(arm_scale_range[1],factor)]
            d=(upper.tail-upper.head).normalized()
            elevation=math.degrees(math.asin(max(-1,min(1,d.z))))
            upper_ranges[side]=[min(upper_ranges[side][0],elevation),max(upper_ranges[side][1],elevation)]
    assert grip<.025,('Grip offset outside palm/shaft contact allowance',grip)
    assert socket_error<1e-5,('Tool detaches',socket_error)
    assert floor>-.001,('Tool below ground',floor)
    assert floor_character>-.001,('Character below ground',floor_character)
    assert tool_peak>1.48,('Hoe is not raised clearly above the head',tool_peak)
    work.set_action('till',1+72*.34)
    peak_pose={side:{'palm_z':((r.pose.bones['hand.'+side].head+r.pose.bones['hand.'+side].tail)*.5).z,
                     'elbow_flex_degrees':math.degrees(r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ').z)} for side in ('L','R')}
    assert all(p['palm_z']>1.20 and -40<p['elbow_flex_degrees']<0 for p in peak_pose.values()),('Hands not raised and extended',peak_pose)
    assert all(high-low>60 for low,high in upper_ranges.values()),('Upper arms barely move',upper_ranges)
    work.set_action('till',1)
    start=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
    work.set_action('till',73)
    end=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
    seam=max((a-b).length for a,b in zip(start,end));assert seam<1e-6
    report={'arm_length_factor_range':arm_scale_range,'peak_overhead_pose':peak_pose,
            'palm_scale_range':palm_scale_range,
            'pose_times':len(work.SPECS['till'][2]),'length_frames':72,'fps':48,
            'forearm_hand_collision_samples_passed':289,'two_hand_grip_error':grip,
            'socket_position_error':socket_error,'min_tool_z':floor,'max_tool_z':tool_peak,
            'grip_spacing_range':grip_spacing,
            'right_hand_z_range':hand_range,'upper_arm_world_elevation_ranges_degrees':upper_ranges,
            'signed_elbow_flex_ranges_degrees':elbow_ranges,'elbow_off_axis_rotation_radians':elbow_twist,
            'min_character_z':floor_character,'loop_pose_error':seam}
    (Path(bpy.data.filepath).parent/'hero1_till_arm_validation.json').write_text(json.dumps(report,indent=2))
    work.set_action('till')
    print('TILL_PREVIEW_VALIDATED',json.dumps(report))
    return report

if __name__=='__main__':validate()
