"""Check lowered grip, anatomical hinges, collisions and fishing/pull handoff."""
from pathlib import Path
import json
import math
import bpy
from mathutils.bvhtree import BVHTree
import hero1_work_actions as work


def validate():
    r=work.rig();m=work.mesh();report={}
    def points():return [v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
    def faces(prefixes):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(prefixes) for g in v.groups)}
        return [tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    arms=faces(('forearm.','hand.'));obstacles=[faces(('body',)),faces(('head',))]
    work.set_action('fishing',1);entry=points();entry_tool=r.pose.bones['tool.R'].matrix.copy()
    for name in ('fishing','fishing_pull'):
        length=int(bpy.data.actions[name]['loop_frames'])
        result=dict(samples=length*4+1,palm_z_range=[100,-100],palm_x_range=[100,-100],
                    elbow_flex_degrees=[100,-100],max_grip_error_m=0,max_wrist_swing_degrees=0,
                    max_wrist_twist_degrees=0,max_entry_difference_m=0)
        for k in range(length*4+1):
            f=1+k*.25;work.set_action(name,f);vs=points()
            for obs in obstacles:
                assert not BVHTree.FromPolygons(vs,obs).overlap(BVHTree.FromPolygons(vs,arms)),(name,'Arm intersects torso/head',f)
            tool=r.pose.bones['tool.R'];axis=(tool.tail-tool.head).normalized()
            left,right=(r.pose.bones['hand.'+s] for s in ('L','R'))
            delta=(left.head+left.tail-right.head-right.tail)*.5
            result['max_grip_error_m']=max(result['max_grip_error_m'],(delta-axis*.08).length)
            for side in ('L','R'):
                hand=r.pose.bones['hand.'+side];centre=(hand.head+hand.tail)*.5
                result['palm_z_range']=[min(result['palm_z_range'][0],centre.z),max(result['palm_z_range'][1],centre.z)]
                result['palm_x_range']=[min(result['palm_x_range'][0],centre.x),max(result['palm_x_range'][1],centre.x)]
                q=r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ');angle=math.degrees(q.z)
                assert -145<angle<0 and max(abs(q.x),abs(q.y))<.001,(name,'Elbow reverses/twists',side,f,tuple(q))
                result['elbow_flex_degrees']=[min(result['elbow_flex_degrees'][0],angle),max(result['elbow_flex_degrees'][1],angle)]
                swing,twist=hand.rotation_quaternion.normalized().to_swing_twist('Y')
                result['max_wrist_swing_degrees']=max(result['max_wrist_swing_degrees'],math.degrees(swing.angle))
                result['max_wrist_twist_degrees']=max(result['max_wrist_twist_degrees'],abs(math.degrees(twist)))
                for segment in ('upper_arm','forearm'):
                    scale=r.pose.bones[segment+'.'+side].scale
                    assert abs(scale.x-1)<1e-6 and abs(scale.z-1)<1e-6
            if name=='fishing':result['max_entry_difference_m']=max(result['max_entry_difference_m'],max((p-q).length for p,q in zip(entry,vs)))
        assert result['max_grip_error_m']<.02,result
        assert result['max_wrist_swing_degrees']<35.1 and result['max_wrist_twist_degrees']<25.1,result
        if name=='fishing':
            assert .53<result['palm_z_range'][0]<.56 and result['palm_z_range'][1]<.61,result
            assert result['palm_x_range'][0]>.28,result
        else:
            assert 72<=length<=80 and result['palm_z_range'][1]>1.12,result
            for f in (1,length+1):
                work.set_action(name,f)
                error=max((p-q).length for p,q in zip(points(),entry))
                socket=max(abs(r.pose.bones['tool.R'].matrix[i][j]-entry_tool[i][j]) for i in range(4) for j in range(4))
                assert error<1e-5 and socket<1e-5,('Fishing handoff pops',f,error,socket)
        report[name]=result
    (Path(bpy.data.filepath).parent/'hero1_fishing_refinement_validation.json').write_text(json.dumps(report,indent=2))
    print('FISHING_REFINEMENT_VALIDATED',json.dumps(report))
    return report


if __name__=='__main__':validate()
