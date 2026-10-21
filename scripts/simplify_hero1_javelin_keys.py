"""Reduce javelin_throw attachment keys using source derivatives and world-space error.

Keep authored body/arm poses, constant reset channels, and restrained wrists.
Fit Hermite curves to the evaluated tool motion rather than resampling every
quarter frame or recomputing monotone tangents after every added key.
"""
from pathlib import Path
import json
import math
import bpy
import hero1_work_actions as work

ROOT=Path(__file__).resolve().parent.parent
POSITION_TOL=.0005
ANGLE_TOL=math.radians(.7)
VERTEX_TOL=.008


def counts(action):
    cs=work.curves(action)
    tool=[c for c in cs if '"tool.R"' in c.data_path]
    return {'keys':sum(len(c.keyframe_points) for c in cs),
            'times':len({round(k.co.x,5) for c in cs for k in c.keyframe_points}),
            'tool_keys':sum(len(c.keyframe_points) for c in tool),
            'tool_times':len({round(k.co.x,5) for c in tool for k in c.keyframe_points})}


def reduce():
    r=work.rig();s=bpy.context.scene;a=bpy.data.actions['javelin_throw']
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero1_before_javelin_throw_key_reduction.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    before=counts(a)
    length=int(a['loop_frames'])
    frames=[1+k*.125 for k in range(length*8+1)]
    mesh=work.mesh();prop=bpy.data.objects['PREVIEW_Javelin']
    local_points=[v.co.copy() for v in prop.data.vertices]
    source=[]
    for f in frames:
        work.set_action('javelin_throw',f)
        source.append((r.pose.bones['tool.R'].matrix.copy(),
                       [v.co.copy() for v in mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]))
    allcurves=work.curves(a)
    saved=a.copy();saved.name='TEMP_JavelinCurveReference';saved.use_fake_user=False
    reference={(c.data_path,c.array_index):c for c in work.curves(saved)}
    const=0
    for c in allcurves:
        values=[k.co.y for k in c.keyframe_points]
        # Sub-micrometre authoring noise should not create thirteen reset keys.
        if len(values)>1 and max(values)-min(values)<2e-6:
            value=values[0];c.keyframe_points.clear();c.keyframe_points.insert(1,value);c.update();const+=1
    variable_tool=[c for c in work.curves(a) if '"tool.R"' in c.data_path and len(c.keyframe_points)>1]
    keys={float(k.co.x) for c in work.curves(a) if '"tool.R"' not in c.data_path for k in c.keyframe_points}
    keys.update((1.,float(length+1)))
    def evaluate(c,f):
        while f<1:f+=length
        while f>length+1:f-=length
        return c.evaluate(f)
    def fit():
        times=sorted(keys)
        for c in variable_tool:
            ref=reference[c.data_path,c.array_index]
            c.keyframe_points.clear()
            for f in times:c.keyframe_points.insert(f,ref.evaluate(f),options={'FAST'})
            c.update()
            for i,k in enumerate(c.keyframe_points):
                f=times[i]
                derivative=0. if not a['looping'] and f in (1.,float(length+1)) else (evaluate(ref,f+.002)-evaluate(ref,f-.002))/.004
                if a.get('contact_hold_start',-1)-1e-4<=f<=a.get('contact_hold_end',-2)+1e-4:
                    derivative=0
                hl=(f-times[i-1])/3 if i else (times[-1]-times[-2])/3
                hr=(times[i+1]-f)/3 if i+1<len(times) else (times[1]-times[0])/3
                k.interpolation='BEZIER';k.handle_left_type=k.handle_right_type='FREE'
                k.handle_left=(f-hl,k.co.y-derivative*hl)
                k.handle_right=(f+hr,k.co.y+derivative*hr)
            c.update()
    errors={}
    for iteration in range(50):
        fit();worst=(0,None);errors={'tool_position_error_m':0.,'tool_angle_error_degrees':0.,'javelin_vertex_error_m':0.,'character_vertex_error_m':0.}
        for f,(target,vertices) in zip(frames,source):
            work.set_action('javelin_throw',f);actual=r.pose.bones['tool.R'].matrix
            if f>float(a['release_frame']):continue
            distance=(actual.translation-target.translation).length
            angle=actual.to_quaternion().rotation_difference(target.to_quaternion()).angle
            angle=min(angle,abs(math.tau-angle))
            # Attachment origin agrees with tool head; model is offset to bone head.
            vertex_error=max(((actual@v)-(target@v)).length for v in local_points)
            errors['tool_position_error_m']=max(errors['tool_position_error_m'],distance)
            errors['tool_angle_error_degrees']=max(errors['tool_angle_error_degrees'],math.degrees(angle))
            errors['javelin_vertex_error_m']=max(errors['javelin_vertex_error_m'],vertex_error)
            error=max(distance/POSITION_TOL,angle/ANGLE_TOL,vertex_error/VERTEX_TOL)
            # Preserve ground clearance where the blade passes close to it.
            lowest=min((actual@v).z for v in local_points)
            if lowest<-.0002:error=max(error,1+(-.0002-lowest)/.001)
            if f not in keys and error>worst[0]:worst=(error,f)
        if worst[0]<=1.001:break
        keys.add(worst[1])
    assert errors['tool_position_error_m']<POSITION_TOL*1.01 and errors['tool_angle_error_degrees']<math.degrees(ANGLE_TOL)*1.01,errors
    assert errors['javelin_vertex_error_m']<VERTEX_TOL*1.01,errors
    for f,(target,vertices) in zip(frames,source):
        work.set_action('javelin_throw',f)
        actual=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices
        errors['character_vertex_error_m']=max(errors['character_vertex_error_m'],max((v.co-p).length for v,p in zip(actual,vertices)))
    assert errors['character_vertex_error_m']<1e-5,errors
    for c in work.curves(a):
        if '"tool.R"' in c.data_path or len(c.keyframe_points)==1:continue
        old=reference[c.data_path,c.array_index]
        assert [(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points]==[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in old.keyframe_points]
    bpy.data.actions.remove(saved)
    after=counts(a)
    report={'before':before,'after':after,'constant_channels_collapsed':const,'sampled_poses':len(frames),**errors}
    a['tool_pose_times']=after['tool_times'];a['sparse_tool_compensation']=True
    clips=json.loads(s.get('animation_clips','{}'));clips.setdefault('javelin_throw',{}).update(keys=after['keys'],tool_pose_times=after['tool_times'])
    s['animation_clips']=json.dumps(clips)
    (ROOT/'hero1_javelin_throw_key_reduction.json').write_text(json.dumps(report,indent=2))
    work.set_action('javelin_throw',1)
    print('JAVELIN_KEYS_REDUCED',json.dumps(report))
    return report


if __name__=='__main__':reduce()
