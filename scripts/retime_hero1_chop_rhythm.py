"""Speed up strike/extraction while retaining the approved chop geometry.

Map complete Bezier segments, including handles. No extra animation keys.
The short impact hold also supplies the hesitation before pulling the axe out.
"""
from pathlib import Path
import json
import bpy
import hero1_work_actions as work

ROOT = Path(__file__).resolve().parent.parent
SOURCE_LENGTH = 84
LENGTH = 52
PHASES = [0,.18,.31,.355,.40,.69,.735,.78,.84,.94,1]
TIMES = [1,13,23,25,27,37,39,41,44,49,53]
SOURCE_TIMES = [1+SOURCE_LENGTH*t for t in PHASES]


def map_time(frame):
    for x,y in zip(SOURCE_TIMES,TIMES):
        if abs(frame-x)<.00003:
            return float(y)
    i=next((i for i in range(len(TIMES)-1) if frame<SOURCE_TIMES[i+1]),len(TIMES)-2)
    x0,x1=SOURCE_TIMES[i:i+2];y0,y1=TIMES[i:i+2]
    return y0+(frame-x0)*(y1-y0)/(x1-x0)


def apply():
    r=work.rig();s=bpy.context.scene;a=bpy.data.actions['chop']
    if a.get('chop_fast_rhythm_v1'):
        return json.loads(a['rhythm_report'])
    assert int(a['loop_frames'])==SOURCE_LENGTH
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    cs=work.curves(a)
    for c in cs:
        if len(c.keyframe_points)>1:
            assert all(any(abs(k.co.x-t)<.00003 for k in c.keyframe_points) for t in SOURCE_TIMES),(c.data_path,c.array_index,'missing timing anchor')
    # Geometry preservation is checked on the evaluated mesh and attached axe.
    m=work.mesh();axe=bpy.data.objects['PREVIEW_Axe'];references=[]
    sample_frames=sorted({1+k*.25 for k in range(SOURCE_LENGTH*4+1)}|set(SOURCE_TIMES))
    for f in sample_frames:
        work.set_action('chop',f)
        references.append(([v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices],
                           [axe.matrix_world@v.co for v in axe.data.vertices]))
    before=sum(len(c.keyframe_points) for c in cs)
    for c in cs:
        for k in c.keyframe_points:
            co=k.co.copy();left=k.handle_left.copy();right=k.handle_right.copy()
            k.handle_left_type=k.handle_right_type='FREE'
            k.co=(map_time(co.x),co.y)
            k.handle_left=(map_time(left.x),left.y)
            k.handle_right=(map_time(right.x),right.y)
        c.update()
    a.frame_start=1;a.frame_end=LENGTH+1;a['loop_frames']=LENGTH
    for prop in ('contact_hold_start','contact_hold_end','extraction_end_frame'):
        a[prop]=map_time(float(a[prop]))
    a['swing_start_frame']=23.;a['swing_end_frame']=27.
    a['chop_fast_rhythm_v1']=True
    max_mesh=max_axe=0.
    for f,(vertices,toolpoints) in zip(sample_frames,references):
        work.set_action('chop',map_time(f))
        actual=m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices
        max_mesh=max(max_mesh,max((v.co-p).length for v,p in zip(actual,vertices)))
        max_axe=max(max_axe,max(((axe.matrix_world@v.co)-p).length for v,p in zip(axe.data.vertices,toolpoints)))
    assert max_mesh<.00001 and max_axe<.00001,(max_mesh,max_axe)
    assert before==sum(len(c.keyframe_points) for c in cs)
    report=dict(frames=LENGTH,fps=s.render.fps,duration=LENGTH/s.render.fps,
                strike_seconds=4/s.render.fps,contact_pause_seconds=10/s.render.fps,
                extraction_seconds=4/s.render.fps,recovery_seconds=12/s.render.fps,
                source_to_target_frames=list(zip(SOURCE_TIMES,TIMES)),keys=before,
                sampled_poses=len(sample_frames),max_mesh_change_m=max_mesh,max_axe_change_m=max_axe)
    a['rhythm_report']=json.dumps(report)
    clips=json.loads(s.get('animation_clips','{}'));clips['chop'].update(frames=LENGTH,duration=report['duration'])
    s['animation_clips']=json.dumps(clips)
    path=ROOT/'hero1_chop_axe_report.json'
    if path.exists():
        data=json.loads(path.read_text());data.update(frames=LENGTH,contact_pause_seconds=report['contact_pause_seconds'],rhythm=report)
        path.write_text(json.dumps(data,indent=2))
    (ROOT/'hero1_chop_rhythm_validation.json').write_text(json.dumps(report,indent=2))
    work.set_action('chop',1)
    print('CHOP_RHYTHM_UPDATED',json.dumps(report))
    return report


if __name__=='__main__':apply()
