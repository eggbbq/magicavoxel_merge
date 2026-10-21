"""Compare corrected palms and tool attachment against the pre-correction .blend."""
from pathlib import Path
import json
import math
import bpy

ROOT=Path(__file__).resolve().parent.parent

def select(a,f):
    r=bpy.data.objects['Hero'];tool=r.pose.bones['tool.R']
    tool.location=(0,0,0);tool.rotation_quaternion=(1,0,0,0);tool.scale=(1,1,1)
    r.animation_data.action=a;r.animation_data.action_slot=a.slots[0]
    bpy.context.scene.frame_set(math.floor(f),subframe=f%1);bpy.context.view_layer.update()
    return r

def run():
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'hero1_before_wrist_relaxation.blend'))
    references={};original_curves={}
    for a in bpy.data.actions:
        original_curves[a.name]=[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points])
                                for l in a.layers for s in l.strips for b in s.channelbags for c in b.fcurves
                                if not any(n in c.data_path for n in ('upper_arm.','forearm.','hand.','tool.R'))]
        for k in range(2*int(a['loop_frames'])+1):
            f=1+k*.5;r=select(a,f);tool=r.pose.bones['tool.R']
            references[(a.name,f)]=([(r.pose.bones['hand.'+side].head+r.pose.bones['hand.'+side].tail)*.5 for side in ('L','R')],
                                   tool.head.copy(),tool.matrix.to_quaternion())
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'hero1_animated.blend'))
    report={}
    for a in bpy.data.actions:
        now=[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points])
             for l in a.layers for s in l.strips for b in s.channelbags for c in b.fcurves
             if not any(n in c.data_path for n in ('upper_arm.','forearm.','hand.','tool.R'))]
        assert original_curves[a.name]==now,(a.name,'Body/head/leg curves changed')
        palm_error=tool_error=angle=0;floor=100
        if a.name=='till':
            # Hidden preview props are excluded from dependency-graph evaluation.
            # Enable it in this isolated process before measuring its world bounds.
            bpy.data.objects['PREVIEW_Hoe'].hide_viewport=False
        for k in range(2*int(a['loop_frames'])+1):
            f=1+k*.5;r=select(a,f);old,old_tool,old_q=references[(a.name,f)]
            for side,expected in zip(('L','R'),old):
                p=r.pose.bones['hand.'+side]
                palm_error=max(palm_error,((p.head+p.tail)*.5-expected).length)
            tool=r.pose.bones['tool.R'];tool_error=max(tool_error,(tool.head-old_tool).length)
            difference=tool.matrix.to_quaternion().rotation_difference(old_q).angle
            angle=max(angle,math.degrees(min(difference,abs(2*math.pi-difference))))
            if a.name=='till':
                obj=bpy.data.objects['PREVIEW_Hoe'];floor=min(floor,min((obj.matrix_world@v.co).z for v in obj.data.vertices))
        report[a.name]={'palm_centre_max_error':palm_error,'tool_socket_position_max_error':tool_error,
                        'tool_socket_rotation_max_degrees':angle,'body_head_leg_curves_unchanged':True}
        if a.name=='till':report[a.name]['min_hoe_z']=floor
        assert palm_error<.013,(a.name,'Palm path changed excessively',palm_error)
        assert tool_error<.0003 and angle<.2,(a.name,'Tool trajectory changed',tool_error,angle)
        if a.name=='till':assert floor>-.001,('Hoe penetrates ground',floor)
    (ROOT/'hero1_wrist_preservation_validation.json').write_text(json.dumps(report,indent=2))
    print('WRIST_PRESERVATION',json.dumps(report))

if __name__=='__main__':run()
