"""Pain and frightened surrender loops with small whole-body tremors.

Keep the existing rig/mesh and all approved Actions. A three/four-Hz tremor
uses 13/17 sparse poses rather than a key on every exported frame.
"""
from pathlib import Path
import json
import math
import bpy
from mathutils import Vector
import hero1_work_actions as work
from hero1_raised_hands_actions import palm_frame

ROOT=Path(__file__).resolve().parent.parent
NAMES=('stomach_pain','surrendering')
SPECS={'stomach_pain':(48,True,[i/12 for i in range(13)]),
       'surrendering':(48,True,[i/16 for i in range(17)])}

def rest_point(bone,point):
    return bone.matrix@bone.bone.matrix_local.inverted()@Vector(point)

def delta_rotation(bone):
    return bone.matrix.to_quaternion()@bone.bone.matrix_local.to_quaternion().inverted()

def pose(name,t):
    work.reset();r=work.rig()
    cycles=3 if name=='stomach_pain' else 4
    w=math.sin(math.tau*cycles*t);v=math.cos(math.tau*cycles*t)
    if name=='stomach_pain':
        thigh=-34-.9*w;knee=68+1.8*w
        pitch=40+1.4*w;roll=1.0*v;head=13+1.3*v
    else:
        thigh=-66-.6*w;knee=130+1.2*w
        pitch=32+1.0*w;roll=.8*v;head=23+1.1*v
    a,b=math.radians(thigh),math.radians(thigh+knee)
    height=.16*math.cos(a)+.14*math.cos(b)+.02*abs(math.sin(b))+.002
    hip=r.pose.bones['hips']
    hip.location=hip.bone.matrix_local.to_quaternion().inverted()@Vector((0,.002*v,height-.3))
    work.rotate('body',pitch,roll=roll,yaw=.4*w)
    work.rotate('head',head,roll=-.5*w)
    for side in ('L','R'):
        work.rotate('thigh.'+side,thigh);work.rotate('shin.'+side,knee)
    bpy.context.view_layer.update()
    if name=='stomach_pain':
        body=r.pose.bones['body'];rot=delta_rotation(body)
        right=rest_point(body,(.215,.045,.60))
        right_axis=rot@Vector((0,0,-1));right_normal=rot@Vector((-1,0,0))
        shoulder=r.pose.bones['upper_arm.L'].head
        left=shoulder+Vector((.035+.004*w,-.055,-.318+.003*v))
        left_axis=Vector((0,0,-1));left_normal=Vector((1,0,0))
        stretch_right,stretch_left=1.10,1.04
    else:
        # Hands stay just outside the scalp, following the bowed head as it trembles.
        head_bone=r.pose.bones['head'];rot=delta_rotation(head_bone)
        right=rest_point(head_bone,(-.015,.235,1.08))
        left=rest_point(head_bone,(-.015,-.195,1.08))
        right_axis=left_axis=rot@Vector((0,0,1))
        right_normal=rot@Vector((0,-1,0));left_normal=rot@Vector((0,1,0))
        stretch_right=stretch_left=1.08
    errors=[work.solve_arm('R',right,right_axis,stretch_right,hinge=True),
            work.solve_arm('L',left,left_axis,stretch_left,hinge=True)]
    palm_frame('R',right_axis,right_normal);palm_frame('L',left_axis,left_normal)
    from relax_hero1_wrists import relax_current_pose
    relax_current_pose(r)
    values={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),
                    'scale':tuple(p.scale)} for p in r.pose.bones}
    return values,max(errors)

def build():
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero1_before_distress_actions.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    original=(work.SPECS,work.NEW_NAMES,work.pose)
    work.SPECS=SPECS;work.NEW_NAMES=NAMES;work.pose=pose
    try:report=work.build_actions()
    finally:work.SPECS,work.NEW_NAMES,work.pose=original
    for name in NAMES:
        a=bpy.data.actions[name]
        a['description']={'stomach_pain':'弓身屈腿，右手捂腹，左手垂下，全身疼痛发抖',
                          'surrendering':'低头蹲下，双手抱头，全身害怕发抖'}[name]
        a['tremor_hz']=3 if name=='stomach_pain' else 4
    (ROOT/'hero1_distress_actions_report.json').write_text(json.dumps(report,indent=2))
    work.set_action(NAMES[0])
    return report

def save():
    r=work.rig();s=bpy.context.scene;work.set_action('stomach_pain')
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);bpy.context.view_layer.objects.active=r
    bpy.ops.object.mode_set(mode='POSE')
    import magicavoxel_hero_action_preview as preview
    preview.sync_scene(s,force=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'hero1_animated.blend'))
    p=ROOT/'hero1_animation_report.json';report=json.loads(p.read_text())
    report['clips']=json.loads(s['animation_clips']);p.write_text(json.dumps(report,indent=2))
    print('DISTRESS_ACTIONS_SAVED',[a.name for a in bpy.data.actions])

if __name__=='__main__':
    build();save()
    from export_hero1_actor import export
    export()
