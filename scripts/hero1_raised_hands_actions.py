"""Two distinct raised-hand states, using sparse poses and flexion-only elbows."""
from pathlib import Path
import json
import math
import bpy
from mathutils import Matrix, Vector
import hero1_work_actions as work

ROOT=Path(__file__).resolve().parent.parent
NAMES=('hands_over_head','hold_up')
SPECS={'hands_over_head':(144,True,[i/8 for i in range(9)]),
       'hold_up':(120,True,[i/8 for i in range(9)])}

def palm_frame(side,fingers,normal):
    p=work.rig().pose.bones['hand.'+side]
    y=Vector(fingers).normalized();z=Vector(normal).normalized();x=y.cross(z).normalized()
    matrix=Matrix((x,y,z)).transposed().to_4x4();matrix.translation=p.head.copy()
    p.matrix=matrix;bpy.context.view_layer.update()

def pose(name,t):
    work.reset();r=work.rig();wave=math.sin(math.tau*t)
    if name=='hands_over_head':
        knee=6;pitch=-2+.4*wave;head=1.5-.2*wave
        z=1.27+.004*wave
        right=Vector((.13,.32,z));left=Vector((.13,-.30,z))
        axis=Vector((0,0,1));normal=Vector((1,0,0));stretch=1.65
    else:
        knee=14+1.5*wave;pitch=-6+.8*wave;head=2-.4*wave
        z=1.25+.008*wave
        right=Vector((.34,.23,z));left=Vector((.34,-.21,z))
        axis=Vector((1,0,0));normal=Vector((0,0,1));stretch=1.85
    half=math.radians(knee*.5)
    height=.3*math.cos(half)+.02*math.sin(half)+.002
    hip=r.pose.bones['hips']
    hip.location=hip.bone.matrix_local.to_quaternion().inverted()@Vector((0,0,height-.3))
    work.rotate('body',pitch);work.rotate('head',head)
    for side in ('L','R'):
        work.rotate('thigh.'+side,-knee*.5);work.rotate('shin.'+side,knee)
    bpy.context.view_layer.update()
    errors=[work.solve_arm('R',right,axis,stretch,hinge=True),
            work.solve_arm('L',left,axis,stretch,hinge=True)]
    for side in ('L','R'):palm_frame(side,axis,normal)
    from relax_hero1_wrists import relax_current_pose
    relax_current_pose(r)
    values={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),
                    'scale':tuple(p.scale)} for p in r.pose.bones}
    return values,max(errors)

def build():
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero1_before_raised_hands.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    original=(work.SPECS,work.NEW_NAMES,work.pose)
    work.SPECS=SPECS;work.NEW_NAMES=NAMES;work.pose=pose
    try:report=work.build_actions()
    finally:work.SPECS,work.NEW_NAMES,work.pose=original
    descriptions={'hands_over_head':'双手举在头侧上方、肘部外展、掌心朝前的投降/防护姿势',
                  'hold_up':'双手向前上方托举，掌心朝上，轻微承重起伏'}
    for n in NAMES:
        bpy.data.actions[n]['description']=descriptions[n]
        bpy.data.actions[n]['palm_normal_world']='+X' if n=='hands_over_head' else '+Z'
    (ROOT/'hero1_raised_hands_report.json').write_text(json.dumps(report,indent=2))
    work.set_action('hands_over_head')
    return report

def save():
    s=bpy.context.scene;r=work.rig();work.set_action('hands_over_head')
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);bpy.context.view_layer.objects.active=r
    bpy.ops.object.mode_set(mode='POSE')
    import magicavoxel_hero_action_preview as preview
    preview.sync_scene(s,force=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'hero1_animated.blend'))
    p=ROOT/'hero1_animation_report.json';report=json.loads(p.read_text())
    report['clips']=json.loads(s['animation_clips']);p.write_text(json.dumps(report,indent=2))
    print('RAISED_HANDS_SAVED',[a.name for a in bpy.data.actions])

if __name__=='__main__':
    build();save()
    from export_hero1_actor import export
    export()
