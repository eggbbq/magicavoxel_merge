"""Sparse sitting, eating, clutching-the-abdomen and crouching Actions.

Reuse the approved rig and the anatomical arm solver. Keep the pending hoe
preview in the .blend; action-aware preview visibility hides it for these clips.
"""
import json
import math
from pathlib import Path
import bpy
from mathutils import Vector
import hero1_work_actions as work

ROOT=Path(__file__).resolve().parent.parent
NAMES=('sitting','eating','clutch','crouching')
SPECS={'sitting':(144,True,[i/8 for i in range(9)]),
       'eating':(120,True,[0,.15,.30,.40,.50,.60,.75,.88,1]),
       'clutch':(144,True,[i/8 for i in range(9)]),
       'crouching':(96,True,[i/8 for i in range(9)])}

def body_point(rest):
    b=work.rig().pose.bones['body']
    return b.matrix@b.bone.matrix_local.inverted()@Vector(rest)

def head_point(rest):
    b=work.rig().pose.bones['head']
    return b.matrix@b.bone.matrix_local.inverted()@Vector(rest)

def pose(name,t):
    work.reset();r=work.rig();wave=math.sin(math.tau*t)
    if name=='sitting':
        thigh=-84+.8*wave;knee=86-.8*wave;pitch=3+.8*wave;head=-2-.5*wave
    elif name=='crouching':
        thigh=-66+1*wave;knee=130-1.5*wave;pitch=24+1.5*wave;head=-16-.8*wave
    elif name=='clutch':
        thigh=-9;knee=18;pitch=17+2*wave;head=7+.8*wave
    else:
        thigh=-4;knee=8
        pitch=2+.6*wave;head=-1+1.3*math.sin(math.tau*t*2)
    # A chair-height seated state and a planted squat use the actual leg lengths.
    a,b=math.radians(thigh),math.radians(thigh+knee)
    height=.16*math.cos(a)+.14*math.cos(b)+.02*abs(math.sin(b))+.002
    p=r.pose.bones['hips'];p.location=p.bone.matrix_local.to_quaternion().inverted()@Vector((0,0,height-.3))
    work.rotate('body',pitch);work.rotate('head',head)
    for side in ('L','R'):
        work.rotate('thigh.'+side,thigh);work.rotate('shin.'+side,knee)
    bpy.context.view_layer.update()
    if name=='sitting':
        right=Vector((.21,.20,.405+.002*wave));left=Vector((.21,-.18,.405+.002*wave))
        right_axis=left_axis=Vector((.8,0,-.6));stretch=1
    elif name=='crouching':
        right=Vector((.43,.20,.55+.003*wave));left=Vector((.43,-.18,.55+.003*wave))
        right_axis=left_axis=Vector((1,0,-.15));stretch=1
    elif name=='clutch':
        right=body_point((.215,.055,.60));left=body_point((.225,-.025,.625))
        right_axis=left_axis=Vector((-1,0,0));stretch=1.12
    else:
        # Right hand moves from the imaginary food held by the left hand to the mouth.
        values=work.interpolate([(0,.245,.01,.755),(.15,.255,.01,.80),
            (.30,.218,.015,.915),(.40,.213,.015,.92),(.50,.235,.015,.89),
            (.60,.25,.01,.84),(.75,.245,.01,.77),(.88,.245,.01,.755),
            (1,.245,.01,.755)],t)
        right=Vector(values);left=Vector((.24,-.065,.735+.002*wave))
        right_axis=Vector((-1,0,.08));left_axis=Vector((.6,0,.8));stretch=1.08
    errors=[work.solve_arm('R',right,right_axis,stretch,hinge=True),
            work.solve_arm('L',left,left_axis,stretch,hinge=True)]
    from relax_hero1_wrists import relax_current_pose
    relax_current_pose(r)
    result={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),
                    'scale':tuple(p.scale)} for p in r.pose.bones}
    return result,max(errors)

def build(names=NAMES):
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero1_before_life_actions.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    original=(work.SPECS,work.NEW_NAMES,work.pose)
    work.SPECS={n:SPECS[n] for n in names};work.NEW_NAMES=tuple(names);work.pose=pose
    try:report=work.build_actions()
    finally:work.SPECS,work.NEW_NAMES,work.pose=original
    desc={'sitting':'坐姿保持与轻微呼吸，可在游戏中配合座椅',
          'eating':'左手托食，右手取食、送入口边、回收',
          'clutch':'双手捂腹，轻微蜷缩和呼吸',
          'crouching':'蹲姿保持、前倾平衡和轻微呼吸'}
    for n in names:bpy.data.actions[n]['description']=desc[n]
    prop=bpy.data.objects.get('PREVIEW_Hoe')
    if prop:
        prop['preview_action']='till';prop.hide_viewport=True;prop.hide_render=True
    (ROOT/'hero1_life_actions_report.json').write_text(json.dumps(report,indent=2))
    work.set_action(names[0])
    return report

def save():
    s=bpy.context.scene;r=work.rig()
    work.set_action('sitting')
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);bpy.context.view_layer.objects.active=r
    bpy.ops.object.mode_set(mode='POSE')
    import magicavoxel_hero_action_preview as preview
    preview.sync_scene(s,force=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'hero1_animated.blend'))
    p=ROOT/'hero1_animation_report.json';report=json.loads(p.read_text())
    report['clips']=json.loads(s['animation_clips']);p.write_text(json.dumps(report,indent=2))
    print('LIFE_ACTIONS_SAVED',[a.name for a in bpy.data.actions])

if __name__=='__main__':
    build();save()
    from export_hero1_actor import export
    export()
