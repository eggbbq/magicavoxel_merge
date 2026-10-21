"""Reduce the approved run speed to 80%, preserving every pose and key count.

Use a 48 fps authoring timebase: the 10-frame/24-fps run becomes 25 frames/48 fps.
All other clips double their frame coordinates and retain their original durations.
"""
from pathlib import Path
import json
import math
import bpy

scene=bpy.context.scene
rig=bpy.data.objects['Hero']
mesh=bpy.data.objects['HeroMesh']
out=Path(bpy.data.filepath).parent
assert scene.render.fps==24 and int(bpy.data.actions['run']['loop_frames'])==10,'Already retimed or unexpected source.'
if bpy.context.screen and bpy.context.screen.is_animation_playing:
    bpy.ops.screen.animation_cancel(restore_frame=False)
backup=out/'hero1_before_run_20percent_slower.blend'
if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)

def curves(a):
    return [c for l in a.layers for s in l.strips for b in s.channelbags for c in b.fcurves]

def points(a,t):
    rig.animation_data.action=a;rig.animation_data.action_slot=a.slots[0]
    f=1+float(a['loop_frames'])*t
    scene.frame_set(math.floor(f),subframe=f%1)
    bpy.context.view_layer.update()
    ev=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [v.co.copy() for v in ev.data.vertices]

phases=(0,.125,.25,.375,.5,.625,.75,.875,1)
before={(a.name,t):points(a,t) for a in bpy.data.actions for t in phases}
counts={a.name:sum(len(c.keyframe_points) for c in curves(a)) for a in bpy.data.actions}
durations={a.name:float(a['loop_frames'])/24 for a in bpy.data.actions}
for a in bpy.data.actions:
    factor=2.5 if a.name=='run' else 2.
    for c in curves(a):
        for k in c.keyframe_points:
            x,l,r=float(k.co.x),float(k.handle_left.x),float(k.handle_right.x)
            k.co.x=1+(x-1)*factor
            k.handle_left.x=1+(l-1)*factor
            k.handle_right.x=1+(r-1)*factor
        c.update()
    a['loop_frames']=round(float(a['loop_frames'])*factor)
    a.frame_start=1;a.frame_end=1+a['loop_frames']
scene.render.fps=48;scene.render.fps_base=1
maximum=0
for a in bpy.data.actions:
    assert counts[a.name]==sum(len(c.keyframe_points) for c in curves(a))
    expected=durations[a.name]*(1.25 if a.name=='run' else 1)
    assert abs(a['loop_frames']/48-expected)<1e-9
    for t in phases:
        maximum=max(maximum,max((p-q).length for p,q in zip(before[(a.name,t)],points(a,t))))
assert maximum<1e-6,maximum

stats=json.loads(scene.get('animation_clips','{}'))
for a in bpy.data.actions:
    if a.name in stats:stats[a.name].update(frames=int(a['loop_frames']),duration=a['loop_frames']/48)
scene['animation_clips']=json.dumps(stats)
rig['preview_notes']='48 fps; idle 1-144; walk 1-64; walk_fast 1-40; run 1-25; jump 1-96; sleep 1-192. Same poses and key count; run at 80% speed.'
a=bpy.data.actions['run'];rig.animation_data.action=a;rig.animation_data.action_slot=a.slots[0]
scene.frame_start=scene.frame_preview_start=1
scene.frame_end=scene.frame_preview_end=25
scene.use_preview_range=True;scene.frame_set(1)
if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT');rig.select_set(True)
bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='POSE')
bpy.ops.wm.save_as_mainfile(filepath=str(out/'hero1_animated.blend'))
bpy.ops.object.mode_set(mode='OBJECT');mesh.select_set(True)
bpy.ops.export_scene.gltf(
    filepath=str(out/'hero1_animated.glb'),export_format='GLB',use_selection=True,
    export_animations=True,export_animation_mode='ACTIONS',export_frame_range=False,
    export_anim_slide_to_zero=True,export_force_sampling=True,export_frame_step=1,
    export_optimize_animation_size=True,export_reset_pose_bones=True,
    export_skins=True,export_def_bones=False,export_leaf_bone=False,
    export_armature_object_remove=True,export_rest_position_armature=True,
    export_morph=False,export_morph_animation=False,export_cameras=False,
    export_lights=False,export_extras=False,export_yup=True,export_materials='EXPORT',
    export_image_format='AUTO',export_draco_mesh_compression_enable=False,
    export_meshopt_compression_enable=False)
mesh.select_set(False);bpy.ops.object.mode_set(mode='POSE')
rig.animation_data.action=a;rig.animation_data.action_slot=a.slots[0];scene.frame_set(1)
report={'run_speed_multiplier':.8,'authoring_fps':48,'key_counts':counts,
        'normalized_pose_max_error':maximum,'durations_seconds':{a.name:a['loop_frames']/48 for a in bpy.data.actions}}
(out/'hero1_run_retime_validation.json').write_text(json.dumps(report,indent=2))
p=out/'hero1_animation_report.json'
if p.exists():
    full=json.loads(p.read_text());full['clips']=stats;full['authoring_fps']=48
    p.write_text(json.dumps(full,indent=2))
print('RETIME_PASSED',json.dumps(report))
