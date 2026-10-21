"""Export only the animated hero and attachment bones; retain authoring preview props."""
from pathlib import Path
import bpy

def export():
    root=Path(bpy.data.filepath).parent
    r=bpy.data.objects['Hero'];mesh=bpy.data.objects['HeroMesh']
    original_mode=r.mode;selected=list(bpy.context.selected_objects)
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    r.select_set(True);mesh.select_set(True);bpy.context.view_layer.objects.active=r
    try:
        bpy.ops.export_scene.gltf(filepath=str(root/'hero1_animated.glb'),export_format='GLB',use_selection=True,
            export_animations=True,export_animation_mode='ACTIONS',export_frame_range=False,
            export_anim_slide_to_zero=True,export_force_sampling=True,export_frame_step=1,
            export_optimize_animation_size=True,export_reset_pose_bones=True,export_skins=True,
            export_def_bones=False,export_leaf_bone=False,export_armature_object_remove=True,
            export_rest_position_armature=True,export_morph=False,export_morph_animation=False,
            export_cameras=False,export_lights=False,export_extras=False,export_yup=True,
            export_materials='EXPORT',export_image_format='AUTO',export_draco_mesh_compression_enable=False,
            export_meshopt_compression_enable=False)
    finally:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:obj.select_set(True)
        bpy.context.view_layer.objects.active=r
        if original_mode=='POSE':bpy.ops.object.mode_set(mode='POSE')
    print('ACTOR_EXPORT_COMPLETE',len(r.data.bones),[a.name for a in bpy.data.actions])

if __name__=='__main__':export()
