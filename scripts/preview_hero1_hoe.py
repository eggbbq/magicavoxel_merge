"""Add a persistent tool.R socket and a removable hoe for till-action review.

Run in the current animated hero scene. This saves the authoring .blend only;
the character-only game GLB remains unchanged until the preview is approved.
"""
from pathlib import Path
import json
import bpy
from mathutils import Matrix, Vector

ROOT=Path(__file__).resolve().parent.parent
SOCKET='tool.R'
PROP='PREVIEW_Hoe'

def curves(action):
    return [c for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags for c in bag.fcurves]

def snapshot():
    return {a.name:[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right))
                    for k in c.keyframe_points]) for c in curves(a)] for a in bpy.data.actions}

def make_material(name,color,metallic=0):
    mat=bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes=True
    node=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    node.inputs['Base Color'].default_value=(*color,1)
    node.inputs['Roughness'].default_value=.65
    node.inputs['Metallic'].default_value=metallic
    mat.diffuse_color=(*color,1)
    return mat

def main():
    scene=bpy.context.scene;r=bpy.data.objects['Hero'];hero=bpy.data.objects['HeroMesh']
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    backup=ROOT/'hero1_before_tool_socket.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    old_actions=snapshot()
    references={}
    for a in bpy.data.actions:
        r.animation_data.action=a;r.animation_data.action_slot=a.slots[0]
        scene.frame_set(1+int(a['loop_frames'])//2);bpy.context.view_layer.update()
        references[a.name]=[v.co.copy() for v in hero.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);bpy.context.view_layer.objects.active=r
    if SOCKET not in r.data.bones:
        bpy.ops.object.mode_set(mode='EDIT')
        hand=r.data.edit_bones['hand.R'];b=r.data.edit_bones.new(SOCKET)
        b.head=(hand.head+hand.tail)*.5
        b.tail=b.head+(hand.tail-hand.head).normalized()*.04
        b.roll=hand.roll;b.parent=hand;b.use_connect=False;b.use_deform=False
        bpy.ops.object.mode_set(mode='OBJECT')
    b=r.data.bones[SOCKET]
    b['purpose']='Persistent right-hand tool attachment socket; keep when removing preview props.'
    b['tool_axis']='+Y shaft, -X blade direction, Z blade width; origin at palm centre'
    p=r.pose.bones[SOCKET];p.rotation_mode='QUATERNION'
    p.location=(0,0,0);p.rotation_quaternion=(1,0,0,0);p.scale=(1,1,1)
    if PROP in bpy.data.objects:bpy.data.objects.remove(bpy.data.objects[PROP],do_unlink=True)
    verts=[];faces=[];material_indices=[]
    def box(x0,x1,y0,y1,z0,z1,material):
        i=len(verts)
        verts.extend([(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                      (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)])
        faces.extend([tuple(i+j for j in f) for f in
            ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))])
        material_indices.extend([material]*6)
    box(-.013,.013,-.16,.60,-.013,.013,0)
    box(-.019,.019,.557,.604,-.022,.022,1)
    box(-.120,.015,.576,.604,-.095,.095,1)
    data=bpy.data.meshes.new('HoePreviewGeometry');data.from_pydata(verts,[],faces);data.update()
    prop=bpy.data.objects.new(PROP,data);scene.collection.objects.link(prop)
    data.materials.append(make_material('PreviewHoe_Wood',(.28,.115,.035)))
    data.materials.append(make_material('PreviewHoe_Iron',(.24,.29,.32),.5))
    for polygon,index in zip(data.polygons,material_indices):polygon.material_index=index
    prop.parent=r;prop.parent_type='BONE';prop.parent_bone=SOCKET
    prop.matrix_parent_inverse=Matrix.Identity(4)
    # Blender bone parenting uses the tail. Compensate so the mesh origin is the socket head.
    prop.location=(0,-b.length,0)
    prop['preview_only']=True
    prop['remove_after_user_approval']=True
    prop['attachment_socket']=SOCKET
    assert snapshot()==old_actions,'Existing action data changed'
    max_error=0
    for a in bpy.data.actions:
        r.animation_data.action=a;r.animation_data.action_slot=a.slots[0]
        scene.frame_set(1+int(a['loop_frames'])//2);bpy.context.view_layer.update()
        current=hero.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices
        max_error=max(max_error,max((v.co-ref).length for v,ref in zip(current,references[a.name])))
    assert max_error<1e-6,('Socket changed character geometry',max_error)
    a=bpy.data.actions['till'];r.animation_data.action=a;r.animation_data.action_slot=a.slots[0]
    ground_min=float('inf');grip_error=0
    for frame in range(1,74):
        scene.frame_set(frame);bpy.context.view_layer.update()
        grip=r.matrix_world@((r.pose.bones['hand.R'].head+r.pose.bones['hand.R'].tail)*.5)
        grip_error=max(grip_error,(prop.matrix_world.translation-grip).length)
        ground_min=min(ground_min,min((prop.matrix_world@v.co).z for v in data.vertices))
    assert grip_error<1e-5,('Tool detached from grip',grip_error)
    assert ground_min>-.01,('Hoe penetrates floor',ground_min)
    scene.frame_start=scene.frame_preview_start=1;scene.frame_end=scene.frame_preview_end=72
    scene.use_preview_range=True;scene.frame_set(1);scene.sync_mode='FRAME_DROP'
    bpy.ops.object.mode_set(mode='POSE')
    r.show_in_front=False
    r['tool_socket']=SOCKET
    scene['tool_preview_notes']='Temporary hoe for till review. Retain until user approval; keep tool.R after prop removal.'
    try:
        import magicavoxel_hero_action_preview as preview
        preview.sync_scene(scene,force=True)
    except ImportError:pass
    report={'socket':SOCKET,'parent_bone':'hand.R','deforming':False,'bone_count':len(r.data.bones),
            'preview_prop':PROP,'grip_position_error':grip_error,'min_tool_z':ground_min,
            'existing_action_keys_unchanged':True,'existing_midpose_vertex_error':max_error}
    (ROOT/'hero1_tool_preview_validation.json').write_text(json.dumps(report,indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'hero1_animated.blend'))
    print('TOOL_PREVIEW_READY',json.dumps(report))

if __name__=='__main__':main()
