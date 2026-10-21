"""Reuse the approved hero1 rig/actions on a matching 12-part voxel character.

Blender -b --python scripts/retarget_voxel_character.py -- hero2.blend
The source and animation template are read-only; outputs are <source>_animated.
This project's source characters face -Y; the approved skeleton faces +X.
"""
from pathlib import Path
import sys,json,math,hashlib
import bpy
from mathutils import Matrix,Vector
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from sew_hero1_joints import sew_limb_joints


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def curves(a):return [c for l in a.layers for s in l.strips for b in s.channelbags for c in b.fcurves]

def action_fingerprint(a):
    payload=[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right),k.interpolation) for k in c.keyframe_points]) for c in curves(a)]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

def bounds(obj,transform=None):
    transform=transform or obj.matrix_world
    vs=[transform@v.co for v in obj.data.vertices]
    return Vector([min(v[i] for v in vs) for i in range(3)]),Vector([max(v[i] for v in vs) for i in range(3)])


def export(path,animated):
    r=bpy.data.objects['Hero'];m=bpy.data.objects['HeroMesh']
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);m.select_set(True);bpy.context.view_layer.objects.active=r
    bpy.ops.export_scene.gltf(filepath=str(path),export_format='GLB',use_selection=True,
        export_animations=animated,export_animation_mode='ACTIONS',export_frame_range=False,
        export_anim_slide_to_zero=True,export_force_sampling=True,export_frame_step=1,
        export_optimize_animation_size=True,export_reset_pose_bones=True,export_skins=True,
        export_def_bones=False,export_leaf_bone=False,export_armature_object_remove=True,
        export_rest_position_armature=True,export_morph=False,export_morph_animation=False,
        export_cameras=False,export_lights=False,export_extras=False,export_yup=True,
        export_materials='EXPORT',export_image_format='AUTO',export_draco_mesh_compression_enable=False,
        export_meshopt_compression_enable=False)


def build(source,template=None,source_forward='-Y'):
    source=Path(source).resolve()
    out=source.with_name(source.stem+'_animated.blend');outglb=out.with_suffix('.glb')
    registry=ROOT/'character_animation_template.json'
    standard=ROOT/json.loads(registry.read_text())['template'] if registry.exists() else ROOT/'hero1_animated.blend'
    template=Path(template or (standard if standard.exists() and standard!=out else ROOT/'hero1_animated.blend')).resolve()
    original_hashes={str(p):digest(p) for p in (source,template)}
    bpy.ops.wm.open_mainfile(filepath=str(template))
    scene=bpy.context.scene;r=bpy.data.objects['Hero'];oldmesh=bpy.data.objects['HeroMesh']
    template_hashes={a.name:action_fingerprint(a) for a in bpy.data.actions}
    template_bones={b.name:[list(row) for row in b.matrix_local] for b in r.data.bones}
    reference_weights=[]
    original_to_reference={}
    for v in oldmesh.data.vertices:
        weights={oldmesh.vertex_groups[g.group].name:g.weight for g in v.groups}
        if any(n.startswith(('upper_arm.','forearm.','hand.','thigh.','shin.')) for n in weights):
            original_to_reference[v.index]=len(reference_weights)
            reference_weights.append((v.co.copy(),weights))
    oldmesh.data.calc_loop_triangles()
    triangles=[tuple(original_to_reference[i] for i in t.vertices) for t in oldmesh.data.loop_triangles if all(i in original_to_reference for i in t.vertices)]
    surface=BVHTree.FromPolygons([p for p,w in reference_weights],triangles,all_triangles=True)
    bodylow,bodyhigh=bounds(oldmesh)
    body_vertices=[v.co for v in oldmesh.data.vertices if any(oldmesh.vertex_groups[g.group].name=='body' for g in v.groups)]
    target_body_centre=Vector([(min(v[i] for v in body_vertices)+max(v[i] for v in body_vertices))/2 for i in range(3)])
    for obj in list(bpy.data.objects):
        if obj!=r:bpy.data.objects.remove(obj,do_unlink=True)
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    with bpy.data.libraries.load(str(source),link=False) as (src,dst):dst.objects=list(src.objects)
    for obj in dst.objects:scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    parts=[o for o in dst.objects if o.type=='MESH']
    assert len(parts)==12 and all(not o.modifiers for o in parts),'Expected twelve unbound body parts'
    body=next(o for o in parts if o.name=='body');head=next(o for o in parts if o.name=='head')
    angle={'+X':0.,'-X':180.,'+Y':-90.,'-Y':90.}[source_forward]
    rotation=Matrix.Rotation(math.radians(angle),4,'Z')
    low,high=bounds(body,rotation@body.matrix_world)
    offset=target_body_centre-(low+high)*.5
    transform=Matrix.Translation(offset)@rotation
    mapping={body:'body',head:'head'};adjustments={}
    for kind,prefix in [('upper-arm','upper_arm'),('forearm','forearm'),('palm','hand'),('thigh','thigh'),('calf','shin')]:
        pair=sorted([o for o in parts if o.name.split('.')[0]==kind],key=lambda o:bounds(o,transform@o.matrix_world)[0].y)
        assert len(pair)==2,(kind,len(pair))
        for side,o in zip(('L','R'),pair):mapping[o]=prefix+'.'+side
    # Bake each instance's world transform and align the limb axes to the shared rig.
    source_vertices=sum(len(o.data.vertices) for o in parts)
    for obj in parts:
        world=transform@obj.matrix_world
        obj.data=obj.data.copy();obj.data.transform(world);obj.parent=None;obj.matrix_world=Matrix.Identity(4)
        name=mapping[obj]
        if name not in ('body','head'):
            lo,hi=bounds(obj);centre=(lo+hi)*.5;b=r.data.bones[name]
            shift=Vector((b.head_local.x-centre.x,b.head_local.y-centre.y,0))
            assert abs(hi.z-b.head_local.z)<1e-5 and abs(lo.z-b.tail_local.z)<1e-5,(name,'Joint heights do not match')
            assert max(abs((hi-lo)[i]-.04) for i in (0,1))<1e-5,(name,'Limb cross-section differs')
            obj.data.transform(Matrix.Translation(shift));adjustments[obj.name]=list(shift)
        obj.vertex_groups.clear();obj.vertex_groups.new(name=name).add(list(range(len(obj.data.vertices))),1.,'REPLACE')
    for obj in list(bpy.data.objects):
        if obj not in parts and obj!=r:bpy.data.objects.remove(obj,do_unlink=True)
    bpy.ops.object.select_all(action='DESELECT')
    for o in parts:o.select_set(True)
    bpy.context.view_layer.objects.active=body;bpy.ops.object.join();mesh=bpy.context.object
    mesh.name='HeroMesh';mesh.data.name=source.stem+'_skinned_geometry';mesh.animation_data_clear()
    for uv in list(mesh.data.uv_layers):
        if not uv.active_render:mesh.data.uv_layers.remove(uv)
    joints=sew_limb_joints(mesh)
    # Verify and copy the template's actual blended weights at all limb vertices.
    kd=KDTree(len(reference_weights))
    for i,(p,weights) in enumerate(reference_weights):kd.insert(p,i)
    kd.balance();maxdistance=maxweight=0.;transferred=0
    for v in mesh.data.vertices:
        if any(mesh.vertex_groups[g.group].name in ('body','head') for g in v.groups):continue
        _,index,distance=kd.find(v.co)
        if distance<1e-5:target=reference_weights[index][1]
        else:
            point,normal,face,distance=surface.find_nearest(v.co)
            assert distance<1e-5,('No corresponding template limb surface',tuple(v.co),distance)
            tri=triangles[face];a,b,c=[reference_weights[i][0] for i in tri]
            e0,e1,d=b-a,c-a,point-a
            d00,d01,d11=e0.dot(e0),e0.dot(e1),e1.dot(e1)
            denominator=d00*d11-d01*d01
            wb=(d11*d.dot(e0)-d01*d.dot(e1))/denominator
            wc=(d00*d.dot(e1)-d01*d.dot(e0))/denominator
            target={}
            for i,blend in zip(tri,(1-wb-wc,wb,wc)):
                for n,w in reference_weights[i][1].items():target[n]=target.get(n,0)+max(0.,blend)*w
            target={n:w for n,w in target.items() if w>1e-6};total=sum(target.values());target={n:w/total for n,w in target.items()}
        maxdistance=max(maxdistance,distance)
        actual={mesh.vertex_groups[g.group].name:g.weight for g in v.groups}
        maxweight=max(maxweight,max(abs(actual.get(n,0)-target.get(n,0)) for n in set(actual)|set(target)))
        for g in list(v.groups):mesh.vertex_groups[g.group].remove([v.index])
        for name,weight in target.items():mesh.vertex_groups[name].add([v.index],weight,'REPLACE')
        transferred+=1
    mod=mesh.modifiers.new('Shared character skin','ARMATURE');mod.object=r
    for image in bpy.data.images:
        if image.source=='FILE' and image.has_data and not image.packed_file:image.pack()
    # Imported material remains the hero2 palette; template props never enter this file.
    assert len(mesh.data.materials)==1
    assert all(action_fingerprint(bpy.data.actions[n])==h for n,h in template_hashes.items())
    assert {b.name:[list(row) for row in b.matrix_local] for b in r.data.bones}==template_bones
    scene['hero_action_preview_auto']=True;scene['hero_action_preview_rig']='Hero'
    scene['shared_animation_source']=template.name;scene['character_source']=source.name
    scene.render.fps=48;scene.render.fps_base=1
    import hero1_work_actions as work,hero_action_preview as preview
    work.set_action('idle',1);preview.sync_scene(scene,force=True)
    for obj in scene.objects:obj.select_set(False)
    r.select_set(True);bpy.context.view_layer.objects.active=r
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    export(outglb,True)
    # Also provide the compact skinned character for shared-animation engine use.
    export(source.with_name(source.stem+'_skinned.glb'),False)
    work.set_action('walk',1);work.set_action('idle',1);preview.sync_scene(scene,force=True)
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);bpy.context.view_layer.objects.active=r;bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    assert all(digest(p)==h for p,h in original_hashes.items()),'Input file modified'
    report={'source':str(source),'template':str(template),'outputs':[str(out),str(outglb),str(source.with_name(source.stem+'_skinned.glb'))],
            'source_unchanged':True,'rotation_z_degrees':angle,'source_forward':source_forward,'global_translation':list(offset),'limb_alignments_m':adjustments,
            'source_vertices':source_vertices,'skinned_vertices':len(mesh.data.vertices),'bones':len(r.data.bones),'drivers':len(r.animation_data.drivers),
            'animations':{a.name:{'length_frames':int(a['loop_frames']),'looping':bool(a['looping']),'unchanged':action_fingerprint(a)==template_hashes[a.name]} for a in bpy.data.actions},
            'welded_limbs':joints,'transferred_limb_vertices':transferred,'max_limb_match_error_m':maxdistance,'max_generated_weight_difference':maxweight,'material':mesh.data.materials[0].name}
    source.with_name(source.stem+'_transfer_report.json').write_text(json.dumps(report,indent=2))
    print('CHARACTER_TRANSFER_COMPLETE',json.dumps(report))
    return report


if __name__=='__main__':
    import argparse
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else ['hero2.blend']
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('--template');parser.add_argument('--forward',choices=['+X','-X','+Y','-Y'],default='-Y')
    options=parser.parse_args(args)
    build(ROOT/options.source,ROOT/options.template if options.template else None,options.forward)
