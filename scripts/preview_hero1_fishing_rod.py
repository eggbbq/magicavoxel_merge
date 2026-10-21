"""Add a simple, action-filtered rod to the existing tool.R preview socket."""
import bpy
from mathutils import Matrix
import hero1_work_actions as work


def add():
    r=work.rig()
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    existing=bpy.data.objects.get('PREVIEW_FishingRod')
    if existing:return existing
    vertices=[];faces=[];indices=[]
    def box(x0,x1,y0,y1,z0,z1,material):
        i=len(vertices)
        vertices.extend([(x,y,z) for z in (z0,z1) for y in (y0,y1) for x in (x0,x1)])
        faces.extend([tuple(i+j for j in f) for f in ((0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5))])
        indices.extend([material]*6)
    box(-.012,.012,-.065,.13,-.012,.012,0)
    for y0,y1,w in ((.13,.45,.008),(.45,.78,.006),(.78,1.08,.004)):
        box(-w,w,y0,y1,-w,w,1)
    box(-.004,.004,1.08,1.12,-.004,.004,3)
    box(.009,.046,-.060,-.015,-.026,.026,2)
    box(.041,.066,-.043,-.031,-.006,.006,2)
    data=bpy.data.meshes.new('Preview fishing rod geometry');data.from_pydata(vertices,[],faces);data.update()
    obj=bpy.data.objects.new('PREVIEW_FishingRod',data);bpy.context.scene.collection.objects.link(obj)
    for name,color in [('Grip',(.24,.095,.028,1)),('Shaft',(.055,.09,.11,1)),
                       ('Reel',(.32,.38,.40,1)),('Tip',(.72,.12,.045,1))]:
        mat=bpy.data.materials.get('PREVIEW_FishingRod_'+name) or bpy.data.materials.new('PREVIEW_FishingRod_'+name)
        mat.use_nodes=True;mat.diffuse_color=color
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value=color;shader.inputs['Roughness'].default_value=.65
        obj.data.materials.append(mat)
    for p,i in zip(data.polygons,indices):p.material_index=i
    obj.parent=r;obj.parent_type='BONE';obj.parent_bone='tool.R'
    obj.matrix_parent_inverse=Matrix.Identity(4)
    obj.location=(0,-r.data.bones['tool.R'].length,0)
    obj['preview_only']=True;obj['preview_actions']='fishing,fishing_pull'
    obj['remove_after_user_approval']=True;obj['attachment_socket']='tool.R'
    bpy.context.view_layer.update()
    return obj


if __name__=='__main__':add()
