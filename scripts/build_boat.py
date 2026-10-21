"""Build the reference-inspired voxel rowing boat. Run with Blender Python."""
import bpy
import math
import random
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
random.seed(31)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for c in list(bpy.data.collections):
    if c.name != 'Collection':
        bpy.data.collections.remove(c)
default = bpy.data.collections.get('Collection')
default.name = 'BOAT | Voxel wooden rowboat'

def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c

hull_col = collection('01 | Stepped hull and gunwales')
interior_col = collection('02 | Floorboards and three seats')
ends_col = collection('03 | End posts and ivory rope collars')
cargo_col = collection('04 | Crate, flask and provision sack')
oar_col = collection('05 | Oar and wooden rowlock')
studio_col = collection('STUDIO | Camera, lights and backdrop')

def material(name, color, roughness=1):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return m

wood = [material('Oak | warm block %02d' % i, c) for i, c in enumerate([
    (.34,.155,.062), (.40,.19,.075), (.44,.215,.09), (.48,.24,.10),
    (.38,.17,.064), (.51,.265,.115), (.43,.195,.075), (.46,.225,.09),
    (.36,.16,.058), (.54,.28,.12)])]
rail = [material('End grain | honey %02d' % i, c) for i,c in enumerate([
    (.68,.36,.15), (.74,.405,.185), (.79,.45,.215), (.70,.37,.16),
    (.82,.48,.235), (.73,.39,.175)])]
deck = [material('Floorboards | oak %02d' % i, c) for i,c in enumerate([
    (.40,.205,.095),(.45,.235,.11),(.48,.26,.125),(.43,.215,.095),(.50,.27,.13)])]
iron = [material('Forged iron | %02d'%i,c) for i,c in enumerate([
    (.22,.225,.21),(.30,.31,.285),(.39,.40,.36)])]
patch = [material('Weathered grey repair | %02d'%i,c) for i,c in enumerate([
    (.38,.38,.32),(.45,.45,.39),(.48,.47,.40)])]
rope = [material('Ivory rope | %02d'%i,c) for i,c in enumerate([
    (.76,.68,.49),(.88,.80,.62),(.82,.74,.56)])]
olive = [material('Olive canvas | %02d'%i,c) for i,c in enumerate([
    (.28,.31,.16),(.36,.39,.22),(.41,.43,.26),(.45,.47,.29),(.32,.35,.18)])]
pewter = [material('Flask | %02d'%i,c,.7) for i,c in enumerate([
    (.44,.46,.43),(.59,.61,.58),(.67,.69,.66)])]

for palette, multiplier in ((wood,(.70,.55,.40)),(rail,(1.1,.68,.32)),(deck,(.73,.60,.45)),(olive,(.65,.69,.55))):
    for mat in palette:
        bsdf=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        rgba=bsdf.inputs['Base Color'].default_value
        color=tuple(rgba[i]*multiplier[i] for i in range(3))
        bsdf.inputs['Base Color'].default_value=(*color,1)
        mat.diffuse_color=(*color,1)

def box(name, center, size, mats, col, top=None):
    x,y,z = (v/2 for v in size)
    verts = [(-x,-y,-z),(-x,-y,z),(-x,y,-z),(-x,y,z),
             (x,-y,-z),(x,-y,z),(x,y,-z),(x,y,z)]
    faces = [(0,4,6,2),(0,1,5,4),(4,5,7,6),(6,7,3,2),(2,3,1,0),(1,3,7,5)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts,[],faces)
    mesh.materials.append(mats)
    if top:
        mesh.materials.append(top)
        mesh.polygons[5].material_index=1
    obj = bpy.data.objects.new(name,mesh)
    col.objects.link(obj)
    obj.location = center
    return obj

U=.24
def width(ix):
    a=abs(ix)
    return 6 if a<=18 else {19:6,20:5,21:5,22:4,23:3,24:2,25:1}[a]

# Voxel occupancy makes the hull a continuous hollow shell with a stair-step bilge.
cells={}
for ix in range(-25,26):
    w=width(ix)
    height = 6 + int(abs(ix)>=20) + int(abs(ix)>=24)
    for iz in range(height):
        half=max(0,w-3) if iz==0 else max(0,w-2) if iz==1 else max(0,w-1) if iz<=3 else w
        prev=max(0,w-3) if iz==1 else max(0,w-2) if iz==2 else max(0,w-1) if iz==4 else half
        for iy in range(-half,half+1):
            if iz==0 or abs(iy)>=min(half,prev):
                cells[(ix,iy,iz)] = True

# Fill the inside of diagonal stair corners so neighboring blocks share a face.
# This avoids four surface faces meeting along a single zero-thickness edge.
from itertools import combinations, product
while True:
    additions=set()
    for cell in list(cells):
        for a,b in combinations(range(3),2):
            for sa,sb in product((-1,1),repeat=2):
                diagonal=list(cell); diagonal[a]+=sa; diagonal[b]+=sb
                if tuple(diagonal) not in cells: continue
                mid_a=list(cell); mid_a[a]+=sa
                mid_b=list(cell); mid_b[b]+=sb
                ca,cb=tuple(mid_a),tuple(mid_b)
                if ca not in cells and cb not in cells:
                    additions.add(min((ca,cb),key=lambda c:(abs(c[1]),c[2],abs(c[0]))))
    if not additions: break
    for cell in additions: cells[cell]=True

all_mats=wood+rail+patch
verts=[]; faces=[]; indices=[]
directions=[((1,0,0),[(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),
            ((-1,0,0),[(0,1,0),(0,0,0),(0,0,1),(0,1,1)]),
            ((0,1,0),[(1,1,0),(0,1,0),(0,1,1),(1,1,1)]),
            ((0,-1,0),[(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
            ((0,0,1),[(0,0,1),(1,0,1),(1,1,1),(0,1,1)]),
            ((0,0,-1),[(0,1,0),(1,1,0),(1,0,0),(0,0,0)])]
for (ix,iy,iz) in cells:
    is_patch=iy<0 and ix in (-15,-14,10,11) and iz in (2,3,4)
    base=random.randrange(len(wood))
    for (dx,dy,dz),corners in directions:
        if (ix+dx,iy+dy,iz+dz) in cells: continue
        offset=len(verts)
        verts.extend([((ix+a-.5)*U,(iy+b-.5)*U,(iz+c)*U) for a,b,c in corners])
        faces.append(tuple(offset+j for j in range(4)))
        if is_patch: idx=len(wood)+len(rail)+random.randrange(len(patch))
        elif dz==1 and iz>=5: idx=len(wood)+random.randrange(len(rail))
        else: idx=base
        indices.append(idx)
mesh=bpy.data.meshes.new('Exposed voxel hull surfaces')
mesh.from_pydata(verts,[],faces)
for m in all_mats: mesh.materials.append(m)
for p,i in zip(mesh.polygons,indices): p.material_index=i
obj=bpy.data.objects.new('Hull | hollow stepped oak shell',mesh)
hull_col.objects.link(obj)
# Weld shared vertices while retaining all of the visible square color panels.
import bmesh
bm=bmesh.new(); bm.from_mesh(mesh)
bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001)
bm.to_mesh(mesh); bm.free()

# Separate the rare point-only contacts between closed voxel surface fans.
# The positions stay identical, but every vertex now belongs to one surface.
coords=[tuple(v.co) for v in mesh.vertices]
polys=[list(p.vertices) for p in mesh.polygons]
mat_ids=[p.material_index for p in mesh.polygons]
incident=[set() for _ in coords]
edge_faces={}
for fi,poly in enumerate(polys):
    for v in poly: incident[v].add(fi)
    for a,b in zip(poly,poly[1:]+poly[:1]):
        edge_faces.setdefault(tuple(sorted((a,b))),[]).append(fi)
neighbors=[{} for _ in coords]
for (a,b),fs in edge_faces.items():
    for v in (a,b):
        for fi in fs: neighbors[v].setdefault(fi,set()).update(set(fs)-{fi})
for vi,remaining in enumerate(incident):
    first=True
    while remaining:
        seed=next(iter(remaining)); group={seed}; todo=[seed]
        remaining.remove(seed)
        while todo:
            for other in neighbors[vi].get(todo.pop(),set()) & remaining:
                remaining.remove(other); group.add(other); todo.append(other)
        if first:
            first=False
        else:
            new_vi=len(coords); coords.append(coords[vi])
            for fi in group: polys[fi]=[new_vi if v==vi else v for v in polys[fi]]
mesh.clear_geometry(); mesh.from_pydata(coords,[],polys)
for p,i in zip(mesh.polygons,mat_ids): p.material_index=i
bm=bmesh.new(); bm.from_mesh(mesh)
assert all(e.is_manifold for e in bm.edges), 'Hull contains a nonmanifold edge'
assert all(v.is_manifold for v in bm.verts), 'Hull contains a point-only surface contact'
bm.free()
print('HULL_VALIDATED: closed manifold surface')

# Narrow staggered planks cover the flat inner sole.
for row in range(-3,4):
    for j in range(-8,9):
        x=j*.70 + (.35 if row%2 else 0)
        if abs(x)>5.1: continue
        ix=round(x/U)
        if abs(row)>max(0,width(ix)-3): continue
        box('Sole | row %02d plank %02d'%(row+4,j+9),(x,row*U,.28),(.692,.234,.09),random.choice(deck),interior_col)

for seat_n,x in enumerate((-1.95,.68,3.30),1):
    for k in range(3):
        # Three full-width boards form each bench; each end is supported inside the hull.
        for across in range(-5,6):
            box('Seat %d | board %d block %02d'%(seat_n,k+1,across+6),
                (x+(k-1)*U,across*U,1.10),(U-.004,U-.002,.23),
                random.choice(wood),interior_col,random.choice(rail))
    for side in (-1,1):
        box('Seat %d | support %s'%(seat_n,side),(x,side*1.04,.65),(.55,.22,.66),wood[1],interior_col)

for side,label in ((-1,'Stern'),(1,'Bow')):
    x=side*6.12
    for iz in range(10):
        box(label+' | end post block %02d'%iz,(x,0,.12+iz*U),(.48,.48,U),random.choice(wood),ends_col,random.choice(rail) if iz==9 else None)
    box(label+' | raised post crown',(x,0,2.52),(.54,.54,.24),wood[5],ends_col,rail[4])
    for z in (1.68,1.99):
        box(label+' | ivory collar front',(x,-.285,z),(.66,.13,.145),rope[0],ends_col,rope[1])
        box(label+' | ivory collar back',(x,.285,z),(.66,.13,.145),rope[0],ends_col,rope[1])
        for sx in (-1,1):
            box(label+' | ivory collar side',(x+sx*.285,0,z),(.13,.46,.145),rope[2],ends_col,rope[1])

# Compact open cargo crate, with timber slats and chunky metal corner straps.
cx,cy=-3.72,-.13
box('Crate | dark inset bottom',(cx,cy,.43),(1.03,.92,.17),wood[0],cargo_col)
for level in range(3):
    z=.57+level*.245
    for side in (-1,1):
        for plank in range(4):
            box('Crate | end slat',(cx+side*.48,cy+(plank-1.5)*.225,z),(.12,.217,.233),random.choice(wood),cargo_col)
            box('Crate | side slat',(cx+(plank-1.5)*.237,cy+side*.45,z),(.229,.12,.233),random.choice(wood),cargo_col)
for sx in (-1,1):
    for sy in (-1,1):
        box('Crate | iron corner upright',(cx+sx*.49,cy+sy*.455,.84),(.135,.135,.89),iron[0],cargo_col,iron[2])
        box('Crate | iron top corner',(cx+sx*.41,cy+sy*.455,1.28),(.29,.15,.12),iron[1],cargo_col)
for side in (-1,1):
    box('Crate | oak top rim',(cx,cy+side*.43,1.255),(.79,.15,.11),wood[2],cargo_col,rail[0])
box('Flask | squared body',(cx-.06,cy+.05,1.40),(.46,.43,.37),pewter[0],cargo_col,pewter[1])
box('Flask | shoulders',(cx-.06,cy+.05,1.60),(.33,.32,.14),pewter[1],cargo_col,pewter[2])
box('Flask | short neck',(cx-.06,cy+.05,1.73),(.19,.19,.16),pewter[1],cargo_col,pewter[2])

for i in range(-2,3):
    for j in range(-2,3):
        if abs(i)==2 and abs(j)==2: continue
        layers=2 if abs(i)==2 or abs(j)==2 else 3
        for k in range(layers):
            box('Provision sack | canvas voxel',(-5.00+i*.20,.03+j*.20,.39+k*.20),(.20,.20,.20),random.choice(olive),cargo_col)
box('Provision sack | tied fold',(-5.0,.03,1.00),(.30,.27,.10),olive[3],cargo_col)

# One solid, square-shaft wooden oar. Its blade is made of stepped voxel strips.
oar_root=bpy.data.objects.new('Oar | pivot at rowlock',None)
oar_col.objects.link(oar_root)
oar_root.location=(1.0,1.54,1.59)
oar_root.rotation_euler[2]=math.radians(25)
def oar_box(name,center,size,mat,top=None):
    ob=box(name,center,size,mat,oar_col,top)
    ob.parent=oar_root
    return ob
oar_box('Oar | long square shaft',(1.15,0,.01),(3.58,.18,.18),wood[2],rail[0])
oar_box('Oar | inner hand grip',(-.70,0,.01),(.45,.235,.23),wood[1],rail[1])
for ix in range(9):
    n=1 if ix==0 else 2 if ix in (1,8) else 3
    for iy in range(-n,n+1):
        x=2.80+ix*.205
        mat=random.choice(rail if iy<1 else wood[2:])
        oar_box('Oar | stepped blade block',(x,iy*.18,-.025),(.205,.18,.18),mat,mat)
for x in (2.95,3.95):
    oar_box('Oar | muted blade binding',(x,0,.077),(.105,.91,.028),patch[1],patch[1])
# Block-built wooden U bracket on the far gunwale.
box('Rowlock | base',(1.,1.46,1.54),(.60,.41,.20),wood[0],oar_col,rail[0])
for x in (.72,1.28):
    box('Rowlock | fork',(x,1.48,1.76),(.14,.36,.45),wood[2],oar_col,rail[0])
box('Rowlock | cross pin',(1.,1.52,1.88),(.79,.14,.14),wood[1],oar_col,rail[0])

boat_root=bpy.data.objects.new('Boat | complete asset',None)
default.objects.link(boat_root)
boat_root.empty_display_size=.35
boat_root['voxel_pitch']=U
for asset_col in (hull_col,interior_col,ends_col,cargo_col,oar_col):
    bpy.context.scene.collection.children.unlink(asset_col)
    default.children.link(asset_col)
    for component in asset_col.objects:
        if component.parent is None:
            component.parent=boat_root

# A separate studio collection keeps the actual model easy to select and reuse.
groundmat=material('Backdrop | deep blue',(.018,.065,.19))
box('Blue studio ground',(0,0,-.20),(200,200,.15),groundmat,studio_col)
scene=bpy.context.scene
world=bpy.data.worlds.new('Blue studio ambient')
world.use_nodes=True
world.node_tree.nodes.get('Background').inputs[0].default_value=(.23,.32,.48,1)
world.node_tree.nodes.get('Background').inputs[1].default_value=.45
scene.world=world
def aim(obj,point): obj.rotation_euler=(Vector(point)-obj.location).to_track_quat('-Z','Y').to_euler()
def light(name,loc,energy,size,color):
    data=bpy.data.lights.new(name,'AREA'); data.energy=energy; data.shape='DISK'; data.size=size; data.color=color
    obj=bpy.data.objects.new(name,data); studio_col.objects.link(obj); obj.location=loc; aim(obj,(0,0,.5))
light('Key | broad warm softbox',(-5,-7,12),2100,7,(1,.83,.65))
light('Fill | cool skylight',(5,4,9),1300,8,(.73,.84,1))
light('Rim | soft honey',(-4,6,7),850,6,(1,.88,.71))
cam_data=bpy.data.cameras.new('Camera | isometric reference view')
cam=bpy.data.objects.new('Camera | isometric reference view',cam_data)
studio_col.objects.link(cam)
cam.location=(20.3,-14.6,23)
aim(cam,(.30,.95,1.0))
cam_data.type='ORTHO'; cam_data.ortho_scale=13.4; cam_data.lens=48
scene.camera=cam
scene.render.engine='CYCLES'
scene.cycles.samples=48
scene.cycles.use_denoising=True
scene.render.resolution_x=1400
scene.render.resolution_y=1250
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.filepath=str(ROOT/'renders'/'boat.png')
scene.view_settings.view_transform='AgX'
scene.render.film_transparent=False
scene['asset']='Voxel wooden rowboat / reference-inspired handcrafted geometry'
scene['model_notes']='Hull, benches, end posts, cargo and oar are separately organized; studio is a separate collection.'
scene.unit_settings.system='METRIC'
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_perspective='CAMERA'
            area.spaces.active.shading.type='MATERIAL'
            area.spaces.active.overlay.show_overlays=False
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active=obj
(ROOT/'renders').mkdir(exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'boat.blend'))
bpy.ops.render.render(write_still=True)
print('BOAT_COMPLETE', len(cells), 'hull voxels;',len(bpy.data.objects),'objects')
