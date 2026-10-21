"""Validate and render new actions without changing the saved Blender scene."""

from pathlib import Path
import sys
import bpy
from mathutils import Vector


scene = bpy.context.scene
rig = bpy.data.objects["Hero_WalkRig"]
hero = bpy.data.objects["hero"]
parts = [obj for obj in hero.children if obj.type == "MESH" and obj.name in
         ("head", "body", "left-hand", "left-hand.001", "left-leg", "left-leg.001")]
assert len(parts) == 6 and len(rig.data.bones) == 7


def select(name, frame=1):
    action = bpy.data.actions[name]
    rig.animation_data.action = action
    rig.animation_data.action_slot = action.slots[0]
    scene.frame_set(frame)
    bpy.context.view_layer.update()


def vertices(objects):
    graph = bpy.context.evaluated_depsgraph_get()
    return {obj.name: [obj.matrix_world @ v.co for v in obj.evaluated_get(graph).data.vertices]
            for obj in objects}


lengths = {"Hero_Sleep": 96, "Hero_Run": int(bpy.data.actions["Hero_Run"]["loop_frames"]), "Hero_ChopTree": 36}
if bpy.data.actions.get("Hero_Run_Exaggerated"):
    lengths["Hero_Run_Exaggerated"] = int(bpy.data.actions["Hero_Run_Exaggerated"]["loop_frames"])
for name, length in lengths.items():
    if "--exaggerated-only" in sys.argv and name != "Hero_Run_Exaggerated":
        continue
    select(name)
    start = vertices(parts)
    select(name, length + 1)
    seam = vertices(parts)
    assert all(max((a-b).length for a,b in zip(start[n],seam[n])) < 1e-5 for n in start), name
    select(name)
    assert bpy.data.objects["Axe_Handle"].hide_render == (name != "Hero_ChopTree"), name
    assert bpy.data.objects["Tree_Trunk"].hide_render == (name != "Hero_ChopTree"), name
    assert bpy.data.objects["Sleep_Eyelid_0"].hide_render == (name != "Hero_Sleep"), name
    print("LOOP_AND_PROPS_OK:", name)

# Existing clips keep their movement and hide the new preview props.
for name in ("Hero_Walk_InPlace", "Hero_Idle", "Hero_Jump"):
    select(name)
    assert bpy.data.objects["Axe_Handle"].hide_render
    assert bpy.data.objects["Sleep_Eyelid_0"].hide_render

select("Hero_Run", 1)
legs = [obj for obj in parts if "leg" in obj.name]
base = min(v.z for vs in vertices(legs).values() for v in vs)
for frame in (1, 1 + lengths["Hero_Run"] // 2, 1 + lengths["Hero_Run"]):
    select("Hero_Run", frame)
    low = min(v.z for vs in vertices(legs).values() for v in vs)
    assert abs(low-base) < 1e-5, (frame, low, base)
select("Hero_Run", 1 + lengths["Hero_Run"] // 4)
assert min(v.z for vs in vertices(legs).values() for v in vs) > base + .05
select("Hero_Sleep", 1)
points = [v for vs in vertices(parts).values() for v in vs]
assert max(v.z for v in points)-min(v.z for v in points) < .65, "lying pose"
assert max(v.x for v in points)-min(v.x for v in points) > 1, "lying pose"
print("MOTION_OK: planted running contacts, flight, sleeping horizontal, actions retained:", len(bpy.data.actions))

output = Path(bpy.data.filepath).parent / "hero1_action_previews"
output.mkdir(exist_ok=True)
scene.render.engine = "CYCLES"
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = 448
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"
scene.world = bpy.data.worlds.new("Action Preview World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.12, 0.15, 0.20, 1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = .6
camera_data = bpy.data.cameras.new("Action Preview Camera")
camera = bpy.data.objects.new("Action Preview Camera", camera_data)
scene.collection.objects.link(camera)
camera_data.type = "ORTHO"
scene.camera = camera
lights = []
for name, offset, energy, size in (("Key", (3,-4,5),350,4),
                                   ("Fill",(-2,-1,3),160,3),("Rim",(1,3,4),250,3)):
    data = bpy.data.lights.new("Action Preview " + name, "AREA")
    data.energy, data.shape, data.size = energy, "DISK", size
    light = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(light)
    lights.append((light,Vector(offset)))
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,base-.002))
ground = bpy.context.object
mat = bpy.data.materials.new("Action Preview Ground")
mat.diffuse_color = (.22,.27,.34,1)
ground.data.materials.append(mat)
for name, length in lengths.items():
    if "--exaggerated-only" in sys.argv and name != "Hero_Run_Exaggerated":
        continue
    if "--run-only" in sys.argv and name != "Hero_Run":
        continue
    select(name)
    subjects = list(parts)
    if name == "Hero_ChopTree":
        subjects += [bpy.data.objects[n] for n in ("Tree_Trunk","Tree_Crown","Tree_CrownTop")]
    points = [v for vs in vertices(subjects).values() for v in vs]
    lo = Vector(tuple(min(v[i] for v in points) for i in range(3)))
    hi = Vector(tuple(max(v[i] for v in points) for i in range(3)))
    center = (lo + hi) / 2
    camera.location = center + Vector((3.4,-4.4,2.6))
    camera.rotation_euler = (center-camera.location).to_track_quat("-Z","Y").to_euler()
    camera_data.ortho_scale = 2.35 if name == "Hero_ChopTree" else 1.85
    for light, offset in lights:
        light.location = center + offset
        light.rotation_euler = (center-light.location).to_track_quat("-Z","Y").to_euler()
    step = 12 if name == "Hero_Sleep" else (1 if name.startswith("Hero_Run") else 4)
    for frame in range(1,length+1,step):
        select(name,frame)
        scene.render.filepath = str(output / f"{name}_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)
print("PREVIEWS_SAVED:", output)
