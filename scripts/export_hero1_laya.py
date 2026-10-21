"""Create a clean, skinned character and seven clips, leaving the source untouched.

Blender -b hero1_actions.blend --python scripts/export_hero1_laya.py -- --output-dir .
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path.cwd())
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
if bpy.context.object and bpy.context.object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
rig = bpy.data.objects["Hero_WalkRig"]
names = ("body", "head", "left-hand", "left-hand.001", "left-leg", "left-leg.001")
parts = [bpy.data.objects[name] for name in names]
actions = list(bpy.data.actions)
assert len(actions) == 7 and len(rig.data.bones) == 7
rig.animation_data.action = None
for bone in rig.pose.bones:
    bone.location = (0, 0, 0)
    bone.rotation_quaternion = (1, 0, 0, 0)
    bone.scale = (1, 1, 1)
bpy.context.view_layer.update()
rig_inverse = rig.matrix_world.inverted()
rest_vertices = [rig_inverse @ obj.matrix_world @ v.co for obj in parts for v in obj.data.vertices]
offset = Vector((0, 0, -min(v.z for v in rest_vertices)))
normalize = Matrix.Translation(offset) @ rig_inverse


def activate(action, frame):
    for bone in rig.pose.bones:
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.scale = (1, 1, 1)
    rig.animation_data.action = action
    rig.animation_data.action_slot = action.slots[0]
    scene.frame_set(int(frame), subframe=frame % 1)
    bpy.context.view_layer.update()


# Compare evaluated geometry before and after cleanup, across every clip.
references = {}
for action in actions:
    end = float(action.frame_range.y)
    for frame in (1.0, 1 + (end - 1) / 4, 1 + (end - 1) / 2, end):
        activate(action, frame)
        graph = bpy.context.evaluated_depsgraph_get()
        references[(action.name, frame)] = [normalize @ obj.matrix_world @ v.co
            for obj in parts for v in obj.evaluated_get(graph).data.vertices]
rig.animation_data.action = None
for bone in rig.pose.bones:
    bone.location = (0, 0, 0)
    bone.rotation_quaternion = (1, 0, 0, 0)
    bone.scale = (1, 1, 1)
bpy.context.view_layer.update()

# Build one mesh without baking its armature modifier into the geometry.
vertices, faces, weights = [], [], {}
# The material uses the first UV set; the imported second set has no consumers.
uv_names = [parts[0].data.uv_layers[0].name]
uvs = {name: [] for name in uv_names}
material = parts[0].data.materials[0]
for obj in parts:
    assert list(obj.data.materials) == [material]
    start = len(vertices)
    vertices.extend(normalize @ obj.matrix_world @ v.co for v in obj.data.vertices)
    faces.extend(tuple(start + i for i in p.vertices) for p in obj.data.polygons)
    for vertex in obj.data.vertices:
        for group in vertex.groups:
            name = obj.vertex_groups[group.group].name
            assert group.weight == 1 and name in rig.data.bones
            weights.setdefault(name, []).append(start + vertex.index)
    for name in uv_names:
        uvs[name].extend(tuple(item.uv) for item in obj.data.uv_layers[name].data)
mesh = bpy.data.meshes.new("HeroMesh")
mesh.from_pydata(vertices, [], faces)
mesh.materials.append(material)
for name in uv_names:
    layer = mesh.uv_layers.new(name=name)
    for item, uv in zip(layer.data, uvs[name]):
        item.uv = uv
mesh.update()
character = bpy.data.objects.new("HeroMesh", mesh)
scene.collection.objects.link(character)
for name, indices in weights.items():
    group = character.vertex_groups.new(name=name)
    group.add(indices, 1, "REPLACE")

rig.parent = None
rig.matrix_world = Matrix.Identity(4)
rig.data.transform(Matrix.Translation(offset))
rig.name = "Hero"
rig.data.name = "HeroSkeleton"
character.parent = None
character.matrix_parent_inverse = Matrix.Identity(4)
character.matrix_basis = Matrix.Identity(4)
modifier = character.modifiers.new("Character Skin", "ARMATURE")
modifier.object = rig
modifier.use_deform_preserve_volume = False

# Work only on this export copy; remove all scenery, props, orphan nodes and controls.
removed = []
for obj in list(bpy.data.objects):
    if obj not in (rig, character):
        removed.append(obj.name)
        bpy.data.objects.remove(obj, do_unlink=True)
clean_collection = bpy.data.collections.new("Character")
scene.collection.children.link(clean_collection)
for obj in (rig, character):
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    clean_collection.objects.link(obj)
for collection in list(bpy.data.collections):
    if collection != clean_collection:
        bpy.data.collections.remove(collection)
scene.camera = None
scene.name = "Hero"
scene.timeline_markers.clear()
for action in actions:
    action.use_fake_user = True
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in list(bag.fcurves):
                    if not curve.data_path.startswith('pose.bones['):
                        bag.fcurves.remove(curve)
    action.use_frame_range = True
    action.frame_start = 1
    action.frame_end = action.get("loop_frames", action.frame_range.y - 1) + 1
    # The original Walk does not key every transform. Explicit rest values make
    # switching from Sleep/Jump to Walk independent of the previous clip.
    existing = {(c.data_path, c.array_index) for layer in action.layers
                for strip in layer.strips for bag in strip.channelbags for c in bag.fcurves}
    activate(action, 1)
    for bone in rig.pose.bones:
        for channel, defaults in (("location", (0, 0, 0)),
                                  ("rotation_quaternion", (1, 0, 0, 0)),
                                  ("scale", (1, 1, 1))):
            path = bone.path_from_id(channel)
            for index, default in enumerate(defaults):
                if (path, index) not in existing:
                    getattr(bone, channel)[index] = default
                    bone.keyframe_insert(channel, index=index, frame=1, group=bone.name)
if rig.animation_data:
    for driver in list(rig.animation_data.drivers):
        rig.driver_remove(driver.data_path, driver.array_index)
    for track in list(rig.animation_data.nla_tracks):
        rig.animation_data.nla_tracks.remove(track)
for prop in ("chop_props", "sleep_lids", "tree_hit"):
    if prop in rig:
        del rig[prop]

max_error = 0
for (name, frame), expected in references.items():
    activate(bpy.data.actions[name], frame)
    graph = bpy.context.evaluated_depsgraph_get()
    actual = [character.matrix_world @ v.co for v in character.evaluated_get(graph).data.vertices]
    assert len(actual) == len(expected)
    max_error = max(max_error, max((a - b).length for a, b in zip(actual, expected)))
assert max_error < 1e-5, max_error
assert len(bpy.data.objects) == 2 and len(mesh.materials) == 1
assert all(len(v.groups) == 1 for v in mesh.vertices)

activate(bpy.data.actions["Hero_Run"], 1)
scene.frame_start, scene.frame_end = 1, 16
scene.render.fps, scene.render.fps_base = 24, 1
bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
for image in bpy.data.images:
    if image.source == "FILE" and image.has_data:
        image.pack()
bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
blend_path = output / "hero1_laya.blend"
glb_path = output / "hero1_laya.glb"
# Save an unambiguous Blender preview: only the skeleton is selected, in Pose
# Mode, so the Action Editor edits its bone animation rather than the mesh.
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
bpy.ops.object.mode_set(mode="OBJECT")
character.select_set(True)
result = bpy.ops.export_scene.gltf(
    filepath=str(glb_path), export_format="GLB", use_selection=True,
    export_animations=True, export_animation_mode="ACTIONS",
    export_frame_range=False, export_anim_slide_to_zero=True,
    export_force_sampling=True, export_frame_step=1,
    export_optimize_animation_size=True, export_reset_pose_bones=True,
    export_skins=True, export_def_bones=False, export_leaf_bone=False,
    export_armature_object_remove=True, export_rest_position_armature=True,
    export_morph=False, export_morph_animation=False,
    export_cameras=False, export_lights=False, export_extras=False,
    export_yup=True, export_materials="EXPORT", export_image_format="AUTO",
    export_draco_mesh_compression_enable=False, export_meshopt_compression_enable=False,
)
assert result == {"FINISHED"}
report = {"blend": str(blend_path), "glb": str(glb_path),
          "objects": [o.name for o in bpy.data.objects], "bones": len(rig.data.bones),
          "vertices": len(mesh.vertices), "triangles": sum(len(p.vertices)-2 for p in mesh.polygons),
          "materials": len(mesh.materials), "removed_objects": removed,
          "geometry_cleanup_max_error": max_error,
          "animations": [{"name": a.name, "seconds": a.get("loop_frames") / 24}
                         for a in actions]}
(output / "hero1_laya_export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("LAYA_EXPORT_OK", json.dumps(report, ensure_ascii=False))
