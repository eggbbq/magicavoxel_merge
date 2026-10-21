"""Round-trip the GLB and compare skinned geometry for every exported action."""

import json
from pathlib import Path

import bpy
from mathutils.kdtree import KDTree

output = Path(bpy.data.filepath).parent
scene = bpy.context.scene
rig = bpy.data.objects["Hero"]
mesh = bpy.data.objects["HeroMesh"]
references = {}
durations = {}


def points(obj):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    return [obj.matrix_world @ v.co for v in obj.evaluated_get(graph).data.vertices]


for action in bpy.data.actions:
    for bone in rig.pose.bones:
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.scale = (1, 1, 1)
    durations[action.name] = action["loop_frames"]
    rig.animation_data.action = action
    rig.animation_data.action_slot = action.slots[0]
    for fraction in (0, .25, .5, .75, 1):
        frame = 1 + action["loop_frames"] * fraction
        scene.frame_set(int(frame), subframe=frame % 1)
        references[(action.name, fraction)] = points(mesh)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 24
bpy.ops.import_scene.gltf(filepath=str(output / "hero1_laya.glb"))
scene = bpy.context.scene
rigs = [o for o in scene.objects if o.type == "ARMATURE"]
# Blender creates an Icosphere as a hidden bone display shape during glTF import.
meshes = [o for o in scene.objects if o.type == "MESH" and o.name == "HeroMesh"]
assert len(rigs) == len(meshes) == 1
rig, mesh = rigs[0], meshes[0]
assert len(rig.data.bones) == 7
assert any(m.type == "ARMATURE" and m.object == rig for m in mesh.modifiers)
clips = {a.name: a for a in bpy.data.actions}
assert set(clips) == set(durations), list(clips)
max_error = 0
errors = []
for (name, fraction), expected in references.items():
    action = clips[name]
    rig.animation_data.action = action
    rig.animation_data.action_slot = action.slots[0]
    frame = float(action.frame_range.x) + durations[name] * fraction
    scene.frame_set(int(frame), subframe=frame % 1)
    actual = points(mesh)
    pose_error = 0
    for source, target in ((expected, actual), (actual, expected)):
        tree = KDTree(len(source))
        for i, point in enumerate(source):
            tree.insert(point, i)
        tree.balance()
        pose_error = max(pose_error, max(tree.find(point)[2] for point in target))
    max_error = max(max_error, pose_error)
    errors.append((name, fraction, pose_error,
                   [min(p[i] for p in expected) for i in range(3)],
                   [min(p[i] for p in actual) for i in range(3)]))
assert max_error < 1e-5, max_error
report = {"clips": len(clips), "sampled_poses": len(references),
          "bones": len(rig.data.bones), "skinned_meshes": len(meshes),
          "roundtrip_max_position_error": max_error}
(output / "hero1_laya_roundtrip.json").write_text(json.dumps(report, indent=2))
print("ROUNDTRIP_OK", json.dumps(report))
