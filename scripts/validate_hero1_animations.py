"""Validate the hero animation library and GLB round-trip in an isolated process."""
from pathlib import Path
import itertools
import json
import math
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

out = Path(bpy.data.filepath).parent
asset_stem = Path(bpy.data.filepath).stem
character_name = asset_stem.removesuffix('_animated')
scene = bpy.context.scene
authoring_fps=scene.render.fps
rig = bpy.data.objects['Hero']
mesh = bpy.data.objects['HeroMesh']
assert all(o.name in ('Hero','HeroMesh') or o.get('preview_only') for o in scene.objects)
expected_bones = 18
assert len(rig.data.bones) == expected_bones
for side in ('L','R'):
    for name in ('elbow_scale','wrist_scale'):
        assert not rig.data.bones[name+'.'+side].use_deform
        assert mesh.vertex_groups.get(name+'.'+side) is None
assert rig.data.bones['tool.R'].parent.name=='hand.R'
assert not rig.data.bones['tool.R'].use_deform
assert len(mesh.data.materials) == 1
assert any(m.type=='ARMATURE' and m.object==rig for m in mesh.modifiers)
assert all(1<=len(v.groups)<=2 and abs(sum(g.weight for g in v.groups)-1)<1e-6 for v in mesh.data.vertices)
specs = {a.name: (int(a['loop_frames']),bool(a['looping'])) for a in bpy.data.actions}
rowing_names={'paddle_left','paddle_right'} & set(specs)
assert not rowing_names or len(rowing_names)==2
grilling_names={'grilling'} & set(specs)
work_names={'fishing','fishing_pull','chop','till','harvest','javelin_throw'}|rowing_names|grilling_names
life_names={'sitting','eating','clutch','crouching'}
raised_names={'hands_over_head','hold_up'}
distress_names={'stomach_pain','surrendering'}
breathing_names={'exhausted_breathing'} & set(specs)
assert set(specs)=={'idle','walk','walk_fast','run','jump','sleep'}|work_names|life_names|raised_names|distress_names|breathing_names

def select(name, frame):
    a = bpy.data.actions[name]
    rig.animation_data.action = a
    rig.animation_data.action_slot = a.slots[0]
    scene.frame_set(math.floor(frame),subframe=frame%1)
    bpy.context.view_layer.update()

def pts():
    graph = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(graph)
    return [mesh.matrix_world@v.co for v in ev.data.vertices]

def error(a,b):
    return max((x-y).length for x,y in zip(a,b))

foot_indices = [v.index for v in mesh.data.vertices
                if mesh.vertex_groups[v.groups[0].group].name.startswith('shin.')]
body_id=mesh.vertex_groups['body'].index
body_indices={v.index for v in mesh.data.vertices if any(g.group==body_id for g in v.groups)}
arm_ids={g.index for g in mesh.vertex_groups if g.name.startswith(('upper_arm.','forearm.','hand.'))}
arm_indices={v.index for v in mesh.data.vertices if any(g.group in arm_ids for g in v.groups)}
body_faces=[tuple(p.vertices) for p in mesh.data.polygons if all(i in body_indices for i in p.vertices)]
arm_faces=[tuple(p.vertices) for p in mesh.data.polygons if all(i in arm_indices for i in p.vertices)]
forearm_indices={v.index for v in mesh.data.vertices if any(mesh.vertex_groups[g.group].name.startswith(('forearm.','hand.')) for g in v.groups)}
forearm_faces=[tuple(p.vertices) for p in mesh.data.polygons if all(i in forearm_indices for i in p.vertices)]
head_indices={v.index for v in mesh.data.vertices if any(mesh.vertex_groups[g.group].name=='head' for g in v.groups)}
head_faces=[tuple(p.vertices) for p in mesh.data.polygons if all(i in head_indices for i in p.vertices)]
leg_ids={g.index for g in mesh.vertex_groups if g.name.startswith(('thigh.','shin.'))}
lower_leg_indices={v.index for v in mesh.data.vertices if v.co.z<.20 and any(g.group in leg_ids for g in v.groups)}
lower_leg_faces=[tuple(p.vertices) for p in mesh.data.polygons if all(i in lower_leg_indices for i in p.vertices)]
wrist_rings={}
for side in ('L','R'):
    ids=[v.index for v in mesh.data.vertices if abs(v.co.z-.52)<1e-5 and any(mesh.vertex_groups[g.group].name=='hand.'+side for g in v.groups)]
    assert len(ids)==4,(side,'Wrist is not a single welded ring')
    centre=sum((mesh.data.vertices[i].co for i in ids),Vector())/4
    wrist_rings[side]=sorted(ids,key=lambda i:math.atan2(mesh.data.vertices[i].co.y-centre.y,mesh.data.vertices[i].co.x-centre.x))
report = {'bones':expected_bones,'meshes':1,'clips':{},'transitions_checked':0}
references = {}
socket_references={}
for name,(length,looping) in specs.items():
    select(name,1)
    start = pts()
    select(name,length+1)
    end = pts()
    seam = error(start,end)
    assert seam < 1e-5,(name,'pose seam',seam)
    h = .001
    select(name,1+h)
    after = pts()
    select(name,length+1-h)
    before = pts()
    speed_error = max(((q-p)/h-(s-r)/h).length for p,q,r,s in zip(start,after,before,end))
    if looping:
        assert speed_error < .01,(name,'velocity seam',speed_error)
    floor = math.inf
    max_feet = -math.inf
    max_motion = 0
    thigh_max=knee_max=0
    grip_error=0
    tremor_pitch=[]
    distress_grip_error=0
    wrist_swing=wrist_twist=0;wrist_area=100
    for k in range(length*2+1):
        select(name,1+k*.5)
        current = pts()
        assert all(math.isfinite(c) for v in current for c in v)
        for side in ('L','R'):
            for segment in ('upper_arm','forearm'):
                scale=rig.pose.bones[segment+'.'+side].scale
                assert abs(scale.x-1)<1e-6 and abs(scale.z-1)<1e-6,(name,'Arm gets thicker',side,segment)
            palm=rig.pose.bones['hand.'+side]
            swing,twist=palm.rotation_quaternion.normalized().to_swing_twist('Y')
            wrist_swing=max(wrist_swing,math.degrees(swing.angle));wrist_twist=max(wrist_twist,abs(math.degrees(twist)))
            assert swing.angle<math.radians(35.1) and abs(twist)<math.radians(25.1),(name,'Exaggerated wrist rotation',side,1+k*.5)
            ring=[current[i] for i in wrist_rings[side]]
            area=sum((ring[i]-ring[0]).cross(ring[i+1]-ring[0]).length*.5 for i in (1,2))
            sf=rig.pose.bones['forearm.'+side].matrix.to_scale();sh=palm.matrix.to_scale()
            expected=.0016*(abs(sf.x)+abs(sh.x))*.5*(abs(sf.z)+abs(sh.z))*.5
            wrist_area=min(wrist_area,area/expected)
            assert area/expected>.85,(name,'Wrist collapses or looks detached',side,1+k*.5,area/expected)
        if name in work_names|life_names|raised_names|distress_names|breathing_names:
            for label,faces in (('body',body_faces),('head',head_faces)):
                intersections=BVHTree.FromPolygons(current,faces).overlap(BVHTree.FromPolygons(current,forearm_faces))
                assert not intersections,(name,'forearm/hand penetration',label,1+k*.5,len(intersections))
            if name in ('fishing','fishing_pull'):
                right,left=rig.pose.bones['hand.R'],rig.pose.bones['hand.L']
                delta=(left.head+left.tail-right.head-right.tail)*.5
                tool=rig.pose.bones['tool.R'];axis=(tool.tail-tool.head).normalized()
                grip_error=max(grip_error,(delta-axis*.08).length)
        if name in life_names|raised_names|distress_names|{'till','harvest','javelin_throw'}|rowing_names|breathing_names|grilling_names:
            for side in ('L','R'):
                hinge=rig.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ')
                assert -145<math.degrees(hinge.z)<0,(name,'elbow reverses bend',side,1+k*.5)
                assert max(abs(hinge.x),abs(hinge.y))<.001,(name,'elbow rotates off hinge',side)
        if name in raised_names:
            for side in ('L','R'):
                palm=rig.pose.bones['hand.'+side]
                # The plain voxel palms are cubes; quarter-turn equivalents are
                # chosen to protect the wrist seam while retaining the held pose.
                assert ((palm.head+palm.tail)*.5).z>1.23,(name,'hands are not raised above the head',side)
        if name in distress_names:
            body=rig.pose.bones['body'];head=rig.pose.bones['head']
            up=(body.tail-body.head).normalized()
            pitch=math.degrees(math.atan2(up.x,up.z));tremor_pitch.append(pitch)
            assert pitch>(35 if name=='stomach_pain' else 28),(name,'body is not hunched')
            head_up=(head.tail-head.head).normalized()
            assert math.degrees(math.atan2(head_up.x,head_up.z))>49,(name,'head is not bowed')
            for side in ('L','R'):
                bend=math.degrees(rig.pose.bones['shin.'+side].rotation_quaternion.angle)
                assert bend>(63 if name=='stomach_pain' else 126),(name,'knees not bent',side,bend)
            if name=='stomach_pain':
                right,left=rig.pose.bones['hand.R'],rig.pose.bones['hand.L']
                local=body.bone.matrix_local@body.matrix.inverted()@((right.head+right.tail)*.5)
                distress_grip_error=max(distress_grip_error,(local-Vector((.215,.045,.60))).length)
                assert ((left.head+left.tail)*.5).z<rig.pose.bones['upper_arm.L'].head.z-.28,'Free hand does not hang down'
            else:
                for side,y in (('L',-.195),('R',.235)):
                    palm=rig.pose.bones['hand.'+side]
                    local=head.bone.matrix_local@head.matrix.inverted()@((palm.head+palm.tail)*.5)
                    distress_grip_error=max(distress_grip_error,(local-Vector((-.015,y,1.08))).length)
        if name in ('sitting','crouching','stomach_pain','surrendering'):
            intersections=BVHTree.FromPolygons(current,body_faces).overlap(BVHTree.FromPolygons(current,lower_leg_faces))
            assert not intersections,(name,'knees/calves intersect torso',1+k*.5,len(intersections))
        if name=='jump':
            intersections=BVHTree.FromPolygons(current,body_faces).overlap(BVHTree.FromPolygons(current,arm_faces))
            assert not intersections,('jump arm/body penetration',1+k*.5,len(intersections))
        if name=='run':
            intersections=BVHTree.FromPolygons(current,body_faces).overlap(BVHTree.FromPolygons(current,lower_leg_faces))
            assert not intersections,('run knees/calves intersect torso',1+k*.5,len(intersections))
        if name in ('walk','walk_fast','run'):
            assert max(abs(current[i].y-mesh.data.vertices[i].co.y) for i in arm_indices)<1e-5,('lateral arm swing',name)
            for side in ('L','R'):
                thigh=rig.pose.bones['thigh.'+side]
                shin=rig.pose.bones['shin.'+side]
                thigh_max=max(thigh_max,math.degrees(thigh.rotation_quaternion.angle))
                knee_max=max(knee_max,math.degrees(shin.rotation_quaternion.angle))
                assert shin.head.z<rig.pose.bones['hips'].head.z-(.045 if name=='run' else .12),('knee raised into torso',name)
        floor = min(floor,min(v.z for v in current))
        max_feet = max(max_feet,min(current[i].z for i in foot_indices))
        max_motion = max(max_motion,error(start,current))
    assert max_motion > .002,(name,'no visible mesh motion')
    assert floor > -.001,(name,'floor penetration',floor)
    if name=='run':
        assert max_feet > .085,('run has no flight',max_feet)
    if name=='jump':
        assert max_feet > .50,('jump has insufficient height',max_feet)
    if name=='sleep':
        assert max(p.z for p in start)-min(p.z for p in start) < .40
    report['clips'][name] = {'length_frames':length,'looping':looping,'seam_position_error':seam,
        'seam_velocity_error_per_frame':speed_error,'min_ground_z':floor,
        'max_lowest_foot_z':max_feet,'max_vertex_displacement':max_motion,
        'wrist_swing_max_degrees':wrist_swing,'wrist_twist_max_degrees':wrist_twist,'min_wrist_crosssection_ratio':wrist_area}
    if name=='jump':report['clips'][name]['arm_body_collision_samples_passed']=length*2+1
    if name in work_names|life_names|raised_names|distress_names|breathing_names:
        report['clips'][name]['forearm_hand_body_head_samples_passed']=length*2+1
        if name in ('fishing','fishing_pull'):
            assert grip_error<.02,(name,'hand outside rod grip',grip_error)
            report['clips'][name]['two_hand_grip_max_error']=grip_error
    if name in raised_names:
        report['clips'][name]['raised_hand_samples_passed']=length*2+1
    if name in distress_names:
        peaks=sum(tremor_pitch[i]>tremor_pitch[i-1] and tremor_pitch[i]>=tremor_pitch[i+1] for i in range(1,len(tremor_pitch)-1))
        expected=3 if name=='stomach_pain' else 4
        assert peaks==expected,(name,'tremor cycle count',peaks)
        assert max(tremor_pitch)-min(tremor_pitch)>1.5,(name,'no torso trembling')
        assert distress_grip_error<.006,(name,'hand slips from stomach/scalp',distress_grip_error)
        report['clips'][name].update(tremor_cycles=peaks,hand_contact_position_error=distress_grip_error)
    if name in ('walk','walk_fast','run'):
        assert thigh_max<={'walk':18,'walk_fast':30,'run':72}[name]+.05,(name,'thigh swing',thigh_max)
        assert knee_max<={'walk':26,'walk_fast':40,'run':80}[name]+.05,(name,'knee bend',knee_max)
        report['clips'][name].update(max_thigh_swing_degrees=thigh_max,max_knee_bend_degrees=knee_max,
                                    lateral_arm_swing=False)
    for fraction in (0,.125,.25,.375,.5,.625,.75,.875,1):
        # Integer export samples provide an exact geometry comparison.
        f = round(length*fraction)+1
        select(name,f)
        references[(name,f)] = pts()
        socket_references[(name,f)]=rig.matrix_world@rig.pose.bones['tool.R'].head

# Every ordered transition must evaluate identically to a clean start.
for previous,target in itertools.permutations(specs,2):
    select(previous,1+specs[previous][0]*.37)
    select(target,1)
    assert error(pts(),references[(target,1)]) < 1e-6,(previous,target)
    report['transitions_checked'] += 1

# Import into an empty scene; do not touch the user's live Blender.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = authoring_fps
bpy.ops.import_scene.gltf(filepath=str(out/(asset_stem+'.glb')))
scene = bpy.context.scene
scene.render.fps = authoring_fps
rigs = [o for o in scene.objects if o.type=='ARMATURE']
meshes = [o for o in scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers)]
assert len(rigs)==len(meshes)==1
rig,mesh = rigs[0],meshes[0]
assert len(rig.data.bones)==expected_bones
assert rig.data.bones['tool.R'].parent.name=='hand.R'
assert set(a.name for a in bpy.data.actions)==set(specs)
maximum = 0
maximum_socket=0
for (name,frame),expected in references.items():
    a = bpy.data.actions[name]
    rig.animation_data.action = a
    rig.animation_data.action_slot = a.slots[0]
    first = a.frame_range.x
    assert abs((a.frame_range.y-first)-specs[name][0])<1e-4,(name,'duration')
    scene.frame_set(round(first)+frame-1)
    bpy.context.view_layer.update()
    socket_error=((rig.matrix_world@rig.pose.bones['tool.R'].head)-socket_references[(name,frame)]).length
    maximum_socket=max(maximum_socket,socket_error)
    assert socket_error<1e-5,(name,frame,'tool attachment moves after roundtrip',socket_error)
    actual = pts()
    e = 0
    for source,target in ((expected,actual),(actual,expected)):
        tree = KDTree(len(source))
        for i,v in enumerate(source):
            tree.insert(v,i)
        tree.balance()
        e = max(e,max(tree.find(v)[2] for v in target))
    maximum = max(maximum,e)
    assert e < 1e-5,(name,frame,'round-trip vertex error',e)
report['roundtrip_sampled_poses'] = len(references)
report['roundtrip_max_position_error'] = maximum
report['roundtrip_max_tool_socket_error']=maximum_socket
(out/(character_name+'_animation_validation.json')).write_text(json.dumps(report,indent=2))
print('VALIDATION_PASSED',json.dumps(report))
