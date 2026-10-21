"""Build the current 12-part hero, not the older six-part hero.

Blender -b hero1.blend --python scripts/build_hero1_animations.py
Outputs hero1_animated.blend and hero1_animated.glb beside the source.
The original source is never overwritten. Can also run in the live Blender.
"""
from pathlib import Path
import json
import math
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

OUT = Path(bpy.data.filepath).parent
SCENE = bpy.context.scene
assert not any(o.type == 'ARMATURE' for o in SCENE.objects), 'Use the unbound source file.'
PARTS = [o for o in SCENE.objects if o.type == 'MESH']
assert len(PARTS) == 12, 'Expected head, body, and ten articulated limb parts.'
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
if bpy.context.screen and bpy.context.screen.is_animation_playing:
    bpy.ops.screen.animation_cancel(restore_frame=False)

def bounds(o):
    pts = [o.matrix_world @ v.co for v in o.data.vertices]
    return (Vector([min(v[i] for v in pts) for i in range(3)]),
            Vector([max(v[i] for v in pts) for i in range(3)]))

boxes = {o: bounds(o) for o in PARTS}
body = next(o for o in PARTS if o.name == 'body')
head = next(o for o in PARTS if o.name == 'head')
limbs = {}
for part in ('upper-arm', 'forearm', 'palm', 'thigh', 'calf'):
    pair = sorted([o for o in PARTS if o.name.split('.')[0] == part],
                  key=lambda o: boxes[o][0].y)
    assert len(pair) == 2, part
    for side, obj in zip(('L', 'R'), pair):
        limbs[(part, side)] = obj

# Bake imported transforms into independent meshes before assigning weights.
reference_points = [o.matrix_world @ v.co for o in PARTS for v in o.data.vertices]
for o in PARTS:
    world = o.matrix_world.copy()
    o.data = o.data.copy()
    o.parent = None
    o.data.transform(world)
    o.matrix_world = Matrix.Identity(4)

data = bpy.data.armatures.new('HeroSkeleton')
rig = bpy.data.objects.new('Hero', data)
SCENE.collection.objects.link(rig)
rig.show_in_front = True
data.display_type = 'STICK'
hip = Vector((.06, .01, .30))
bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')

def bone(name, start, end, parent=None, deform=True):
    b = data.edit_bones.new(name)
    b.head, b.tail, b.use_deform = start, end, deform
    if parent:
        b.parent = data.edit_bones[parent]
    return b

bone('hips', hip, hip + Vector((0, 0, .08)), deform=False)
bone('body', hip, (.06, .01, .78), 'hips')
bone('head', (.00, .02, .82), (.00, .02, 1.16), 'body')
mapping = {body: 'body', head: 'head'}
for side in ('L', 'R'):
    for kind, prefix, parent in (
        ('upper-arm', 'upper_arm', 'body'),
        ('forearm', 'forearm', 'upper_arm.' + side),
        ('palm', 'hand', 'forearm.' + side),
        ('thigh', 'thigh', 'hips'),
        ('calf', 'shin', 'thigh.' + side),
    ):
        obj = limbs[(kind, side)]
        low, high = boxes[obj]
        c = (low + high) * .5
        name = prefix + '.' + side
        bone(name, (c.x, c.y, high.z), (c.x, c.y, low.z), parent)
        mapping[obj] = name
bpy.ops.object.mode_set(mode='OBJECT')
for obj, name in mapping.items():
    obj.vertex_groups.clear()
    obj.vertex_groups.new(name=name).add(list(range(len(obj.data.vertices))), 1, 'REPLACE')
    mod = obj.modifiers.new('Character skin', 'ARMATURE')
    mod.object = rig

# Keep only the character. The source and the in-memory backup retain the import hierarchy.
for o in list(SCENE.objects):
    if o not in PARTS and o != rig:
        bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.select_all(action='DESELECT')
for o in PARTS:
    o.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
mesh = bpy.context.object
mesh.name = 'HeroMesh'
mesh.data.name = 'HeroGeometry'
mesh.animation_data_clear()
# The material uses the active render UV through an unconnected Image Texture vector.
used_uvs = {n.uv_map for mat in mesh.data.materials if mat and mat.use_nodes
            for n in mat.node_tree.nodes if n.type == 'UVMAP' and n.uv_map}
for uv in list(mesh.data.uv_layers):
    if not uv.active_render and uv.name not in used_uvs:
        mesh.data.uv_layers.remove(uv)
assert len(mesh.data.vertices) == len(reference_points)
from mathutils.kdtree import KDTree
tree = KDTree(len(reference_points))
for i, p in enumerate(reference_points):
    tree.insert(p, i)
tree.balance()
assert max(tree.find(v.co)[2] for v in mesh.data.vertices) < 1e-6
sys.path.insert(0,str(OUT/'scripts'))
from sew_hero1_joints import sew_limb_joints
joint_report = sew_limb_joints(mesh)
assert len(data.bones) == 13
for image in bpy.data.images:
    if image.source == 'FILE' and image.has_data and not image.packed_file:
        image.pack()

REST = {b.name: b.matrix_local.to_quaternion() for b in data.bones}
NAMES = list(REST)
RAD = math.pi / 180
TAU = math.tau

def qrot(pitch=0, roll=0, yaw=0):
    return (Quaternion((0, 0, 1), yaw * RAD) @
            Quaternion((0, 1, 0), pitch * RAD) @
            Quaternion((1, 0, 0), roll * RAD))

def empty_pose():
    return {n: {'location': (0., 0., 0.), 'rotation_quaternion': (1., 0., 0., 0.),
                'scale': (1., 1., 1.)} for n in NAMES}

def rotate(pose, name, pitch=0, roll=0, yaw=0):
    rest = REST[name]
    q = rest.inverted() @ qrot(pitch, roll, yaw) @ rest
    if q.w < 0:
        q.negate()
    pose[name]['rotation_quaternion'] = tuple(q)

def move(pose, name, xyz):
    pose[name]['location'] = tuple(REST[name].inverted() @ Vector(xyz))

def leg(pose, side, hip_z, foot_x, foot_z):
    """Sagittal two-link solve, knee points forwards (+X), no runtime constraints."""
    upper = data.bones['thigh.' + side].length
    lower = data.bones['shin.' + side].length
    dx, dz = foot_x - .06, foot_z - hip_z
    distance = math.hypot(dx, dz)
    # Smoothly approach extension; a hard reach clamp produces a visible knee snap.
    cap, width = upper + lower - .003, .004
    d = cap - width * math.log1p(math.exp(min(100, (cap - distance) / width)))
    d = max(abs(upper - lower) + .002, d)
    center = math.atan2(-dx, -dz)
    beta = math.acos(max(-1, min(1, (upper**2 + d*d - lower**2) / (2*upper*d))))
    a = center - beta
    sx, sz = -math.sin(center)*d + upper*math.sin(a), -math.cos(center)*d + upper*math.cos(a)
    b = math.atan2(-sx, -sz)
    rotate(pose, 'thigh.' + side, a / RAD)
    rotate(pose, 'shin.' + side, (b-a) / RAD)

def hermite(a, b, va, vb, t):
    return (2*t**3-3*t*t+1)*a + (t**3-2*t*t+t)*va + (-2*t**3+3*t*t)*b + (t**3-t*t)*vb

def locomotion(t, running):
    p = empty_pose()
    phase = t % 1
    if running:
        # A readable voxel run: low thigh swing, modest heel recovery, two flights.
        # Direct joint angles avoid the old short-leg IK pulling the knees into the dress.
        if running=='fast':
            height = .283 + .130 * math.sin(TAU*phase)**2
            tilt, arm_amp, thigh_amp, thigh_bias, knee_base, knee_recovery = 10,45,28,2,5,35
        else:
            height = .224 + .170 * math.sin(TAU*phase)**2
            tilt = 34 + 6*math.sin(TAU*phase)**2
            arm_amp, thigh_amp, thigh_bias, knee_base, knee_recovery = 90,61,11,8,72
    else:
        height = .296 + .007 * math.sin(TAU*phase)**2
        tilt, arm_amp, thigh_amp, thigh_bias, knee_base, knee_recovery = 2,22,18,0,3,23
    move(p, 'hips', (0, 0, height-.30))
    rotate(p, 'body', tilt)
    rotate(p, 'head', -tilt*(.80 if running is True else .55))
    for side, offset in (('L', 0), ('R', .5)):
        f = (phase+offset) % 1
        swing = math.cos(TAU*f)
        recovery=max(0,-math.sin(TAU*f))**2
        rotate(p,'thigh.'+side,thigh_bias-thigh_amp*swing)
        rotate(p,'shin.'+side,knee_base+knee_recovery*recovery)
        # No lateral shoulder rotation or torso twist: hands stay in their side planes.
        rotate(p, 'upper_arm.'+side, arm_amp*swing+(-35 if running is True else 0))
        elbow=-(48+8*math.sin(TAU*f)) if running=='fast' else (-(60+20*swing) if running else -8)
        rotate(p, 'forearm.'+side, elbow)
        rotate(p, 'hand.'+side, 0)
    return p

def idle(t):
    p = empty_pose()
    breath = math.sin(TAU*t)
    p['body']['scale'] = (1, 1+.006*breath, 1)
    rotate(p, 'body', .6*breath)
    rotate(p, 'head', -.7*breath, yaw=1.8*math.sin(TAU*t))
    for side, sign in (('L', 1), ('R', -1)):
        rotate(p, 'upper_arm.'+side, 1.4*breath, roll=sign*2)
        rotate(p, 'forearm.'+side, -5-1.5*breath)
    return p

JUMP_POSES = [
    # frame offset, hip height, foot x relative to hip, foot z, lean, arm, elbow
    (0, .300, 0, .000, 0, 0, -5),
    (6, .205, .018, .020, 14, 32, -28),
    (10, .190, .025, .020, 17, 40, -35),
    (13, .295, .005, .022, 5, -90, -20),
    (19, .650, .060, .460, 0, -145, -25),
    (25, .800, .075, .615, -4, -130, -45),
    (31, .635, .050, .410, 0, -105, -30),
    (36, .300, .018, .020, 7, -35, -25),
    (40, .195, .025, .020, 16, 24, -32),
    (44, .265, .008, .014, 5, 10, -12),
    (48, .300, 0, .000, 0, 0, -5),
]

def jump_slope(index, channel):
    if index == 0 or index == len(JUMP_POSES)-1:
        return 0.
    a,b,c = JUMP_POSES[index-1:index+2]
    h0,h1 = b[0]-a[0],c[0]-b[0]
    d0,d1 = (b[channel]-a[channel])/h0,(c[channel]-b[channel])/h1
    if d0*d1 <= 0:
        return 0.
    w0,w1 = 2*h1+h0,h1+2*h0
    return (w0+w1)/(w0/d0+w1/d1)

def jump(t):
    f = min(48, max(0, t*48))
    for index,(a, b) in enumerate(zip(JUMP_POSES, JUMP_POSES[1:])):
        if f <= b[0]:
            u = (f-a[0])/(b[0]-a[0])
            h = b[0]-a[0]
            height, x, z, lean, arm, elbow = [hermite(a[i],b[i],h*jump_slope(index,i),
                h*jump_slope(index+1,i),u) for i in range(1,7)]
            break
    p = empty_pose()
    # Allow for the square voxel shin corners near full extension.
    move(p, 'hips', (0, 0, height-.3+.004))
    rotate(p, 'body', lean)
    rotate(p, 'head', -lean*.5)
    for side, sign in (('L', 1), ('R', -1)):
        leg(p, side, height, .06+x, z)
        # Abduct the arm first, then swing: the lateral offset keeps its sign
        # even when the upper arm passes above shoulder height.
        # With forward +X, negative-Y left arms abduct with a negative X rotation.
        outward = 14 + 24 * math.sin(math.pi*f/48)**2
        rest = REST['upper_arm.'+side]
        q = qrot(arm,roll=-sign*outward)
        p['upper_arm.'+side]['rotation_quaternion'] = tuple(rest.inverted() @ q @ rest)
        rotate(p, 'forearm.'+side, elbow)
    return p

SLEEP_OFFSET = -.055
def sleep(t):
    p = empty_pose()
    breathe = math.sin(TAU*t)
    rotate(p, 'hips', -90)
    move(p, 'hips', (0, 0, SLEEP_OFFSET))
    p['body']['scale'] = (1, 1+.004*breathe, 1)
    rotate(p, 'head', 2+.5*breathe)
    for side, sign in (('L',1),('R',-1)):
        rotate(p, 'upper_arm.'+side, -5, roll=sign*10)
        rotate(p, 'forearm.'+side, -150-1*breathe)
        rotate(p, 'hand.'+side, 8)
        rotate(p, 'thigh.'+side, -9)
        rotate(p, 'shin.'+side, 18)
    return p

def apply_pose(p):
    for name, channels in p.items():
        pb = rig.pose.bones[name]
        pb.rotation_mode = 'QUATERNION'
        for channel, value in channels.items():
            setattr(pb, channel, value)
    bpy.context.view_layer.update()

def points():
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(graph)
    return [mesh.matrix_world @ v.co for v in evaluated.data.vertices]

# Ground the lying pose once, not every frame, to avoid breathing-driven root jitter.
apply_pose(sleep(0))
SLEEP_OFFSET += .004 - min(v.z for v in points())

SPECS = {'idle': (72, idle, True),
         'walk': (32, lambda t: locomotion(t, False), True),
         'walk_fast': (20, lambda t: locomotion(t, 'fast'), True),
         'run': (10, lambda t: locomotion(t, True), True),
         'jump': (48, jump, False),
         'sleep': (96, sleep, True)}

def curves(action):
    return [c for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags for c in bag.fcurves]

SCENE.render.fps = 24
SCENE.render.fps_base = 1
SCENE.sync_mode = 'FRAME_DROP'
rig.animation_data_create()
stats = {}
for name, (length, function, looping) in SPECS.items():
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    action['loop_frames'] = length
    action['looping'] = looping
    action.use_frame_range = True
    action.frame_start, action.frame_end = 1, length+1
    rig.animation_data.action = action
    # Seed all reset channels, including constants, so action switching is independent.
    apply_pose(function(0))
    for pb in rig.pose.bones:
        for channel in ('location','rotation_quaternion','scale'):
            pb.keyframe_insert(channel, frame=1, group=pb.name)
    action = rig.animation_data.action
    cache = {}
    def values(frame):
        if frame not in cache:
            cache[frame] = function((frame-1)/length)
        return cache[frame]
    for c in curves(action):
        bone_name = c.data_path.split('"')[1]
        channel = c.data_path.rsplit('.',1)[-1]
        component = c.array_index
        def val(frame):
            return values(frame)[bone_name][channel][component]
        sample_values = [val(1+length*k/64) for k in range(65)]
        if max(sample_values)-min(sample_values) < 1e-8:
            # Keep the root quaternion timeline extent even when it is constant.
            if bone_name == 'hips' and channel == 'rotation_quaternion' and component == 0:
                c.keyframe_points.insert(length+1, val(length+1))
            continue
        # Shared major poses, rather than independently densifying every channel.
        # Eight poses + duplicate endpoint for loops; eleven poses for the jump.
        ordered = ([1+length*k/8 for k in range(9)] if name != 'jump'
                   else [1+row[0] for row in JUMP_POSES])
        ys=[val(f) for f in ordered]
        def slope_at(i):
            if i in (0,len(ordered)-1):
                if not looping:return 0.
                h0=ordered[-1]-ordered[-2];h1=ordered[1]-ordered[0]
                d0=(ys[-1]-ys[-2])/h0;d1=(ys[1]-ys[0])/h1
            else:
                h0=ordered[i]-ordered[i-1];h1=ordered[i+1]-ordered[i]
                d0=(ys[i]-ys[i-1])/h0;d1=(ys[i+1]-ys[i])/h1
            if d0*d1<=0:return 0.
            w0,w1=2*h1+h0,h1+2*h0
            return (w0+w1)/(w0/d0+w1/d1)
        c.keyframe_points.clear()
        for f in ordered:
            c.keyframe_points.insert(f,val(f),options={'FAST'})
        c.update()
        for i,k in enumerate(c.keyframe_points):
            f = ordered[i]
            left = (f-ordered[i-1])/3 if i else (ordered[-1]-ordered[-2])/3
            right = (ordered[i+1]-f)/3 if i+1<len(ordered) else (ordered[1]-ordered[0])/3
            k.interpolation = 'BEZIER'
            k.handle_left_type = k.handle_right_type = 'FREE'
            slope = slope_at(i)
            k.handle_left = (f-left,val(f)-slope*left)
            k.handle_right = (f+right,val(f)+slope*right)
        c.update()
    if name=='jump':
        # Three root-only support keys keep the simplified crouch/landing on the
        # floor without densifying the limb animation or changing the major poses.
        axis=REST['hips'].inverted()@Vector((0,0,1))
        index=max(range(3),key=lambda i:abs(axis[i]))
        c=next(c for c in curves(action) if c.data_path=='pose.bones["hips"].location' and c.array_index==index)
        data_points={float(k.co.x):(float(k.co.y),float((k.handle_right.y-k.co.y)/(k.handle_right.x-k.co.x)))
                     for k in c.keyframe_points}
        for f,lift in ((4,.014),(12.5,.017),(38.5,.023)):
            data_points[f]=(c.evaluate(f)+lift*axis[index],(c.evaluate(f+.01)-c.evaluate(f-.01))/.02)
        times=sorted(data_points)
        c.keyframe_points.clear()
        for f in times:c.keyframe_points.insert(f,data_points[f][0],options={'FAST'})
        c.update()
        for i,k in enumerate(c.keyframe_points):
            f=times[i];y,slope=data_points[f]
            left=(f-times[i-1])/3 if i else (times[1]-f)/3
            right=(times[i+1]-f)/3 if i+1<len(times) else (f-times[i-1])/3
            k.interpolation='BEZIER';k.handle_left_type=k.handle_right_type='FREE'
            k.handle_left=(f-left,y-slope*left);k.handle_right=(f+right,y+slope*right)
        c.update()
    rig.animation_data.action_slot = action.slots[0]
    stats[name] = {'frames':length,'duration':length/24,'looping':looping,
                   'keys':sum(len(c.keyframe_points) for c in curves(action)),
                   'distinct_key_times':len({round(k.co.x,5) for c in curves(action) for k in c.keyframe_points})}
    if name in ('walk','walk_fast','run'):
        action['thigh_angle_limit_degrees']={'walk':18,'walk_fast':30,'run':72}[name]
        action['knee_bend_limit_degrees']={'walk':26,'walk_fast':40,'run':80}[name]

# An integer 25-frame loop at 48 fps gives exactly 80% of the approved 10/24 run.
# Other clips use a doubled timebase, preserving their durations and sparse poses.
for action in bpy.data.actions:
    factor=2.5 if action.name=='run' else 2.
    for c in curves(action):
        for k in c.keyframe_points:
            x,left,right=float(k.co.x),float(k.handle_left.x),float(k.handle_right.x)
            k.co.x=1+(x-1)*factor
            k.handle_left.x=1+(left-1)*factor
            k.handle_right.x=1+(right-1)*factor
        c.update()
    action['loop_frames']=round(action['loop_frames']*factor)
    action.frame_end=1+action['loop_frames']
    stats[action.name].update(frames=action['loop_frames'],duration=action['loop_frames']/48)
SCENE.render.fps=48
rig['forward_axis'] = '+X, Blender Z up'
rig['preview_notes'] = '48 fps; idle 1-144; walk 1-64; walk_fast 1-40; run 1-25; jump 1-96; sleep 1-192. Run at 80% speed; sparse key poses retained.'
rig['rig_notes'] = '13 bones. Elbows, wrists and knees welded with blended joint rings; shoulders and hips remain separate. No props.'
SCENE['hero_action_preview_auto'] = True
SCENE['hero_action_preview_rig'] = rig.name

def select_action(name, frame=1):
    action = bpy.data.actions[name]
    rig.animation_data.action = action
    rig.animation_data.action_slot = action.slots[0]
    SCENE.frame_start, SCENE.frame_end = 1, int(action['loop_frames'])
    SCENE.frame_set(frame)
    bpy.context.view_layer.update()

# Save authoring file with the armature active and an Action Editor visible.
select_action('run')
bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='POSE')
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'DOPESHEET_EDITOR':
            area.spaces.active.mode = 'ACTION'
            area.spaces.active.ui_mode = 'ACTION'
            area.spaces.active.dopesheet.show_only_selected = False
        if area.type == 'VIEW_3D':
            area.spaces.active.shading.type = 'MATERIAL'
            area.spaces.active.overlay.show_overlays = False
            area.spaces.active.region_3d.view_distance = 2.6
            area.spaces.active.region_3d.view_location = (.02,.01,.64)
            direction = Vector((3,-4,1.7))
            area.spaces.active.region_3d.view_rotation = direction.to_track_quat('Z','Y')
SCENE.timeline_markers.clear()
SCENE['animation_clips'] = json.dumps(stats)
for name,(length,_,_) in SPECS.items():
    action = bpy.data.actions[name]
    action.asset_mark()
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'hero1_animated.blend'))

# Export a single mesh with the complete skeleton and all five independent clips.
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
mesh.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.export_scene.gltf(
    filepath=str(OUT/'hero1_animated.glb'), export_format='GLB', use_selection=True,
    export_animations=True, export_animation_mode='ACTIONS', export_frame_range=False,
    export_anim_slide_to_zero=True, export_force_sampling=True, export_frame_step=1,
    export_optimize_animation_size=True, export_reset_pose_bones=True,
    export_skins=True, export_def_bones=False, export_leaf_bone=False,
    export_armature_object_remove=True, export_rest_position_armature=True,
    export_morph=False, export_morph_animation=False,
    export_cameras=False, export_lights=False, export_extras=False,
    export_yup=True, export_materials='EXPORT', export_image_format='AUTO',
    export_draco_mesh_compression_enable=False, export_meshopt_compression_enable=False,
)
mesh.select_set(False)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='POSE')
select_action('run')
(OUT/'hero1_animation_report.json').write_text(json.dumps({'bones':len(data.bones),
    'meshes':1,'vertices':len(mesh.data.vertices),'welded_limbs':joint_report,'clips':stats},indent=2))
print('BUILD_COMPLETE',json.dumps(stats))
