"""Add sparse fishing, pulling, chopping and tilling Actions to the approved 48-fps hero.

Import in live Blender, call build_actions(), add_test_rod(), then inspect.
Call finish() after the grip test: it removes the temporary rod and saves/exports the actor.
"""
from pathlib import Path
import copy
import json
import math
import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT=Path('/Users/graylian/workspace/magicavoxel_merge')
NEW_NAMES=('fishing','fishing_pull','chop','till')
RAD=math.pi/180
TILL_ELBOW_POLE=(.55,1.8,-.8)

def rig():return bpy.data.objects['Hero']
def mesh():return bpy.data.objects['HeroMesh']
def curves(a):return [c for l in a.layers for s in l.strips for b in s.channelbags for c in b.fcurves]

def qrot(pitch=0,roll=0,yaw=0):
    return Quaternion((0,0,1),yaw*RAD)@Quaternion((0,1,0),pitch*RAD)@Quaternion((1,0,0),roll*RAD)

def rotate(name,pitch=0,roll=0,yaw=0):
    p=rig().pose.bones[name];r=p.bone.matrix_local.to_quaternion()
    p.rotation_quaternion=r.inverted()@qrot(pitch,roll,yaw)@r

def reset():
    for p in rig().pose.bones:
        p.rotation_mode='QUATERNION';p.location=(0,0,0);p.rotation_quaternion=(1,0,0,0);p.scale=(1,1,1)

def set_world_direction(p,direction,stretch=1):
    orientation=Vector((0,0,-1)).rotation_difference(Vector(direction).normalized())@p.bone.matrix_local.to_quaternion()
    # Uniform segment scale avoids inherited nonuniform shear at the elbow.
    # The palm is solved at unit world scale and therefore retains its size.
    matrix=orientation.to_matrix().to_4x4()@Matrix.Diagonal((stretch,stretch,stretch,1));matrix.translation=p.head.copy()
    p.matrix=matrix
    bpy.context.view_layer.update()

def solve_arm(side,grip,axis,stretch=1,bend_lift=0,hinge=False,elbow_pole=None):
    r=rig();upper=r.pose.bones['upper_arm.'+side];lower=r.pose.bones['forearm.'+side];hand=r.pose.bones['hand.'+side]
    axis=Vector(axis).normalized();target=Vector(grip)-axis*(hand.bone.length*.5)
    start=upper.head.copy();delta=target-start;distance=delta.length
    l1,l2=upper.bone.length*stretch,lower.bone.length*stretch
    reached=min(l1+l2-.0005,max(abs(l1-l2)+.0005,distance))
    direction=delta.normalized()
    cosine=max(-1,min(1,(l1*l1+reached*reached-l2*l2)/(2*l1*reached)))
    # Raised tool poses lift the upper arms and put the elbows high and outside.
    # Low tool poses keep the elbows forward, away from the dress.
    if hinge:
        # A consistent anatomical elbow plane: elbows stay outside and below the
        # wrists rather than flipping forward/up as the tool crosses horizontal.
        body=r.pose.bones['body']
        torso_rotation=body.matrix.to_quaternion()@body.bone.matrix_local.to_quaternion().inverted()
        px,py,pz=elbow_pole or TILL_ELBOW_POLE
        pole=torso_rotation@Vector((px,py if side=='R' else -py,pz))
    else:
        pole=Vector((1-bend_lift,(.9 if side=='R' else -.9)*bend_lift,1.4*bend_lift))
    pole=(pole-direction*pole.dot(direction)).normalized()
    elbow=start+direction*(l1*cosine)+pole*(l1*math.sqrt(max(0,1-cosine*cosine)))
    target_actual=start+direction*reached
    if hinge:
        u=(elbow-start).normalized();v=(target_actual-elbow).normalized()
        z=-u.cross(v).normalized()
        def set_hinge_frame(pb,y):
            x=y.cross(z).normalized()
            frame=Matrix((x,y,z)).transposed().to_4x4()@Matrix.Diagonal((stretch,stretch,stretch,1))
            frame.translation=pb.head.copy();pb.matrix=frame;bpy.context.view_layer.update()
        set_hinge_frame(upper,u)
        set_hinge_frame(lower,v)
    else:
        set_world_direction(upper,elbow-start,stretch)
        set_world_direction(lower,target_actual-lower.head,stretch)
    set_world_direction(hand,axis)
    actual=(hand.head+hand.tail)*.5
    return (actual-Vector(grip)).length

def interpolate(rows,t):
    for a,b in zip(rows,rows[1:]):
        if t<=b[0]+1e-8:
            u=max(0,min(1,(t-a[0])/(b[0]-a[0])));u=u*u*(3-2*u)
            return [a[i]+(b[i]-a[i])*u for i in range(1,len(a))]
    return list(rows[-1][1:])

# phase, body pitch, yaw, knee bend, right-hand grip x/y/z, shaft elevation.
PULL=[(0,8,0,8,.242,.01,.65,35),
      (.12,10,0,12,.257,.01,.65,25),
      (.22,12,0,16,.267,.01,.65,20),
      (.30,-4,0,12,.155,.01,.80,65),
      (.40,-10,0,10,.12,.01,.85,80),
      (.55,-8,0,12,.13,.01,.82,74),
      (.68,3,0,10,.235,.01,.76,58),
      (.85,7,0,8,.242,.01,.67,39),
      (1,8,0,8,.242,.01,.65,35)]
CHOP=[(0,6,0,12,.22,.01,.72,68),
      (.18,3,-1,16,.275,.0,.79,78),
      (.34,5,0,18,.29,.0,.82,78),
      (.43,6,0,16,.30,.0,.80,75),
      (.50,26,4,22,.40,.01,.63,52),
      (.55,33,6,26,.43,.01,.61,50),
      (.70,24,3,20,.39,.01,.63,56),
      (.86,10,0,14,.24,.01,.69,66),
      (1,6,0,12,.22,.01,.72,68)]
TILL=[(0,12,0,12,.27,.01,.60,-28),
      (.17,4,0,12,.245,.01,.79,35),
      (.34,-18,0,10,.095,.01,1.24,65),
      (.44,-32,0,16,.010,.01,1.18,80),
      (.53,36,3,24,.445,.01,.57,-52),
      (.58,42,3,28,.475,.01,.575,-51),
      (.73,31,0,22,.390,.01,.55,-47),
      (.90,15,0,15,.29,.01,.61,-28),
      (1,12,0,12,.27,.01,.60,-28)]
SPECS={'fishing':(144,True,[i/8 for i in range(9)]),
       'fishing_pull':(64,False,[r[0] for r in PULL]),
       'chop':(56,True,[r[0] for r in CHOP]),
       # Sparse transition poses keep both palms on the handle during the large swing.
       'till':(72,True,sorted([r[0] for r in TILL]+[.085,.25,.295,.40,.4625,.485,.5075]))}

def pose(name,t):
    reset()
    # Keep the original inheritance used by the approved breathing/locomotion clips.
    for side in ('L','R'):
        rig().data.bones['forearm.'+side].inherit_scale='FULL'
        rig().data.bones['hand.'+side].inherit_scale='FULL'
    if name=='fishing':
        breath=math.sin(math.tau*t)
        pitch,yaw,knee=8+.5*breath,0,8
        gx,gy,gz,angle=.242+.002*breath,.01,.65+.003*breath,35+.8*breath
    else:
        pitch,yaw,knee,gx,gy,gz,angle=interpolate({'fishing_pull':PULL,'chop':CHOP,'till':TILL}[name],t)
    # A small, planted stance: no leg translation or unneeded deep crouch.
    half=knee*RAD*.5
    height=.3*math.cos(half)+.02*math.sin(half)+.002
    r=rig();hip=r.pose.bones['hips'];rest=hip.bone.matrix_local.to_quaternion()
    hip.location=rest.inverted()@Vector((0,0,height-.3))
    rotate('body',pitch,yaw=yaw)
    rotate('head',-pitch*.55,yaw=-yaw*.5)
    for side in ('L','R'):
        rotate('thigh.'+side,-knee*.5);rotate('shin.'+side,knee)
    bpy.context.view_layer.update()
    axis=Vector((math.cos(angle*RAD),0,math.sin(angle*RAD)))
    base=Vector((gx,gy,gz))
    overhead=max(0,min(1,(angle-20)/45)) if name=='till' else 0
    overhead=overhead*overhead*(3-2*overhead)
    stretch=1.25+.65*overhead if name=='till' else 1
    spacing=.08-.04*overhead
    if name=='till':base.x+=.025
    lift=max(0,min(1,(angle-20)/50)) if name=='till' else 0
    lift=lift*lift*(3-2*lift)
    # The lower grip is closer to its shoulder; slightly less extension keeps
    # both elbows comparably open instead of leaving the right arm folded.
    right_stretch=stretch*(1-.06*overhead) if name=='till' else stretch
    errors=[solve_arm('R',base,axis,right_stretch,lift,name=='till'),solve_arm('L',base+axis*spacing,axis,stretch,lift,name=='till')]
    from relax_hero1_wrists import relax_current_pose
    relax_current_pose(r)
    result={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),'scale':tuple(p.scale)} for p in r.pose.bones}
    return result,max(errors)

def set_action(name,frame=1):
    r=rig();a=bpy.data.actions[name]
    r.animation_data.action=a;r.animation_data.action_slot=a.slots[0]
    s=bpy.context.scene;s.frame_start=s.frame_preview_start=1
    s.frame_end=s.frame_preview_end=int(a['loop_frames']);s.use_preview_range=True
    s.frame_set(math.floor(frame),subframe=frame%1);bpy.context.view_layer.update()

def build_actions():
    s=bpy.context.scene;r=rig()
    assert s.render.fps==48 and int(bpy.data.actions['run']['loop_frames'])==25
    assert not any(n in bpy.data.actions for n in NEW_NAMES),'Work Actions already exist; preserve or remove them explicitly first.'
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero1_before_work_actions.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    old_names=[a.name for a in bpy.data.actions]
    old_values={a.name:[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points]) for c in curves(a)] for a in bpy.data.actions}
    report={}
    for name,(length,looping,phases) in SPECS.items():
        r.animation_data.action=None
        samples=[];reach=0
        for t in phases:
            p,e=pose(name,t);samples.append(p);reach=max(reach,e)
        # Choose quaternion hemispheres consistently before sparse interpolation.
        for i in range(1,len(samples)):
            for bone in samples[i]:
                q=Quaternion(samples[i][bone]['rotation_quaternion']);prev=Quaternion(samples[i-1][bone]['rotation_quaternion'])
                if q.dot(prev)<0:q.negate()
                samples[i][bone]['rotation_quaternion']=tuple(q)
        samples[-1]=copy.deepcopy(samples[0])
        a=bpy.data.actions.new(name);a.use_fake_user=True;a.use_frame_range=True
        a.frame_start=1;a.frame_end=length+1;a['loop_frames']=length;a['looping']=looping
        a['description']={'fishing':'垂钓等待，双手握杆','fishing_pull':'垂钓拉杆、后仰收力、恢复','chop':'伐木，抬起、蓄力、劈下、回收','till':'翻土，抬锄、下压、回拉、复位'}.get(name,name)
        r.animation_data.action=a
        for pb in r.pose.bones:
            for channel,value in samples[0][pb.name].items():
                setattr(pb,channel,value);pb.keyframe_insert(channel,frame=1,group=pb.name)
        times=[1+t*length for t in phases]
        for c in curves(a):
            bone=c.data_path.split('"')[1];channel=c.data_path.rsplit('.',1)[-1];index=c.array_index
            ys=[p[bone][channel][index] for p in samples]
            if max(ys)-min(ys)<1e-7:
                if bone=='hips' and channel=='rotation_quaternion' and index==0:c.keyframe_points.insert(length+1,ys[-1])
                continue
            def slope(i):
                if i in (0,len(times)-1):
                    if not looping:return 0.
                    h0,h1=times[-1]-times[-2],times[1]-times[0]
                    d0,d1=(ys[-1]-ys[-2])/h0,(ys[1]-ys[0])/h1
                else:
                    h0,h1=times[i]-times[i-1],times[i+1]-times[i]
                    d0,d1=(ys[i]-ys[i-1])/h0,(ys[i+1]-ys[i])/h1
                if d0*d1<=0:return 0.
                w0,w1=2*h1+h0,h1+2*h0
                return (w0+w1)/(w0/d0+w1/d1)
            c.keyframe_points.clear()
            for f,y in zip(times,ys):c.keyframe_points.insert(f,y,options={'FAST'})
            c.update()
            for i,k in enumerate(c.keyframe_points):
                f,y=times[i],ys[i];d=slope(i)
                left=(f-times[i-1])/3 if i else (times[-1]-times[-2])/3
                right=(times[i+1]-f)/3 if i+1<len(times) else (times[1]-times[0])/3
                k.interpolation='BEZIER';k.handle_left_type=k.handle_right_type='FREE'
                k.handle_left=(f-left,y-d*left);k.handle_right=(f+right,y+d*right)
            c.update()
        a.asset_mark()
        report[name]={'frames':length,'duration':length/48,'looping':looping,
                      'distinct_key_times':len(times),'keys':sum(len(c.keyframe_points) for c in curves(a)),
                      'key_pose_grip_reach_error':reach}
    for name in old_names:
        now=[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points]) for c in curves(bpy.data.actions[name])]
        assert now==old_values[name],('Existing Action changed',name)
    stats=json.loads(s.get('animation_clips','{}'));stats.update(report);s['animation_clips']=json.dumps(stats)
    work_stats={n:stats[n] for n in ('fishing','fishing_pull','chop','till') if n in stats}
    (ROOT/'hero1_work_actions_report.json').write_text(json.dumps({'new_actions':work_stats,'preserved_actions':old_names},indent=2))
    set_action('fishing')
    print('WORK_ACTIONS_CREATED',json.dumps(report))
    return report

def add_test_rod():
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    vertices=[];faces=[]
    def box(x0,x1,y0,y1,z0,z1):
        i=len(vertices)
        vertices.extend([(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)])
        faces.extend([tuple(i+j for j in f) for f in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]])
    box(-.010,.010,-.14,.09,-.010,.010)
    for y0,y1,w in ((.09,.40,.007),(.40,.72,.005),(.72,1.05,.003)):
        box(-w,w,y0,y1,-w,w)
    box(.009,.045,-.08,-.025,-.027,.027)
    box(.040,.065,-.063,-.05,-.006,.006)
    data=bpy.data.meshes.new('Temporary fishing rod geometry');data.from_pydata(vertices,[],faces);data.update()
    obj=bpy.data.objects.new('TEMP_FishingRod',data);bpy.context.scene.collection.objects.link(obj)
    material=bpy.data.materials.new('TEMP_FishingRod_Wood');material.use_nodes=True
    shader=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value=(.20,.075,.018,1)
    shader.inputs['Roughness'].default_value=.7
    obj.data.materials.append(material)
    obj.parent=rig();obj.parent_type='BONE';obj.parent_bone='hand.R'
    obj.matrix_parent_inverse=Matrix.Identity(4);obj.location=(0,-rig().data.bones['hand.R'].length*.5,0)
    bpy.context.view_layer.update()
    assert (obj.matrix_world.translation-rig().matrix_world@((rig().pose.bones['hand.R'].head+rig().pose.bones['hand.R'].tail)*.5)).length<1e-5
    print('TEMP_ROD_ATTACHED','hand.R')
    return obj

def check_grips():
    result={}
    for name in ('fishing','fishing_pull'):
        length=int(bpy.data.actions[name]['loop_frames']);error=0
        for k in range(33):
            set_action(name,1+length*k/32)
            r=rig();right=r.pose.bones['hand.R'];left=r.pose.bones['hand.L']
            base=(right.head+right.tail)*.5;tip=(left.head+left.tail)*.5
            axis=(right.tail-right.head).normalized();delta=tip-base
            error=max(error,(delta-axis*.08).length)
        result[name]=error
        assert error<.02,(name,'hand outside the rod grip',error)
    (ROOT/'hero1_fishing_grip_validation.json').write_text(json.dumps(result,indent=2))
    print('TWO_HAND_GRIP_MAX_ERROR',result)
    return result

def finish():
    for obj in list(bpy.data.objects):
        if obj.name.startswith('TEMP_FishingRod'):bpy.data.objects.remove(obj,do_unlink=True)
    s=bpy.context.scene;r=rig();m=mesh()
    assert len(s.objects)==2
    set_action('fishing')
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);bpy.context.view_layer.objects.active=r
    bpy.ops.object.mode_set(mode='POSE')
    r['preview_notes']='48 fps; Actions: idle, walk, walk_fast, run, jump, sleep, fishing, fishing_pull, chop, till. Preview range follows the selected Action.'
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'hero1_animated.blend'))
    bpy.ops.object.mode_set(mode='OBJECT');m.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(ROOT/'hero1_animated.glb'),export_format='GLB',use_selection=True,
        export_animations=True,export_animation_mode='ACTIONS',export_frame_range=False,
        export_anim_slide_to_zero=True,export_force_sampling=True,export_frame_step=1,
        export_optimize_animation_size=True,export_reset_pose_bones=True,export_skins=True,
        export_def_bones=False,export_leaf_bone=False,export_armature_object_remove=True,
        export_rest_position_armature=True,export_morph=False,export_morph_animation=False,
        export_cameras=False,export_lights=False,export_extras=False,export_yup=True,
        export_materials='EXPORT',export_image_format='AUTO',export_draco_mesh_compression_enable=False,
        export_meshopt_compression_enable=False)
    m.select_set(False);bpy.ops.object.mode_set(mode='POSE');set_action('fishing')
    p=ROOT/'hero1_animation_report.json'
    report=json.loads(p.read_text());report['clips']=json.loads(s['animation_clips']);p.write_text(json.dumps(report,indent=2))
    print('WORK_ACTIONS_SAVED_ACTOR_ONLY',[a.name for a in bpy.data.actions])

if __name__=='__main__':
    build_actions();add_test_rod();check_grips();finish()
