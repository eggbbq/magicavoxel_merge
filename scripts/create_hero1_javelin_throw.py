"""Author an exaggerated overarm javelin throw on the existing voxel hero."""
from pathlib import Path
import copy
import json
import math
import bpy
from mathutils import Matrix,Vector,Quaternion
import hero1_work_actions as work
import relax_hero1_wrists as wrists
from refine_hero1_fishing import write_property
from preview_hero1_chop import tool_frame

ROOT=Path(__file__).resolve().parent.parent
NAME='javelin_throw';LENGTH=120;RELEASE_FRAME=43
# FK shoulders and anatomical elbows. No arm stretching or palm compensation.
# Frame, torso pitch/yaw, right elevation/out/twist/flex, left equivalent,
# spear elevation/heading. The empty hands return to a straight standing stance.
ROWS=[(1,0,0,0,3,0,5,0,3,0,5,0,0),
      (13,-6,22,45,25,-20,65,35,8,0,15,10,0),
      (25,-18,56,70,40,-35,95,75,10,0,12,12,0),
      (33,-23,62,75,45,-40,100,85,12,0,8,20,0),
      (39,-5,18,125,22,-15,30,50,12,0,45,24,0),
      (43,22,-30,115,12,0,5,-10,10,0,35,18,-30),
      (49,38,-38,60,12,0,8,-22,15,0,10,-25,0),
      (61,24,-23,25,8,0,10,-10,8,0,8,0,0),
      (77,9,-6,8,3,0,7,2,3,0,7,0,0),
      (93,0,0,0,3,0,5,0,3,0,5,0,0),
      (105,0,0,0,3,0,5,0,3,0,5,0,0),
      (121,0,0,0,3,0,5,0,3,0,5,0,0)]
FRAMES=sorted({row[0] for row in ROWS}|{2,3,5,9,36,41})
# Frame, hips yaw/dx/dy/z, left ankle x/lift, right ankle x/lift.
LEGS=[(1,0,0,0,.300,.06,0,.06,0),
      (13,10,-.015,.012,.265,.08,0,-.04,0),
      (25,22,-.040,.025,.235,.18,.055,-.08,0),
      (33,26,-.025,.025,.230,.26,.10,-.08,0),
      (39,12,.040,.010,.225,.31,0,-.08,0),
      (43,-14,.105,-.005,.230,.31,0,.0,.065),
      (49,-22,.170,-.015,.225,.31,0,.02,.13),
      (61,-18,.140,-.010,.245,.31,0,.08,.06),
      (77,-6,.060,0,.280,.18,0,.10,.015),
      (93,0,0,0,.298,.06,0,.06,0),
      (105,0,0,0,.300,.06,0,.06,0),
      (121,0,0,0,.300,.06,0,.06,0)]
STANDING=None


def lower_body(frame):
    r=work.rig();m=work.mesh()
    yaw,dx,dy,height,lx,lift_l,rx,lift_r=work.interpolate(LEGS,frame)
    hip=r.pose.bones['hips'];work.rotate('hips',yaw=yaw)
    hip.location=hip.bone.matrix_local.to_quaternion().inverted()@Vector((dx,dy,height-hip.bone.head_local.z))
    bpy.context.view_layer.update()
    forward=Quaternion((0,0,1),math.radians(yaw))@Vector((1,0,0))
    stance=work.interpolate([(1,0),(25,1),(61,1),(93,0),(121,0)],frame)[0]
    for side,x,y,lift in (('L',lx,-.10-.06124591*stance,lift_l),('R',rx,.12+.06124591*stance,lift_r)):
        upper,lower=(r.pose.bones[n+'.'+side] for n in ('thigh','shin'))
        target=Vector((x,y,.0108+lift))
        ids=[v.index for v in m.data.vertices if v.co.z<.021 and any(m.vertex_groups[g.group].name=='shin.'+side for g in v.groups)]
        for _ in range(6):
            start=upper.head.copy();direction=(target-start).normalized();distance=(target-start).length
            l1,l2=upper.bone.length,lower.bone.length
            assert abs(l1-l2)+.00005<distance<l1+l2-.00005,('Leg unreachable',frame,side,distance)
            pole=(forward-direction*forward.dot(direction)).normalized()
            cosine=max(-1,min(1,(l1*l1+distance*distance-l2*l2)/(2*l1*distance)))
            knee=start+direction*l1*cosine+pole*l1*math.sqrt(max(0,1-cosine*cosine))
            u=(knee-start).normalized();v=(target-knee).normalized();z=u.cross(v).normalized()
            for p,axis in ((upper,u),(lower,v)):
                mat=Matrix((axis.cross(z).normalized(),axis,z)).transposed().to_4x4()
                mat.translation=p.head.copy();p.matrix=mat;bpy.context.view_layer.update()
            ev=m.evaluated_get(bpy.context.evaluated_depsgraph_get())
            correction=.004+lift-min(ev.data.vertices[i].co.z for i in ids)
            if abs(correction)<1e-6:break
            target.z+=correction
    return yaw


def pose(frame,previous):
    r=work.rig();work.reset()
    pitch,yaw,*arms=work.interpolate(ROWS,frame)
    hip_yaw=lower_body(frame) if 1<frame<105 else 0
    work.rotate('body',pitch,yaw=yaw-hip_yaw)
    work.rotate('head',-pitch*.4,yaw=-yaw*.5)
    for side,params in (('R',arms[:4]),('L',arms[4:8])):
        elevation,out,twist,flex=params
        work.rotate('upper_arm.'+side,-elevation,roll=out if side=='R' else -out)
        u=r.pose.bones['upper_arm.'+side]
        u.rotation_quaternion=u.rotation_quaternion@Quaternion((0,1,0),math.radians(twist))
        r.pose.bones['forearm.'+side].rotation_quaternion=Quaternion((0,0,1),-math.radians(flex))
        r.pose.bones['hand.'+side].rotation_quaternion=Quaternion()
    # Complete the return with a short blend into the standing pose.
    blend=max(0.,min(1.,(frame-77)/28));blend=blend*blend*(3-2*blend)
    if frame==1:blend=1.
    for p in r.pose.bones:
        if blend:
            idle=STANDING[p.name]
            p.location=p.location.lerp(Vector(idle['location']),blend)
            p.rotation_quaternion=p.rotation_quaternion.slerp(Quaternion(idle['rotation_quaternion']),blend)
            p.scale=p.scale.lerp(Vector(idle['scale']),blend)
        for key in ('authored_length_ratio','authored_palm_scale_x','authored_palm_scale_y','authored_palm_scale_z'):
            if key in p:p[key]=1.
    bpy.context.view_layer.update()
    if 1<frame<105:
        ev=work.mesh().evaluated_get(bpy.context.evaluated_depsgraph_get())
        correction=.004*(1-blend)-min(v.co.z for v in ev.data.vertices)
        hip=r.pose.bones['hips'];hip.location+=hip.bone.matrix_local.to_quaternion().inverted()@Vector((0,0,correction))
        bpy.context.view_layer.update()
    h=r.pose.bones['hand.R'];target,_=tool_frame(*arms[-2:]);target.translation=(h.head+h.tail)*.5
    r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update()
    raw={}
    for side in ('L','R'):
        raw['forearm.'+side]={'authored_length_ratio':1.}
        raw['hand.'+side]={'authored_palm_scale_'+axis:1. for axis in 'xyz'}
    values={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),'scale':tuple(p.scale)} for p in r.pose.bones}
    return values,raw,previous,0.


def add_javelin():
    r=work.rig()
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    obj=bpy.data.objects.get('PREVIEW_Javelin')
    if obj:return obj
    vs=[];fs=[];materials=[]
    def prism(outline,width,material):
        i=len(vs);n=len(outline);vs.extend([(x,y,z) for z in (-width,width) for x,y in outline])
        fs.extend([tuple(i+j for j in reversed(range(n))),tuple(i+n+j for j in range(n))])
        fs.extend([(i+j,i+(j+1)%n,i+(j+1)%n+n,i+j+n) for j in range(n)]);materials.extend([material]*(n+2))
    prism([(-.009,-.72),(.009,-.72),(.009,.75),(-.009,.75)],.009,0)
    prism([(-.015,-.09),(.015,-.09),(.015,.09),(-.015,.09)],.015,1)
    prism([(-.038,.72),(0,.94),(.038,.72),(0,.76)],.012,2)
    prism([(-.013,-.76),(.013,-.76),(.013,-.70),(-.013,-.70)],.013,2)
    data=bpy.data.meshes.new('Preview javelin geometry');data.from_pydata(vs,[],fs);data.update()
    obj=bpy.data.objects.new('PREVIEW_Javelin',data);bpy.context.scene.collection.objects.link(obj)
    for name,color in [('Shaft',(.30,.13,.04,1)),('Grip',(.075,.055,.04,1)),('Tip',(.50,.60,.66,1))]:
        mat=bpy.data.materials.get('PREVIEW_Javelin_'+name) or bpy.data.materials.new('PREVIEW_Javelin_'+name)
        mat.use_nodes=True;mat.diffuse_color=color
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value=color;shader.inputs['Roughness'].default_value=.6
        data.materials.append(mat)
    for p,i in zip(data.polygons,materials):p.material_index=i
    obj.parent=r;obj.parent_type='BONE';obj.parent_bone='tool.R'
    obj.matrix_parent_inverse=Matrix.Identity(4);obj.location=(0,-r.data.bones['tool.R'].length,0)
    obj['preview_only']=True;obj['preview_action']=NAME;obj['remove_after_user_approval']=True
    obj['attachment_socket']='tool.R';obj['preview_projectile']=True
    obj['release_frame']=RELEASE_FRAME;obj['flight_end_frame']=61.;obj['reappear_frame']=LENGTH+2.
    obj['flight_speed']=7.;obj['flight_gravity']=9.8
    return obj


def fit_tool(a):
    r=work.rig();targets={};desired={};last=None
    frames=sorted({1+k*.25 for k in range(LENGTH*4+1)}|set(FRAMES))
    for f in frames:
        if f>RELEASE_FRAME:continue
        wrists.select(r,a,f);target,axis=tool_frame(*work.interpolate(ROWS,f)[-2:])
        h=r.pose.bones['hand.R'];target.translation=(h.head+h.tail)*.5
        targets[f]=target;r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update()
        p=r.pose.bones['tool.R'];q=p.rotation_quaternion.copy()
        reference=last
        if reference and q.dot(reference)<0:q.negate()
        last=q;desired[f]={'tool.R':dict(location=tuple(p.location),rotation_quaternion=tuple(q),scale=tuple(p.scale))}
    # The released weapon is independent: hold the empty socket's local pose.
    start=desired[float(RELEASE_FRAME)]['tool.R']
    for f in FRAMES:
        if f>RELEASE_FRAME:desired[float(f)]={'tool.R':copy.deepcopy(start)}
    keys=set(float(f) for f in FRAMES)
    for _ in range(12):
        times=sorted(keys);samples=[desired[f] for f in times]
        wrists.write_samples(r,a,times,samples,['tool.R'])
        worst={};maxpos=maxangle=0
        for f,target in targets.items():
            wrists.select(r,a,f);p=r.pose.bones['tool.R'];distance=(p.head-target.translation).length
            angle=p.matrix.to_quaternion().rotation_difference(target.to_quaternion()).angle;angle=min(angle,abs(math.tau-angle))
            maxpos=max(maxpos,distance);maxangle=max(maxangle,angle);error=max(distance/.0004,angle/math.radians(.6))
            if error>1.001 and f not in keys:
                interval=next((i for i in range(len(times)-1) if times[i]<f<times[i+1]),None)
                if interval is not None and (interval not in worst or error>worst[interval][0]):worst[interval]=(error,f)
        if not worst:break
        keys.update(f for _,f in worst.values())
    assert maxpos<.00041 and maxangle<math.radians(.61),(maxpos,maxangle)
    return len(keys)


def build(backup_name='hero1_before_javelin_throw.blend'):
    global STANDING
    r=work.rig();s=bpy.context.scene
    assert len(r.data.bones)==18 and s.get('hero_arm_width_fixed_v1')
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/backup_name
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    snapshot=lambda a:[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points]) for c in work.curves(a)]
    preserved={a.name:snapshot(a) for a in bpy.data.actions if a.name!=NAME}
    work.set_action('idle',1)
    for side in ('L','R'):work.rotate('upper_arm.'+side,roll=3 if side=='R' else -3)
    bpy.context.view_layer.update()
    STANDING={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),'scale':tuple(p.scale)} for p in r.pose.bones}
    old=bpy.data.actions.get(NAME);r.animation_data.action=None
    drivers=list(r.animation_data.drivers);driven={(c.data_path,c.array_index) for c in drivers}
    samples=[];custom=[];previous={};reach=0
    try:
        for c in drivers:c.mute=True
        for f in FRAMES:
            values,raw,previous,e=pose(f,previous);samples.append(values);custom.append(raw);reach=max(reach,e)
    finally:
        for c in drivers:c.mute=False
    assert reach<.001,('Javelin hand unreachable',reach)
    for i in range(1,len(samples)):
        for n in samples[i]:
            q=Quaternion(samples[i][n]['rotation_quaternion'])
            if q.dot(Quaternion(samples[i-1][n]['rotation_quaternion']))<0:q.negate()
            samples[i][n]['rotation_quaternion']=tuple(q)
    a=bpy.data.actions.new(NAME+'_build');a.use_fake_user=True;a.use_frame_range=True
    a.frame_start=1;a.frame_end=LENGTH+1;a['loop_frames']=LENGTH;a['looping']=False
    a['description']='自然站立、后拉蓄力、跨步转髋投掷、随挥放臂、回到自然站立'
    a['ends_relaxed']=True;a['standing_frame']=105;a['arm_authoring']='FK; unit lengths; neutral wrists'
    a['author_pose_times']=len(FRAMES);a['release_frame']=RELEASE_FRAME
    r.animation_data.action=a;wrists.write_samples(r,a,FRAMES,samples,list(r.pose.bones.keys()))
    for layer in a.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for c in list(bag.fcurves):
                    if (c.data_path,c.array_index) in driven:bag.fcurves.remove(c)
    for bone,props in custom[0].items():
        p=r.pose.bones[bone]
        for key,value in props.items():p[key]=value;p.keyframe_insert('["'+key+'"]',frame=1,group=bone)
    for c in work.curves(a):
        if c.data_path.endswith(']'):
            bone,key=c.data_path.split('"')[1::2];write_property(c,FRAMES,[v[bone][key] for v in custom],False)
    tool_times=fit_tool(a)
    for c in work.curves(a):
        ys=[k.co.y for k in c.keyframe_points]
        if len(ys)>1 and max(ys)-min(ys)<2e-6:
            c.keyframe_points.clear();c.keyframe_points.insert(1,ys[0]);c.update()
    if old:bpy.data.actions.remove(old)
    a.name=NAME;a.asset_mark()
    assert all(snapshot(bpy.data.actions[n])==v for n,v in preserved.items())
    report=dict(frames=LENGTH,fps=s.render.fps,duration=LENGTH/s.render.fps,looping=False,
                release_frame=RELEASE_FRAME,distinct_key_times=len(FRAMES),tool_pose_times=tool_times,
                keys=sum(len(c.keyframe_points) for c in work.curves(a)),max_grip_reach_error_m=reach,preserved_actions=list(preserved))
    stats=json.loads(s.get('animation_clips','{}'));stats[NAME]=report;s['animation_clips']=json.dumps(stats)
    prop=add_javelin();prop['release_frame']=RELEASE_FRAME;prop['flight_end_frame']=61.;prop['reappear_frame']=LENGTH+2.;work.set_action(NAME,RELEASE_FRAME)
    tool=r.pose.bones['tool.R'];prop['release_position']=list(r.matrix_world@tool.head)
    prop['release_elevation'],prop['release_heading']=work.interpolate(ROWS,RELEASE_FRAME)[-2:]
    from simplify_hero1_javelin_keys import reduce
    report['key_reduction']=reduce();report['tool_pose_times']=report['key_reduction']['after']['tool_times'];report['keys']=report['key_reduction']['after']['keys']
    work.set_action(NAME,RELEASE_FRAME);prop['release_position']=list(r.matrix_world@r.pose.bones['tool.R'].head)
    matrix=r.matrix_world@r.pose.bones['tool.R'].matrix
    prop['release_direction']=list((matrix.to_3x3()@Vector((0,1,0))).normalized())
    prop['release_rotation']=list(matrix.to_quaternion())
    work.set_action(NAME,1)
    (ROOT/'hero1_javelin_throw_report.json').write_text(json.dumps(report,indent=2))
    print('JAVELIN_THROW_CREATED',json.dumps(report));return report


if __name__=='__main__':build()
