"""Revise chop around a two-handed axe; retain the axe for user preview.

Author poses are solved with uniform virtual segment scales, then converted
to the existing length-only rig before keys are written. Other clips, mesh,
weights, and the previously approved hoe are preserved.
"""
from pathlib import Path
import copy
import json
import math
import bpy
from mathutils import Matrix, Vector, Quaternion
import hero1_work_actions as work
import relax_hero1_wrists as wrists

ROOT = Path(__file__).resolve().parent.parent
# phase, torso pitch/yaw, knee, right/front-hand x/y/z, shaft elevation/heading.
# Stop at the front trunk, hold the embedded edge, then pull it back out.
ROWS = [(0,6,14,16,.30,.075,.60,0,75),
        (.18,-4,38,16,.24,.140,.62,0,130),
        (.31,-14,74,16,.08,.240,.64,0,205),
        (.355,5,25,16,.36,.040,.605,0,125),
        (.40,24,-24,16,.30,-.20,.57,0,45),
        (.69,24,-24,16,.30,-.20,.57,0,45),
        (.78,-4,-22,16,.235,-.14,.59,0,45),
        (.84,-4,0,16,.27,-.10,.60,0,55),
        (.94,4,10,16,.31,.035,.60,0,70),
        (1,6,14,16,.30,.075,.60,0,75)]
PHASES = sorted({r[0] for r in ROWS} | {.355,.735})
LENGTH = 84
GRIP_HEIGHT_OFFSET = -.04
GRIP_FORWARD_OFFSET = .06
CONTACT_HOLD = (.40,.69)
TRUNK_FRONT = .805+GRIP_FORWARD_OFFSET
GRIP_SPACING = .193
ARM_LENGTH_FACTOR = 1.8
LOWER_BODY = [(0,0,0,0,0),(.18,8,-.005,.012,-.002),
              (.31,12,-.010,.020,-.003),(.355,2,.003,0,-.005),
              (.40,-10,.010,-.018,-.008),(.69,-10,.010,-.018,-.008),
              (.78,-6,-.006,-.008,-.003),(.84,0,-.004,0,0),
              (.94,0,0,0,0),(1,0,0,0,0)]


def lower_body_pose(t):
    r=work.rig();mesh=work.mesh()
    yaw,dx,dy,dz=work.interpolate(LOWER_BODY,t)
    hip=r.pose.bones['hips'];work.rotate('hips',yaw=yaw)
    hip.location=hip.bone.matrix_local.to_quaternion().inverted()@Vector((dx,dy,.29948777+dz-.30))
    bpy.context.view_layer.update()
    forward=Quaternion((0,0,1),math.radians(yaw))@Vector((1,0,0))
    for side,y in (('L',-.16124591),('R',.18124591)):
        upper,lower=(r.pose.bones[name+'.'+side] for name in ('thigh','shin'))
        target=Vector((.06188753,y,.0108))
        feet=[v.index for v in mesh.data.vertices if v.co.z<.021 and any(mesh.vertex_groups[g.group].name=='shin.'+side for g in v.groups)]
        for _ in range(6):
            start=upper.head.copy();direction=(target-start).normalized()
            length=(target-start).length;l1,l2=upper.bone.length,lower.bone.length
            assert length<l1+l2-.00005,('Leg overextends',t,side,length)
            pole=(forward-direction*forward.dot(direction)).normalized()
            cosine=max(-1,min(1,(l1*l1+length*length-l2*l2)/(2*l1*length)))
            knee=start+direction*(l1*cosine)+pole*l1*math.sqrt(max(0,1-cosine*cosine))
            u=(knee-start).normalized();v=(target-knee).normalized();z=u.cross(v).normalized()
            for p,axis in ((upper,u),(lower,v)):
                matrix=Matrix((axis.cross(z).normalized(),axis,z)).transposed().to_4x4()
                matrix.translation=p.head.copy();p.matrix=matrix;bpy.context.view_layer.update()
            ev=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
            low=min(ev.data.vertices[i].co.z for i in feet)
            correction=.004-low
            if abs(correction)<1e-6:break
            target.z+=correction
    return yaw


def tool_frame(elevation, heading):
    e,h=math.radians(elevation),math.radians(heading)
    y=Vector((math.cos(e)*math.cos(h),math.cos(e)*math.sin(h),math.sin(e)))
    # Blade points toward decreasing heading, following the horizontal strike.
    x=Vector((-math.sin(h),math.cos(h),0))
    return Matrix((x,y,x.cross(y))).transposed().to_4x4(),y


def shaft_angles(t):
    # The extra hand-steering pose clears the torso without adding a pause
    # halfway through the fast swing of the axe.
    return work.interpolate([row for row in ROWS if row[0]!=.355],t)[-2:]


def add_axe():
    r = work.rig()
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    previous = bpy.data.objects.get('PREVIEW_Axe')
    if previous:
        # The rear hand now holds below the right-hand socket; extend the butt.
        for v in previous.data.vertices[:8]:
            if v.co.y<0:v.co.y=-.22
        previous.data.update()
        return previous
    verts, faces, materials = [], [], []
    def prism(outline, half_width, material):
        # Outline is in local X/Y; thickness runs along Z.
        i, n = len(verts), len(outline)
        verts.extend([(x,y,z) for z in (-half_width,half_width) for x,y in outline])
        faces.extend([tuple(i+j for j in reversed(range(n))), tuple(i+n+j for j in range(n))])
        faces.extend([(i+j,i+(j+1)%n,i+(j+1)%n+n,i+j+n) for j in range(n)])
        materials.extend([material]*(n+2))
    prism([(-.013,-.22),(.013,-.22),(.013,.51),(-.013,.51)],.013,0)
    # Wedge-shaped blade, cutting edge toward local -X (forward/down in swing).
    prism([(-.19,.39),(-.19,.55),(-.07,.53),(.055,.50),(.055,.44),(-.07,.42)],.028,1)
    prism([(-.195,.39),(-.195,.55),(-.17,.546),(-.17,.394)],.030,2)
    data=bpy.data.meshes.new('Preview axe geometry')
    data.from_pydata(verts,[],faces);data.update()
    obj=bpy.data.objects.new('PREVIEW_Axe',data);bpy.context.scene.collection.objects.link(obj)
    for name,color in [('PREVIEW_Axe_Wood',(.22,.085,.025,1)),
                       ('PREVIEW_Axe_Iron',(.20,.25,.28,1)),
                       ('PREVIEW_Axe_Edge',(.58,.65,.67,1))]:
        mat=bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes=True
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value=color
        shader.inputs['Roughness'].default_value=.7
        obj.data.materials.append(mat)
    for p,index in zip(data.polygons,materials):p.material_index=index
    obj.parent=r;obj.parent_type='BONE';obj.parent_bone='tool.R'
    obj.matrix_parent_inverse=Matrix.Identity(4)
    obj.location=(0,-r.data.bones['tool.R'].length,0)
    obj['preview_only']=True;obj['preview_action']='chop'
    obj['remove_after_user_approval']=True;obj['attachment_socket']='tool.R'
    return obj


def add_trunk():
    if bpy.data.objects.get('PREVIEW_Trunk'):
        obj=bpy.data.objects['PREVIEW_Trunk']
        offset=TRUNK_FRONT-float(obj.get('contact_front_x',.805))
        for vertex in obj.data.vertices:vertex.co.x+=offset
        obj.data.update();obj['contact_front_x']=TRUNK_FRONT
        return obj
    # A plain voxel trunk is only a contact reference, never part of the actor GLB.
    x0,x1=TRUNK_FRONT,TRUNK_FRONT+.24;y0,y1=-.12,.14;z0,z1=0,1.38
    vertices=[(x,y,z) for z in (z0,z1) for y in (y0,y1) for x in (x0,x1)]
    faces=[(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
    data=bpy.data.meshes.new('Preview trunk geometry');data.from_pydata(vertices,[],faces);data.update()
    obj=bpy.data.objects.new('PREVIEW_Trunk',data);bpy.context.scene.collection.objects.link(obj)
    for name,color in [('PREVIEW_Trunk_Bark',(.18,.07,.025,1)),('PREVIEW_Trunk_End',(.42,.23,.10,1))]:
        mat=bpy.data.materials.get(name) or bpy.data.materials.new(name);mat.use_nodes=True
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value=color;shader.inputs['Roughness'].default_value=.9
        data.materials.append(mat)
    data.polygons[1].material_index=1
    obj['preview_only']=True;obj['preview_action']='chop';obj['remove_after_user_approval']=True
    obj['contact_front_x']=TRUNK_FRONT
    return obj


def pose(t, previous):
    r=work.rig();work.reset()
    pitch,yaw,knee,gx,gy,gz,elevation,heading=work.interpolate(ROWS,t)
    gz += GRIP_HEIGHT_OFFSET
    gx += GRIP_FORWARD_OFFSET
    hip_yaw=lower_body_pose(t)
    # Keep the approved world-space shoulder swing while the pelvis follows it.
    work.rotate('body',pitch,yaw=yaw-hip_yaw)
    work.rotate('head',-pitch*.5,yaw=-yaw*.45)
    bpy.context.view_layer.update()
    target,axis=tool_frame(elevation,heading)
    base=Vector((gx,gy,gz))
    # Right hand near the head, left hand toward the butt. Keeping that order
    # relative to the shoulder line prevents the old crossed-arm grip.
    errors=[work.solve_arm('R',base,axis,ARM_LENGTH_FACTOR,hinge=True),
            work.solve_arm('L',base-axis*GRIP_SPACING,axis,ARM_LENGTH_FACTOR,hinge=True)]
    target.translation=base
    r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update()
    wrist_errors,previous=wrists.relax_current_pose(r,previous)
    # Convert the solved uniform-scale pose to the permanent width-preserving rig.
    raw={}
    for side in ('L','R'):
        u,f,h=(r.pose.bones[n+'.'+side] for n in ('upper_arm','forearm','hand'))
        U,F,H=u.scale.y,f.scale.y,h.scale.copy()
        raw['forearm.'+side]={'authored_length_ratio':F}
        raw['hand.'+side]={'authored_palm_scale_'+axis:H[i] for i,axis in enumerate('xyz')}
        u.scale=(1,U,1)
        r.pose.bones['elbow_scale.'+side].scale=(1,1/U,1)
        f.scale=(1,U*F,1)
        r.pose.bones['wrist_scale.'+side].scale=(1,1/(U*F),1)
        h.scale=H*(U*F)
        for bone,values in [(f,raw[f.name]),(h,raw[h.name])]:
            for key,value in values.items():bone[key]=value
    bpy.context.view_layer.update()
    values={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),
                    'scale':tuple(p.scale)} for p in r.pose.bones}
    return values,raw,previous,max(errors+list(wrist_errors.values()))


def build(backup_name='hero1_before_lower_forward_chop.blend', sparse_attachment=True, fast_rhythm=True):
    s=bpy.context.scene;r=work.rig()
    assert s.get('hero_arm_width_fixed_v1') and len(r.data.bones)==18
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/backup_name
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    old=bpy.data.actions['chop']
    preserved={a.name: [(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right))
                      for k in c.keyframe_points]) for c in work.curves(a)]
               for a in bpy.data.actions if a!=old}
    drivers=list(r.animation_data.drivers)
    driven={(c.data_path,c.array_index) for c in drivers}
    r.animation_data.action=None
    samples=[];custom=[];previous={};reach=0
    try:
        for c in drivers:c.mute=True
        for t in PHASES:
            values,raw,previous,e=pose(t,previous)
            samples.append(values);custom.append(raw);reach=max(reach,e)
    finally:
        for c in drivers:c.mute=False
    assert reach<.001,('Unreachable chop grip',reach)
    for i in range(1,len(samples)):
        for n in samples[i]:
            q=Quaternion(samples[i][n]['rotation_quaternion'])
            if q.dot(Quaternion(samples[i-1][n]['rotation_quaternion']))<0:q.negate()
            samples[i][n]['rotation_quaternion']=tuple(q)
    samples[-1]=copy.deepcopy(samples[0]);custom[-1]=copy.deepcopy(custom[0])
    contact,release=(PHASES.index(t) for t in CONTACT_HOLD)
    samples[release]=copy.deepcopy(samples[contact]);custom[release]=copy.deepcopy(custom[contact])
    a=bpy.data.actions.new('chop_axe_build');a.use_fake_user=True;a.use_frame_range=True
    a.frame_start=1;a.frame_end=LENGTH+1;a['loop_frames']=LENGTH;a['looping']=True
    a['description']='前后分握斧柄，分腿屈膝随髋部转移重心，大幅蓄力后砍入前方树干停顿、拔出'
    a['author_pose_times']=len(PHASES)
    a['contact_hold_start']=1+LENGTH*CONTACT_HOLD[0]
    a['contact_hold_end']=1+LENGTH*CONTACT_HOLD[1]
    a['trunk_front_x']=TRUNK_FRONT
    a['extraction_end_frame']=1+LENGTH*.78
    a['grip_spacing']=GRIP_SPACING
    a['grip_order']='right_front_left_rear'
    a['lower_body_follow']=True
    a['swing_arc_degrees']=160
    r.animation_data.action=a
    for p in r.pose.bones:
        for channel,value in samples[0][p.name].items():
            setattr(p,channel,value)
            for i in range(len(value)):
                if (p.path_from_id(channel),i) not in driven:
                    p.keyframe_insert(channel,index=i,frame=1,group=p.name)
        for key,value in custom[0].get(p.name,{}).items():
            p[key]=value;p.keyframe_insert('["'+key+'"]',frame=1,group=p.name)
    times=[1+t*LENGTH for t in PHASES]
    for c in work.curves(a):
        n=c.data_path.split('"')[1]
        if c.data_path.endswith(']'):
            key=c.data_path.split('"')[3]
            ys=[p[n][key] for p in custom]
        else:
            channel=c.data_path.rsplit('.',1)[1]
            ys=[p[n][channel][c.array_index] for p in samples]
        if max(ys)-min(ys)<1e-7:continue
        c.keyframe_points.clear()
        for f,y in zip(times,ys):c.keyframe_points.insert(f,y,options={'FAST'})
        c.update()
        for i,k in enumerate(c.keyframe_points):
            if i in (0,len(times)-1):
                h0,h1=times[-1]-times[-2],times[1]-times[0]
                d0,d1=(ys[-1]-ys[-2])/h0,(ys[1]-ys[0])/h1
            else:
                h0,h1=times[i]-times[i-1],times[i+1]-times[i]
                d0,d1=(ys[i]-ys[i-1])/h0,(ys[i+1]-ys[i])/h1
            slope=0 if d0*d1<=0 else ((2*h1+h0)+(h1+2*h0))/((2*h1+h0)/d0+(h1+2*h0)/d1)
            left=(times[i]-times[i-1])/3 if i else h0/3
            right=(times[i+1]-times[i])/3 if i+1<len(times) else h1/3
            k.interpolation='BEZIER';k.handle_left_type=k.handle_right_type='FREE'
            k.handle_left=(times[i]-left,ys[i]-slope*left)
            k.handle_right=(times[i]+right,ys[i]+slope*right)
        c.update()
    # Fit attachment orientation independently: restrained wrists need not twist
    # to steer the axe. Its origin stays at the evaluated right palm centre.
    targets={};desired={};last=None
    tool_times=sorted({1+k*.25 for k in range(LENGTH*4+1)}|{1+t*LENGTH for t in PHASES})
    for f in tool_times:
        wrists.select(r,a,f)
        target,axis=tool_frame(*shaft_angles((f-1)/LENGTH))
        hand=r.pose.bones['hand.R'];target.translation=(hand.head+hand.tail)*.5
        targets[f]=target
        tool=r.pose.bones['tool.R'];tool.matrix=target;bpy.context.view_layer.update()
        q=tool.rotation_quaternion.copy()
        if last and q.dot(last)<0:q.negate()
        last=q
        desired[f]={'tool.R':{'location':tuple(tool.location),'rotation_quaternion':tuple(q),'scale':tuple(tool.scale)}}
    keys={1.,float(LENGTH+1)}|{1+t*LENGTH for t in PHASES}
    # Tool-only fitting does not add body/arm key poses.
    for iteration in range(12):
        ts=sorted(keys);values=[desired[f] for f in ts];values[-1]=copy.deepcopy(values[0])
        wrists.write_samples(r,a,ts,values,['tool.R'])
        worst={};maxpos=maxangle=0
        for f,target in targets.items():
            wrists.select(r,a,f);tool=r.pose.bones['tool.R']
            distance=(tool.head-target.translation).length
            angle=tool.matrix.to_quaternion().rotation_difference(target.to_quaternion()).angle
            angle=min(angle,abs(math.tau-angle))
            maxpos=max(maxpos,distance);maxangle=max(maxangle,angle)
            error=max(distance/.0003,angle/math.radians(.2))
            if error>1.001 and f not in keys:
                interval=next((i for i in range(len(ts)-1) if ts[i]<f<ts[i+1]),None)
                if interval is not None and (interval not in worst or error>worst[interval][0]):worst[interval]=(error,f)
        if not worst:break
        keys.update(f for _,f in worst.values())
    assert maxpos<.00031 and maxangle<math.radians(.21),(maxpos,maxangle)
    # The source Action remains recoverable in the whole-file backup.
    bpy.data.actions.remove(old)
    a.name='chop';a.asset_mark()
    for name,expected in preserved.items():
        actual=[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right))
                 for k in c.keyframe_points]) for c in work.curves(bpy.data.actions[name])]
        assert actual==expected,('Unrelated clip changed',name)
    report={'pose_times':len(PHASES),'frames':LENGTH,'fps':s.render.fps,'tool_pose_times':len(keys),
            'grip_height_offset_m':GRIP_HEIGHT_OFFSET,
            'grip_forward_offset_m':GRIP_FORWARD_OFFSET,'arm_length_factor':ARM_LENGTH_FACTOR,
            'grip_spacing_m':GRIP_SPACING,'grip_order':'right_front_left_rear',
            'lower_body_follow':True,
            'swing_arc_degrees':160,
            'contact_pause_seconds':LENGTH*(CONTACT_HOLD[1]-CONTACT_HOLD[0])/s.render.fps,
            'key_pose_grip_error':reach,'tool_position_error':maxpos,
            'tool_angle_error_degrees':math.degrees(maxangle),'preserved_actions':list(preserved)}
    stats=json.loads(s.get('animation_clips','{}'))
    stats.setdefault('chop',{}).update(frames=LENGTH,duration=LENGTH/s.render.fps,
        looping=True,distinct_key_times=len(PHASES),tool_pose_times=len(keys),
        keys=sum(len(c.keyframe_points) for c in work.curves(a)))
    s['animation_clips']=json.dumps(stats)
    add_axe();add_trunk();work.set_action('chop',1)
    if sparse_attachment:
        from simplify_hero1_chop_keys import reduce
        reduction=reduce()
        report['tool_pose_times']=reduction['after']['tool_times']
        report['key_reduction']=reduction
    (ROOT/'hero1_chop_axe_report.json').write_text(json.dumps(report,indent=2))
    print('CHOP_AXE_CREATED',json.dumps(report))
    if fast_rhythm:
        from retime_hero1_chop_rhythm import apply
        report['rhythm']=apply()
        report['frames']=report['rhythm']['frames']
        report['contact_pause_seconds']=report['rhythm']['contact_pause_seconds']
    return report


if __name__=='__main__':build()
