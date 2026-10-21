"""Sparse full-body scythe harvesting, retaining the approved hero clips."""
from pathlib import Path
import copy
import json
import math
import bpy
from mathutils import Matrix, Vector, Quaternion
import hero1_work_actions as work
import relax_hero1_wrists as wrists
from refine_hero1_fishing import write_property
from preview_hero1_chop import tool_frame

ROOT=Path(__file__).resolve().parent.parent
NAME='harvest'
LENGTH=72
SPACING=.193
ARM_LENGTH=1.75
GRIP_FORWARD_OFFSET=.13
GRIP_HEIGHT_OFFSET=-.06
# phase, torso pitch/yaw, front grip x/y/z, shaft elevation/heading
ROWS=[(0,12,20,.34,.07,.62,-22,75),
      (.16,8,42,.29,.15,.64,-18,112),
      (.28,14,38,.30,.16,.60,-27,110),
      (.38,22,4,.44,.00,.57,-27,65),
      (.48,23,-32,.43,-.18,.57,-27,15),
      (.56,18,-38,.41,-.20,.60,-24,8),
      (.70,12,-20,.38,-.12,.69,-12,25),
      (.86,9,15,.39,.05,.70,-10,70),
      (1,12,20,.34,.07,.62,-22,75)]
PHASES=[row[0] for row in ROWS]
LOWER_BODY=[(0,0,0,0,-.002),(.16,10,-.005,.016,-.004),
            (.28,12,-.004,.020,-.006),(.38,0,.006,0,-.009),
            (.48,-12,.010,-.020,-.008),(.56,-14,.007,-.025,-.005),
            (.70,-8,-.003,-.012,-.002),(.86,0,0,0,0),(1,0,0,0,-.002)]


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



def pose(t,previous):
    r=work.rig();work.reset()
    pitch,yaw,gx,gy,gz,elevation,heading=work.interpolate(ROWS,t)
    gx+=GRIP_FORWARD_OFFSET
    gz+=GRIP_HEIGHT_OFFSET
    hip_yaw=lower_body_pose(t)
    work.rotate('body',pitch,yaw=yaw-hip_yaw)
    work.rotate('head',-pitch*.4,yaw=-yaw*.35)
    bpy.context.view_layer.update()
    target,axis=tool_frame(elevation,heading);base=Vector((gx,gy,gz))
    errors=[work.solve_arm('R',base,axis,ARM_LENGTH,hinge=True,elbow_pole=(1.4,1.6,-.65)),
            work.solve_arm('L',base-axis*SPACING,axis,ARM_LENGTH,hinge=True,elbow_pole=(1.4,1.6,-.65))]
    target.translation=base;r.pose.bones['tool.R'].matrix=target
    bpy.context.view_layer.update()
    e,previous=wrists.relax_current_pose(r,previous);errors.extend(e.values())
    raw={}
    for side in ('L','R'):
        u,f,h=(r.pose.bones[n+'.'+side] for n in ('upper_arm','forearm','hand'))
        U,F,H=u.scale.y,f.scale.y,h.scale.copy()
        raw[f.name]={'authored_length_ratio':F}
        raw[h.name]={'authored_palm_scale_'+axis:H[i] for i,axis in enumerate('xyz')}
        u.scale=(1,U,1);r.pose.bones['elbow_scale.'+side].scale=(1,1/U,1)
        f.scale=(1,U*F,1);r.pose.bones['wrist_scale.'+side].scale=(1,1/(U*F),1)
        h.scale=H*(U*F)
        for bone in (f,h):
            for key,value in raw[bone.name].items():bone[key]=value
    bpy.context.view_layer.update()
    values={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),
                    'scale':tuple(p.scale)} for p in r.pose.bones}
    return values,raw,previous,max(errors)


def add_scythe():
    r=work.rig()
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    if bpy.data.objects.get('PREVIEW_Scythe'):return bpy.data.objects['PREVIEW_Scythe']
    vertices=[];faces=[];indices=[]
    def prism(outline,half_width,material,blade=False):
        i=len(vertices);n=len(outline)
        vertices.extend([(x,y,z-(math.tan(math.radians(27))*(y-.95) if blade else 0)) for z in (-half_width,half_width) for x,y in outline])
        faces.extend([tuple(i+j for j in reversed(range(n))),tuple(i+n+j for j in range(n))])
        faces.extend([(i+j,i+(j+1)%n,i+(j+1)%n+n,i+j+n) for j in range(n)])
        indices.extend([material]*(n+2))
    prism([(-.014,-.22),(.014,-.22),(.014,.98),(-.014,.98)],.014,0)
    prism([(-.023,.91),(.025,.91),(.025,.99),(-.023,.99)],.023,1)
    # A broad crescent, mounted level with the ground during the cutting stroke.
    outline=[(.03,1.04),(-.18,1.08),(-.40,1.01),(-.62,.86),(-.76,.63),
             (-.54,.82),(-.31,.90),(-.10,.90),(.03,.90)]
    prism(outline,.012,1,True)
    prism([(-.18,1.08),(-.40,1.01),(-.62,.86),(-.76,.63),(-.59,.84),(-.38,.985),(-.18,1.055)],.013,2,True)
    data=bpy.data.meshes.new('Preview long handle scythe geometry');data.from_pydata(vertices,[],faces);data.update()
    obj=bpy.data.objects.new('PREVIEW_Scythe',data);bpy.context.scene.collection.objects.link(obj)
    for name,color in [('Wood',(.27,.11,.035,1)),('Steel',(.24,.31,.34,1)),('Edge',(.65,.73,.76,1))]:
        mat=bpy.data.materials.get('PREVIEW_Scythe_'+name) or bpy.data.materials.new('PREVIEW_Scythe_'+name)
        mat.use_nodes=True;mat.diffuse_color=color
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Base Color'].default_value=color;shader.inputs['Roughness'].default_value=.65
        data.materials.append(mat)
    for p,i in zip(data.polygons,indices):p.material_index=i
    obj.parent=r;obj.parent_type='BONE';obj.parent_bone='tool.R'
    obj.matrix_parent_inverse=Matrix.Identity(4);obj.location=(0,-r.data.bones['tool.R'].length,0)
    obj['preview_only']=True;obj['preview_action']=NAME
    obj['remove_after_user_approval']=True;obj['attachment_socket']='tool.R'
    return obj


def fit_tool(a):
    r=work.rig();targets={};desired={};last=None
    frames=sorted({1+k*.25 for k in range(LENGTH*4+1)}|{1+t*LENGTH for t in PHASES})
    for f in frames:
        wrists.select(r,a,f)
        target,axis=tool_frame(*work.interpolate(ROWS,(f-1)/LENGTH)[-2:])
        h=r.pose.bones['hand.R'];target.translation=(h.head+h.tail)*.5
        targets[f]=target;r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update()
        p=r.pose.bones['tool.R'];q=p.rotation_quaternion.copy()
        if last and q.dot(last)<0:q.negate()
        last=q
        desired[f]={'tool.R':dict(location=tuple(p.location),rotation_quaternion=tuple(q),scale=tuple(p.scale))}
    keys={1+t*LENGTH for t in PHASES}
    for _ in range(12):
        times=sorted(keys);values=[desired[f] for f in times];values[-1]=copy.deepcopy(values[0])
        wrists.write_samples(r,a,times,values,['tool.R'])
        worst={};maxpos=maxangle=0
        for f,target in targets.items():
            wrists.select(r,a,f);p=r.pose.bones['tool.R']
            distance=(p.head-target.translation).length
            angle=p.matrix.to_quaternion().rotation_difference(target.to_quaternion()).angle
            angle=min(angle,abs(math.tau-angle));maxpos=max(maxpos,distance);maxangle=max(maxangle,angle)
            error=max(distance/.0004,angle/math.radians(.6))
            if error>1.001 and f not in keys:
                interval=next((i for i in range(len(times)-1) if times[i]<f<times[i+1]),None)
                if interval is not None and (interval not in worst or error>worst[interval][0]):worst[interval]=(error,f)
        if not worst:break
        keys.update(f for _,f in worst.values())
    assert maxpos<.00041 and maxangle<math.radians(.61),(maxpos,maxangle)
    return len(keys)


def build(backup_name='hero1_before_harvest.blend'):
    r=work.rig();s=bpy.context.scene
    assert len(r.data.bones)==18 and s.get('hero_arm_width_fixed_v1')
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/backup_name
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    snapshot=lambda a:[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points]) for c in work.curves(a)]
    preserved={a.name:snapshot(a) for a in bpy.data.actions if a.name!=NAME}
    old=bpy.data.actions.get(NAME);r.animation_data.action=None
    drivers=list(r.animation_data.drivers);driven={(c.data_path,c.array_index) for c in drivers}
    samples=[];custom=[];previous={};reach=0
    try:
        for c in drivers:c.mute=True
        for t in PHASES:
            values,raw,previous,e=pose(t,previous)
            samples.append(values);custom.append(raw);reach=max(reach,e)
    finally:
        for c in drivers:c.mute=False
    assert reach<.001,('Scythe grip unreachable',reach)
    for i in range(1,len(samples)):
        for bone in samples[i]:
            q=Quaternion(samples[i][bone]['rotation_quaternion'])
            if q.dot(Quaternion(samples[i-1][bone]['rotation_quaternion']))<0:q.negate()
            samples[i][bone]['rotation_quaternion']=tuple(q)
    samples[-1]=copy.deepcopy(samples[0]);custom[-1]=copy.deepcopy(custom[0])
    a=bpy.data.actions.new(NAME+'_build');a.use_fake_user=True;a.use_frame_range=True
    a.frame_start=1;a.frame_end=LENGTH+1;a['loop_frames']=LENGTH;a['looping']=True
    a['description']='双手握长柄镰刀，分腿屈膝转腰、低位横扫收割、抬刀回收'
    a['author_pose_times']=len(PHASES);a['grip_spacing']=SPACING
    a['cut_start_frame']=1+LENGTH*.28;a['cut_end_frame']=1+LENGTH*.48
    times=[1+t*LENGTH for t in PHASES];r.animation_data.action=a
    wrists.write_samples(r,a,times,samples,list(r.pose.bones.keys()))
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
            bone,key=c.data_path.split('"')[1::2]
            write_property(c,times,[v[bone][key] for v in custom],True)
    tool_times=fit_tool(a)
    for c in work.curves(a):
        ys=[k.co.y for k in c.keyframe_points]
        if len(ys)>1 and max(ys)-min(ys)<2e-6:
            c.keyframe_points.clear();c.keyframe_points.insert(1,ys[0]);c.update()
    if old:bpy.data.actions.remove(old)
    a.name=NAME;a.asset_mark()
    assert all(snapshot(bpy.data.actions[n])==v for n,v in preserved.items())
    report=dict(frames=LENGTH,fps=s.render.fps,duration=LENGTH/s.render.fps,looping=True,
                grip_forward_offset_m=GRIP_FORWARD_OFFSET,grip_height_offset_m=GRIP_HEIGHT_OFFSET,arm_length_factor=ARM_LENGTH,
                distinct_key_times=len(PHASES),tool_pose_times=tool_times,
                keys=sum(len(c.keyframe_points) for c in work.curves(a)),max_grip_reach_error_m=reach,
                preserved_actions=list(preserved))
    stats=json.loads(s.get('animation_clips','{}'));stats[NAME]=report;s['animation_clips']=json.dumps(stats)
    add_scythe();work.set_action(NAME,1)
    from simplify_hero1_harvest_keys import reduce
    report['key_reduction']=reduce()
    report['tool_pose_times']=report['key_reduction']['after']['tool_times']
    report['keys']=report['key_reduction']['after']['keys']
    (ROOT/'hero1_harvest_report.json').write_text(json.dumps(report,indent=2))
    print('HARVEST_CREATED',json.dumps(report))
    return report


if __name__=='__main__':build()
