"""Lower the fishing grip and author a higher, connected pull on the existing rig."""
from pathlib import Path
import copy
import json
import math
import bpy
from mathutils import Vector, Quaternion
import hero1_work_actions as work
import relax_hero1_wrists as wrists
from preview_hero1_chop import tool_frame

ROOT=Path(__file__).resolve().parent.parent
ARM_LENGTH=1.5
SPACING=.08
# phase, torso pitch, knee bend, right grip x/z, rod elevation
PULL=[(0,6,8,.285,.55,28),
      (.12,11,14,.305,.54,18),
      (.30,-9,12,.22,.78,58),
      (.44,-18,10,.13,1.05,82),
      (.58,-16,10,.14,1.06,80),
      (.72,-5,10,.24,.82,60),
      (.88,5,8,.29,.59,35),
      (1,6,8,.285,.55,28)]
SPECS={'fishing':(144,True,[i/8 for i in range(9)]),
       'fishing_pull':(76,False,sorted({r[0] for r in PULL}|{.21,.37,.65,.80}))}


def parameters(name,t):
    if name=='fishing':
        breath=math.sin(math.tau*t)
        return 6+.4*breath,8,.285+.002*breath,.55+.003*breath,28+.7*breath
    return work.interpolate(PULL,t)


def pose(name,t,previous):
    r=work.rig();work.reset()
    pitch,knee,gx,gz,angle=parameters(name,t)
    half=math.radians(knee)*.5
    height=.3*math.cos(half)+.02*math.sin(half)+.002
    hip=r.pose.bones['hips']
    hip.location=hip.bone.matrix_local.to_quaternion().inverted()@Vector((0,0,height-.3))
    work.rotate('body',pitch);work.rotate('head',-pitch*.55)
    for side in ('L','R'):
        work.rotate('thigh.'+side,-knee*.5);work.rotate('shin.'+side,knee)
    bpy.context.view_layer.update()
    target,axis=tool_frame(angle,0);base=Vector((gx,.01,gz))
    errors=[work.solve_arm('R',base,axis,ARM_LENGTH,hinge=True),
            work.solve_arm('L',base+axis*SPACING,axis,ARM_LENGTH,hinge=True)]
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


def write_property(c,times,ys,looping):
    c.keyframe_points.clear()
    constant=max(ys)-min(ys)<1e-7
    for t,y in zip(times[:1] if constant else times,ys):c.keyframe_points.insert(t,y,options={'FAST'})
    c.update()
    if constant:return
    for i,k in enumerate(c.keyframe_points):
        if i in (0,len(times)-1):
            h0,h1=times[-1]-times[-2],times[1]-times[0]
            d0,d1=(ys[-1]-ys[-2])/h0,(ys[1]-ys[0])/h1
        else:
            h0,h1=times[i]-times[i-1],times[i+1]-times[i]
            d0,d1=(ys[i]-ys[i-1])/h0,(ys[i+1]-ys[i])/h1
        slope=0 if d0*d1<=0 or (not looping and i in (0,len(times)-1)) else ((2*h1+h0)+(h1+2*h0))/((2*h1+h0)/d0+(h1+2*h0)/d1)
        left=(times[i]-times[i-1])/3 if i else h0/3
        right=(times[i+1]-times[i])/3 if i+1<len(times) else h1/3
        k.interpolation='BEZIER';k.handle_left_type=k.handle_right_type='FREE'
        k.handle_left=(times[i]-left,ys[i]-slope*left)
        k.handle_right=(times[i]+right,ys[i]+slope*right)
    c.update()


def fit_tool(a,name,length,phases):
    r=work.rig();targets={};desired={};last=None
    frames=sorted({1+k*.25 for k in range(length*4+1)}|{1+t*length for t in phases})
    for f in frames:
        wrists.select(r,a,f)
        target,axis=tool_frame(parameters(name,(f-1)/length)[-1],0)
        h=r.pose.bones['hand.R'];target.translation=(h.head+h.tail)*.5
        targets[f]=target;r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update()
        p=r.pose.bones['tool.R'];q=p.rotation_quaternion.copy()
        if last and q.dot(last)<0:q.negate()
        last=q
        desired[f]={'tool.R':dict(location=tuple(p.location),rotation_quaternion=tuple(q),scale=tuple(p.scale))}
    keys={1+t*length for t in phases}
    for _ in range(12):
        times=sorted(keys);values=[desired[f] for f in times];values[-1]=copy.deepcopy(values[0])
        wrists.write_samples(r,a,times,values,['tool.R'])
        worst={};maxpos=maxangle=0
        for f,target in targets.items():
            wrists.select(r,a,f);p=r.pose.bones['tool.R']
            distance=(p.head-target.translation).length
            angle=p.matrix.to_quaternion().rotation_difference(target.to_quaternion()).angle
            angle=min(angle,abs(math.tau-angle));maxpos=max(maxpos,distance);maxangle=max(maxangle,angle)
            error=max(distance/.0004,angle/math.radians(.4))
            if error>1.001 and f not in keys:
                interval=next((i for i in range(len(times)-1) if times[i]<f<times[i+1]),None)
                if interval is not None and (interval not in worst or error>worst[interval][0]):worst[interval]=(error,f)
        if not worst:break
        keys.update(f for _,f in worst.values())
    assert maxpos<.00041 and maxangle<math.radians(.41),(maxpos,maxangle)
    return len(keys)


def build(backup_name='hero1_before_fishing_refinement.blend'):
    r=work.rig();s=bpy.context.scene
    assert len(r.data.bones)==18 and s.get('hero_arm_width_fixed_v1')
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/backup_name
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    snapshot=lambda a:[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points]) for c in work.curves(a)]
    preserved={a.name:snapshot(a) for a in bpy.data.actions if a.name not in SPECS}
    drivers=list(r.animation_data.drivers);driven={(c.data_path,c.array_index) for c in drivers}
    report={};entry=None;entry_raw=None
    for name,(length,looping,phases) in SPECS.items():
        old=bpy.data.actions[name];r.animation_data.action=None
        samples=[];custom=[];previous={};reach=0
        try:
            for c in drivers:c.mute=True
            for t in phases:
                values,raw,previous,e=pose(name,t,previous)
                samples.append(values);custom.append(raw);reach=max(reach,e)
        finally:
            for c in drivers:c.mute=False
        assert reach<.001,(name,'Grip unreachable',reach)
        if entry is None:entry=copy.deepcopy(samples[0]);entry_raw=copy.deepcopy(custom[0])
        else:samples[0]=copy.deepcopy(entry);custom[0]=copy.deepcopy(entry_raw)
        for i in range(1,len(samples)):
            for bone in samples[i]:
                q=Quaternion(samples[i][bone]['rotation_quaternion'])
                if q.dot(Quaternion(samples[i-1][bone]['rotation_quaternion']))<0:q.negate()
                samples[i][bone]['rotation_quaternion']=tuple(q)
        samples[-1]=copy.deepcopy(entry);custom[-1]=copy.deepcopy(entry_raw)
        a=bpy.data.actions.new(name+'_refined');a.use_fake_user=True;a.use_frame_range=True
        a.frame_start=1;a.frame_end=length+1;a['loop_frames']=length;a['looping']=looping
        a['description']='低位前伸双手垂钓' if looping else '衔接低位垂钓，抬高拉杆、后仰收力、回到垂钓'
        times=[1+t*length for t in phases];r.animation_data.action=a
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
                write_property(c,times,[v[bone][key] for v in custom],looping)
        tool_keys=fit_tool(a,name,length,phases)
        bpy.data.actions.remove(old);a.name=name;a.asset_mark()
        report[name]=dict(frames=length,duration=length/s.render.fps,looping=looping,
                          distinct_key_times=len(phases),tool_pose_times=tool_keys,
                          keys=sum(len(c.keyframe_points) for c in work.curves(a)),max_grip_reach_error_m=reach)
    assert all(snapshot(bpy.data.actions[n])==v for n,v in preserved.items())
    stats=json.loads(s.get('animation_clips','{}'));stats.update(report);s['animation_clips']=json.dumps(stats)
    (ROOT/'hero1_fishing_refinement_report.json').write_text(json.dumps(dict(clips=report,preserved_actions=list(preserved)),indent=2))
    work.set_action('fishing',1)
    print('FISHING_REFINED',json.dumps(report))
    return report


if __name__=='__main__':build()
