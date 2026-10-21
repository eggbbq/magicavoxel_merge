"""Keep voxel palms attached with restrained wrists, preserving grip centres.

The palms are unmarked cubes. Their 24 equivalent orientations let us choose
one close to the forearm before limiting flexion/twist. Re-solve the elbow and
wrist around the original palm centre; compensate tool.R to keep its world pose.
No constraints, new bones, mesh edits, or per-frame authoring keys are required.
"""
from pathlib import Path
import itertools
import json
import math
import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT=Path(__file__).resolve().parent.parent
KEY_SWING=math.radians(32)
KEY_TWIST=math.radians(22)
MAX_SWING=math.radians(35)
MAX_TWIST=math.radians(25)

def cube_rotations():
    out=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            m=Matrix([[signs[c] if row==perm[c] else 0 for c in range(3)] for row in range(3)])
            if m.determinant()>.5:out.append(m.to_quaternion())
    return out

SYMMETRIES=cube_rotations()

def clamp_wrist(q):
    q=q.normalized()
    if q.w<0:q.negate()
    swing,twist=q.to_swing_twist('Y')
    angle=swing.angle
    if angle>KEY_SWING:swing=Quaternion().slerp(swing,KEY_SWING/angle)
    twist=max(-KEY_TWIST,min(KEY_TWIST,twist))
    return (swing@Quaternion((0,1,0),twist)).normalized()

def frame(y,z):
    return Matrix((y.cross(z).normalized(),y,z)).transposed().to_quaternion()

def ik(start,target,old_elbow,l1,l2):
    delta=target-start;distance=min(l1+l2-1e-6,max(abs(l1-l2)+1e-6,delta.length))
    direction=delta.normalized()
    pole=old_elbow-start;pole-=direction*pole.dot(direction)
    if pole.length<1e-7:
        pole=Vector((1,0,-.5));pole-=direction*pole.dot(direction)
    pole.normalize()
    cosine=max(-1,min(1,(l1*l1+distance*distance-l2*l2)/(2*l1*distance)))
    elbow=start+direction*(l1*cosine)+pole*(l1*math.sqrt(max(0,1-cosine*cosine)))
    u=(elbow-start).normalized();v=(start+direction*distance-elbow).normalized()
    z=-u.cross(v).normalized()
    return elbow,frame(u,z),frame(v,z)

def matrix(q,scale,head):
    m=q.to_matrix().to_4x4()@Matrix.Diagonal((*scale,1));m.translation=head
    return m

def relax_current_pose(rig=None,previous=None):
    r=rig or bpy.data.objects['Hero']
    tool=r.pose.bones.get('tool.R')
    old_tool=tool.matrix.copy() if tool else None
    original={}
    for side in ('L','R'):
        u,f,h=(r.pose.bones[n+'.'+side] for n in ('upper_arm','forearm','hand'))
        original[side]=(u.head.copy(),f.head.copy(),h.head.copy(),(h.head+h.tail)*.5,
                        u.matrix.to_scale(),f.matrix.to_scale(),h.matrix.to_scale(),h.matrix.to_quaternion())
    errors={};local_refs={}
    for side in ('L','R'):
        u,f,h=(r.pose.bones[n+'.'+side] for n in ('upper_arm','forearm','hand'))
        start,old_elbow,wrist,centre,su,sf,sh,desired=original[side]
        l1=u.bone.length*abs(su.y);l2=f.bone.length*abs(sf.y)
        half=h.bone.length*abs(sh.y)*.5
        _,_,lower=ik(start,wrist,old_elbow,l1,l2)
        # Orient the solid cube equivalently, avoiding a half-turn at its skin seam.
        best=None
        for sym in SYMMETRIES:
            target=desired@sym;relative=lower.inverted()@target
            bounded=clamp_wrist(relative)
            diff=relative.rotation_difference(bounded).angle
            continuity=previous[side].rotation_difference(bounded).angle if previous and side in previous else 0
            cost=diff*diff+relative.angle**2*.03+continuity*continuity*.12
            if best is None or cost<best[0]:best=(cost,target)
        target=best[1]
        for _ in range(18):
            elbow,upper,lower=ik(start,wrist,old_elbow,l1,l2)
            local=clamp_wrist(lower.inverted()@target)
            hand=lower@local
            next_wrist=centre-(hand@Vector((0,1,0)))*half
            if (next_wrist-wrist).length<1e-8:break
            wrist=next_wrist
        elbow,upper,lower=ik(start,wrist,old_elbow,l1,l2)
        local=clamp_wrist(lower.inverted()@target);hand=lower@local
        # The body, shoulder point, segment lengths and palm size stay as authored.
        u.matrix=matrix(upper,su,start);bpy.context.view_layer.update()
        f.location=(0,0,0);f.matrix=matrix(lower,sf,f.head.copy());bpy.context.view_layer.update()
        h.location=(0,0,0);h.matrix=matrix(hand,sh,h.head.copy());bpy.context.view_layer.update()
        errors[side]=((h.head+h.tail)*.5-centre).length
        local_refs[side]=h.rotation_quaternion.copy()
    if tool:
        tool.matrix=old_tool;bpy.context.view_layer.update()
    return errors,local_refs

def curves(a):
    return [c for l in a.layers for s in l.strips for b in s.channelbags for c in b.fcurves]

def select(r,a,f):
    tool=r.pose.bones.get('tool.R')
    if tool:
        tool.location=(0,0,0);tool.rotation_quaternion=(1,0,0,0);tool.scale=(1,1,1)
    r.animation_data.action=a;r.animation_data.action_slot=a.slots[0]
    bpy.context.scene.frame_set(math.floor(f),subframe=f%1);bpy.context.view_layer.update()

def write_samples(r,a,times,samples,names):
    for layer in a.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for c in list(bag.fcurves):
                    if any(c.data_path.startswith('pose.bones["'+n+'"]') for n in names):bag.fcurves.remove(c)
    for n in names:
        for channel,value in samples[0][n].items():
            setattr(r.pose.bones[n],channel,value)
            r.pose.bones[n].keyframe_insert(channel,frame=times[0],group=n)
    for c in curves(a):
        if not any(c.data_path.startswith('pose.bones["'+n+'"]') for n in names):continue
        bone=c.data_path.split('"')[1];channel=c.data_path.rsplit('.',1)[-1]
        ys=[s[bone][channel][c.array_index] for s in samples]
        if max(ys)-min(ys)<1e-7:continue
        def slope(i):
            if i in (0,len(times)-1):
                h0,h1=times[-1]-times[-2],times[1]-times[0]
                d0,d1=(ys[-1]-ys[-2])/h0,(ys[1]-ys[0])/h1
                if not a.get('looping'):return 0
            else:
                h0,h1=times[i]-times[i-1],times[i+1]-times[i]
                d0,d1=(ys[i]-ys[i-1])/h0,(ys[i+1]-ys[i])/h1
            if d0*d1<=0:return 0
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

def apply():
    r=bpy.data.objects['Hero'];s=bpy.context.scene
    assert not s.get('hero_wrists_relaxed_v1'),'Already applied; use the saved backup for a fresh correction.'
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero1_before_wrist_relaxation.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    bones=[n+'.'+side for side in ('L','R') for n in ('upper_arm','forearm','hand')]
    if r.pose.bones.get('tool.R'):bones.append('tool.R')
    report={};tool_targets={}
    # Capture before editing any Action: old clips without tool channels must not
    # inherit compensation left by the previously evaluated Action.
    for a in bpy.data.actions:
        times={round(k.co.x,6) for c in curves(a) for k in c.keyframe_points}
        times.update(1+k*.25 for k in range(4*int(a['loop_frames'])+1))
        targets={}
        for f in sorted(times):
            select(r,a,f)
            if r.pose.bones.get('tool.R'):targets[f]=r.pose.bones['tool.R'].matrix.copy()
        tool_targets[a.name]=targets
    for a in bpy.data.actions:
        needs=False
        for k in range(33):
            select(r,a,1+int(a['loop_frames'])*k/32)
            for side in ('L','R'):
                sw,tw=r.pose.bones['hand.'+side].rotation_quaternion.normalized().to_swing_twist('Y')
                needs|=sw.angle>MAX_SWING or abs(tw)>MAX_TWIST
        if not needs:
            if r.pose.bones.get('tool.R'):
                select(r,a,1)
                p=r.pose.bones['tool.R'];p.location=(0,0,0);p.rotation_quaternion=(1,0,0,0);p.scale=(1,1,1)
                for channel in ('location','rotation_quaternion','scale'):p.keyframe_insert(channel,frame=1,group='tool.R')
            report[a.name]={'changed':False};continue
        times=sorted({round(k.co.x,6) for c in curves(a) for k in c.keyframe_points})
        samples=[];max_error=0;previous={}
        untouched=[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points])
                   for c in curves(a) if not any(c.data_path.startswith('pose.bones["'+n+'"]') for n in bones)]
        for f in times:
            select(r,a,f)
            errors,previous=relax_current_pose(r,previous);max_error=max(max_error,*errors.values())
            sample={n:{'location':tuple(r.pose.bones[n].location),'rotation_quaternion':tuple(r.pose.bones[n].rotation_quaternion),
                       'scale':tuple(r.pose.bones[n].scale)} for n in bones}
            if samples:
                for n in bones:
                    q=Quaternion(sample[n]['rotation_quaternion']);old=Quaternion(samples[-1][n]['rotation_quaternion'])
                    if q.dot(old)<0:q.negate()
                    sample[n]['rotation_quaternion']=tuple(q)
            samples.append(sample)
        samples[-1]={n:dict(samples[0][n]) for n in bones}
        write_samples(r,a,times,samples,bones)
        now=[(c.data_path,c.array_index,[(tuple(k.co),tuple(k.handle_left),tuple(k.handle_right)) for k in c.keyframe_points])
             for c in curves(a) if not any(c.data_path.startswith('pose.bones["'+n+'"]') for n in bones)]
        assert untouched==now,(a.name,'Body/head/legs changed')
        assert max_error<1e-5,(a.name,'Grip centre moved',max_error)
        a['wrist_swing_limit_degrees']=35;a['wrist_twist_limit_degrees']=25
        report[a.name]={'changed':True,'pose_times':len(times),'key_pose_palm_centre_error':max_error}
    # Compensate the attachment independently of the palms, with adaptive sparse
    # keys. The tool's approved world-space swing must survive the wrist changes.
    for a in bpy.data.actions:
        if not report[a.name]['changed'] or not tool_targets[a.name]:continue
        targets=tool_targets[a.name];desired={};last=None
        for f,target in targets.items():
            select(r,a,f);p=r.pose.bones['tool.R'];p.matrix=target;bpy.context.view_layer.update()
            q=p.rotation_quaternion.copy()
            if last and q.dot(last)<0:q.negate()
            last=q
            desired[f]={'tool.R':{'location':tuple(p.location),'rotation_quaternion':tuple(q),'scale':tuple(p.scale)}}
        keys={round(k.co.x,6) for c in curves(a) if 'tool.R' not in c.data_path for k in c.keyframe_points}
        for iteration in range(12):
            times=sorted(keys);samples=[desired[f] for f in times]
            samples[-1]={'tool.R':dict(samples[0]['tool.R'])}
            write_samples(r,a,times,samples,['tool.R'])
            worst={};max_angle=max_position=0
            for f,target in targets.items():
                select(r,a,f);p=r.pose.bones['tool.R']
                angle=p.matrix.to_quaternion().rotation_difference(target.to_quaternion()).angle
                angle=min(angle,abs(2*math.pi-angle))
                distance=(p.head-target.translation).length
                max_angle=max(max_angle,angle);max_position=max(max_position,distance)
                error=max(angle/math.radians(.15),distance/.0002)
                if error>1.001 and f not in keys:
                    interval=next((i for i in range(len(times)-1) if times[i]<f<times[i+1]),None)
                    if interval is not None and (interval not in worst or error>worst[interval][0]):worst[interval]=(error,f)
            if not worst:break
            keys.update(f for _,f in worst.values())
        assert max_angle<math.radians(.17) and max_position<.00021,(a.name,'tool compensation did not converge',max_angle,max_position)
        report[a.name].update(tool_pose_times=len(keys),tool_rotation_error_degrees=math.degrees(max_angle),tool_position_error=max_position)
    s['hero_wrists_relaxed_v1']=True
    (ROOT/'hero1_wrist_relaxation_report.json').write_text(json.dumps(report,indent=2))
    print('WRISTS_RELAXED',json.dumps(report))
    return report

if __name__=='__main__':apply()
