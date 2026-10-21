"""Sparse skewer cooking loop, with a hand-held preview and a simple campfire."""
from pathlib import Path
import copy, json, math, sys
import bpy
from mathutils import Quaternion, Vector, Matrix

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
import hero1_work_actions as work
import relax_hero1_wrists as wrists
from refine_hero1_fishing import write_property
from preview_hero1_chop import tool_frame
from retarget_voxel_character import action_fingerprint
from create_hero2_paddling import Geometry

NAME='grilling'; LENGTH=192; ARM_LENGTH=1.35
# phase, torso pitch, head bow, right grip X/Z, shaft elevation/heading/roll,
# left open hand X/Z. Feed over fire, turn, lift to inspect, lower again.
ROWS=[(0,10,15,.31,.53,-9,-5,-28,.29,.53),
      (.12,14,18,.36,.51,-8,-5,-28,.35,.49),
      (.24,14,18,.36,.51,-8,-5,28,.30,.59),
      (.36,13,17,.35,.52,-8,-5,28,.35,.49),
      (.47,13,17,.35,.52,-8,-5,-28,.30,.58),
      (.56,10,14,.33,.57,-5,-5,-28,.28,.50),
      (.69,3,8,.27,.73,2,-5,22,.27,.49),
      (.78,3,8,.27,.73,2,-5,22,.27,.49),
      (.91,8,13,.30,.57,-6,-5,-28,.28,.51),
      (1,10,15,.31,.53,-9,-5,-28,.29,.53)]
PHASES=sorted({r[0] for r in ROWS}|{.18,.415,.625,.845})


def parameters(t):return work.interpolate(ROWS,t)


def shaft(t):
    p=parameters(t)
    matrix,axis=tool_frame(p[4],p[5])
    return matrix@Quaternion((0,1,0),math.radians(p[6])).to_matrix().to_4x4(),axis


def pose(t,previous):
    r=work.rig();work.reset()
    pitch,head,gx,gz,e,h,roll,lx,lz=parameters(t)
    thigh,knee=-7,14
    a,b=math.radians(thigh),math.radians(thigh+knee)
    height=.16*math.cos(a)+.14*math.cos(b)+.02*abs(math.sin(b))+.002
    hip=r.pose.bones['hips'];hip.location=hip.bone.matrix_local.to_quaternion().inverted()@Vector((0,0,height-.3))
    work.rotate('body',pitch);work.rotate('head',head)
    for side in ('L','R'):
        work.rotate('thigh.'+side,thigh);work.rotate('shin.'+side,knee)
    bpy.context.view_layer.update()
    target,axis=shaft(t);grip=Vector((gx,.145,gz))
    errors=[work.solve_arm('R',grip,axis,ARM_LENGTH,hinge=True,elbow_pole=(-.3,.9,-1.4)),
            work.solve_arm('L',Vector((lx,-.20,lz)),Vector((.8,.1,-.6)),ARM_LENGTH,hinge=True,elbow_pole=(-.3,.9,-1.4))]
    target.translation=grip;r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update()
    es,previous=wrists.relax_current_pose(r,previous);errors.extend(es.values())
    raw={}
    for side in ('L','R'):
        u,f,hand=(r.pose.bones[n+'.'+side] for n in ('upper_arm','forearm','hand'))
        U,F,H=u.scale.y,f.scale.y,hand.scale.copy()
        raw[f.name]={'authored_length_ratio':F};raw[hand.name]={'authored_palm_scale_'+c:H[i] for i,c in enumerate('xyz')}
        u.scale=(1,U,1);r.pose.bones['elbow_scale.'+side].scale=(1,1/U,1)
        f.scale=(1,U*F,1);r.pose.bones['wrist_scale.'+side].scale=(1,1/(U*F),1);hand.scale=H*(U*F)
        for bone in (f,hand):
            for key,value in raw[bone.name].items():bone[key]=value
    bpy.context.view_layer.update()
    values={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),'scale':tuple(p.scale)} for p in r.pose.bones}
    return values,raw,previous,max(errors)


def fit_socket(action):
    r=work.rig();desired={};targets={};last=None
    for f in sorted({1+k*.25 for k in range(LENGTH*4+1)}|{1+t*LENGTH for t in PHASES}):
        wrists.select(r,action,f);target,_=shaft((f-1)/LENGTH)
        hand=r.pose.bones['hand.R'];target.translation=(hand.head+hand.tail)*.5;targets[f]=target
        r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update();p=r.pose.bones['tool.R'];q=p.rotation_quaternion.copy()
        if last and q.dot(last)<0:q.negate()
        last=q;desired[f]={'tool.R':dict(location=tuple(p.location),rotation_quaternion=tuple(q),scale=tuple(p.scale))}
    keys={1+t*LENGTH for t in PHASES}
    for _ in range(10):
        times=sorted(keys);values=[desired[f] for f in times];values[-1]=copy.deepcopy(values[0])
        wrists.write_samples(r,action,times,values,['tool.R'])
        worst={};maxpos=maxangle=0.
        for f,target in targets.items():
            wrists.select(r,action,f);p=r.pose.bones['tool.R'];distance=(p.head-target.translation).length
            angle=p.matrix.to_quaternion().rotation_difference(target.to_quaternion()).angle;angle=min(angle,abs(math.tau-angle))
            maxpos=max(maxpos,distance);maxangle=max(maxangle,angle)
            error=max(distance/.0006,angle/math.radians(.7))
            if error>1.001 and f not in keys:
                interval=next((i for i in range(len(times)-1) if times[i]<f<times[i+1]),None)
                if interval is not None and (interval not in worst or error>worst[interval][0]):worst[interval]=(error,f)
        if not worst:break
        keys.update(f for _,f in worst.values())
    assert maxpos<.00061 and maxangle<math.radians(.71),(maxpos,maxangle)
    return len(keys)


def previews():
    r=work.rig()
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    for n in ('PREVIEW_Skewer','PREVIEW_Campfire'):
        if bpy.data.objects.get(n):bpy.data.objects.remove(bpy.data.objects[n],do_unlink=True)
    g=Geometry();g.box((-.006,-.08,-.006),(.006,.56,.006),0);g.box((-.013,-.08,-.013),(.013,.035,.013),0)
    for i,y in enumerate((.245,.345,.445)):
        g.box((-.039,y-.036,-.034),(.039,y+.036,.034),1)
        # A pale fat side and visible sear marks make the turn readable.
        g.box((-.039,y-.036,.034),(.039,y+.036,.040),2)
        for dy in (-.017,.014):g.box((-.040,y+dy,.040),(.040,y+dy+.007,.042),3)
        if i<2:g.box((-.038,y+.044,-.038),(.038,y+.057,.038),4+i)
    obj=g.object('PREVIEW_Skewer',[('Wood',(.27,.105,.025)),('Meat',(.40,.105,.035)),('Fat',(.68,.32,.12)),('Sear',(.10,.035,.01)),('Pepper',(.12,.32,.035)),('Onion',(.68,.52,.32))])
    obj['preview_actions']=NAME;obj['attachment_socket']='tool.R';obj.parent=r;obj.parent_type='BONE';obj.parent_bone='tool.R';obj.matrix_parent_inverse=Matrix.Identity(4);obj.location=(0,-r.data.bones['tool.R'].length,0)
    g=Geometry()
    for i in range(8):
        angle=math.tau*i/8;x,y=.72+.205*math.cos(angle),.10+.205*math.sin(angle)
        g.box((x-.043,y-.038,0),(x+.043,y+.038,.062),0)
    g.box((.56,-.01,.042),(.88,.07,.10),1);g.box((.67,-.06,.045),(.75,.26,.105),1)
    g.box((.61,.015,.10),(.84,.19,.13),2)
    for x,y,w,top in ((.67,.08,.075,.34),(.77,.13,.065,.29),(.73,.03,.045,.26)):
        i=len(g.v);g.v.extend([(x-w,y-w,.12),(x+w,y-w,.12),(x+w,y+w,.12),(x-w,y+w,.12),(x+.015,y,top)])
        for fs in ((0,1,4),(1,2,4),(2,3,4),(3,0,4)):g.face([i+k for k in fs],3)
        g.box((x-w*.4,y-w*.4,.13),(x+w*.4,y+w*.4,.19),4)
    fire=g.object('PREVIEW_Campfire',[('Stones',(.20,.22,.23)),('Logs',(.12,.045,.012)),('Coals',(.55,.055,.008)),('Flame',(.95,.20,.01)),('Core',(1.,.63,.045))]);fire['preview_actions']=NAME
    # Static low-poly fire is sufficient to frame the character's cooking motion.
    for mat in fire.data.materials[2:]:
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        shader.inputs['Emission Color'].default_value=(*mat.diffuse_color[:3],1)
        shader.inputs['Emission Strength'].default_value=.6


def build():
    r=work.rig();s=bpy.context.scene
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    preserved={a.name:action_fingerprint(a) for a in bpy.data.actions if a.name!=NAME}
    old=bpy.data.actions.get(NAME);r.animation_data.action=None
    drivers=list(r.animation_data.drivers);driven={(c.data_path,c.array_index) for c in drivers}
    samples=[];custom=[];previous={};reach=0.
    try:
        for c in drivers:c.mute=True
        for t in PHASES:
            values,raw,previous,error=pose(t,previous);samples.append(values);custom.append(raw);reach=max(reach,error)
    finally:
        for c in drivers:c.mute=False
    assert reach<.001,('Grip unreachable',reach)
    for i in range(1,len(samples)):
        for n in samples[i]:
            q=Quaternion(samples[i][n]['rotation_quaternion'])
            if q.dot(Quaternion(samples[i-1][n]['rotation_quaternion']))<0:q.negate()
            samples[i][n]['rotation_quaternion']=tuple(q)
    samples[-1]=copy.deepcopy(samples[0]);custom[-1]=copy.deepcopy(custom[0])
    action=bpy.data.actions.new(NAME+'_build');action.use_fake_user=True;action.use_frame_range=True
    action.frame_start=1;action.frame_end=LENGTH+1;action['loop_frames']=LENGTH;action['looping']=True
    action['description']='烧烤：前伸烤串、翻面、抬起查看，再放回火堆上方；左手轻扇'
    action['author_pose_times']=len(PHASES);action['tool_socket']='tool.R'
    times=[1+t*LENGTH for t in PHASES];r.animation_data.action=action
    wrists.write_samples(r,action,times,samples,list(r.pose.bones.keys()))
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for c in list(bag.fcurves):
                    if (c.data_path,c.array_index) in driven:bag.fcurves.remove(c)
    for bone,props in custom[0].items():
        for key,value in props.items():
            p=r.pose.bones[bone];p[key]=value;p.keyframe_insert('["'+key+'"]',frame=1,group=bone)
    for c in work.curves(action):
        if c.data_path.endswith(']'):
            bone,key=c.data_path.split('"')[1::2];write_property(c,times,[v[bone][key] for v in custom],True)
    tool_times=fit_socket(action)
    if old:bpy.data.actions.remove(old)
    action.name=NAME;action.asset_mark()
    assert all(action_fingerprint(bpy.data.actions[n])==h for n,h in preserved.items())
    previews();work.set_action(NAME,1)
    import hero_action_preview as preview
    preview.sync_scene(s,force=True)
    result={'name':NAME,'frames':LENGTH,'fps':s.render.fps,'duration':LENGTH/s.render.fps,'looping':True,'author_pose_times':len(PHASES),'tool_pose_times':tool_times,'keys':sum(len(c.keyframe_points) for c in work.curves(action)),'max_reach_error_m':reach,'preserved_fingerprints':preserved,'preview_only':['PREVIEW_Skewer','PREVIEW_Campfire']}
    stats=json.loads(s.get('animation_clips','{}'));stats[NAME]={k:v for k,v in result.items() if k!='preserved_fingerprints'};s['animation_clips']=json.dumps(stats)
    (ROOT/'hero2_grilling_report.json').write_text(json.dumps(result,indent=2));print('GRILLING_CREATED',json.dumps({k:v for k,v in result.items() if k!='preserved_fingerprints'}))
    return result


if __name__=='__main__':build()
