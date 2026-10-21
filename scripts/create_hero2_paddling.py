"""Two standing paddle strokes on a common foot plane, independent of vessel."""
from pathlib import Path
import copy,json,math,sys
import bpy
from mathutils import Matrix,Vector,Quaternion
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
import hero1_work_actions as work
import relax_hero1_wrists as wrists
from refine_hero1_fishing import write_property
from preview_hero1_chop import tool_frame
from retarget_voxel_character import action_fingerprint
NAMES=('paddle_left','paddle_right');LENGTH=96;SPACING=.64;ARM_LENGTH=1.65
# phase, torso pitch/yaw/roll, lower grip xyz, shaft elevation and heading
ROWS=[(0,18,26,-3,.48,.38,.42,25,-90),
      (.14,27,30,-5,.54,.39,.40,27,-90),
      (.24,28,30,-5,.54,.39,.34,30,-90),
      (.47,-12,-22,2,.08,.38,.38,35,-75),
      (.58,-18,-26,3,.02,.36,.46,25,-65),
      (.72,-2,-10,0,.23,.35,.67,5,-70),
      (.86,12,16,-2,.42,.38,.60,15,-85),
      (1,18,26,-3,.48,.38,.42,25,-90)]
PHASES=sorted({row[0] for row in ROWS}|{.295,.35,.55,.64,.755,.79,.93})


def standing_offset(name,t):
    sign=1 if name=='paddle_right' else -1
    wave=math.sin(math.tau*t)
    return Vector((.008*wave,sign*.005*wave,.298+.001*wave-.2))


def parameters(name,t):
    pitch,yaw,roll,x,y,z,e,h=work.interpolate(ROWS,t);sign=1 if name=='paddle_right' else -1
    return pitch,-yaw*sign,roll*sign,Vector((x,.01+sign*y,z))+standing_offset(name,t),e,h if sign==1 else -h


def standing_legs(name,t):
    r=work.rig();m=work.mesh();offset=standing_offset(name,t)
    hip=r.pose.bones['hips'];hip.location=hip.bone.matrix_local.to_quaternion().inverted()@(offset+Vector((0,0,-.1)))
    bpy.context.view_layer.update()
    for side,x,y in (('L',.02,-.13),('R',.10,.15)):
        upper,lower=(r.pose.bones[n+'.'+side] for n in ('thigh','shin'))
        target=Vector((x,y,.0108))
        ids=[v.index for v in m.data.vertices if v.co.z<.021 and any(m.vertex_groups[g.group].name=='shin.'+side for g in v.groups)]
        for _ in range(6):
            start=upper.head.copy();delta=target-start;distance=delta.length;direction=delta.normalized()
            l1,l2=upper.bone.length,lower.bone.length
            assert abs(l1-l2)+.00005<distance<l1+l2-.00005,('Standing leg unreachable',name,t,side,distance)
            pole=Vector((1,0,0));pole=(pole-direction*pole.dot(direction)).normalized()
            cosine=max(-1,min(1,(l1*l1+distance*distance-l2*l2)/(2*l1*distance)))
            knee=start+direction*(l1*cosine)+pole*l1*math.sqrt(max(0,1-cosine*cosine))
            u=(knee-start).normalized();v=(target-knee).normalized();z=u.cross(v).normalized()
            for p,axis in ((upper,u),(lower,v)):
                matrix=Matrix((axis.cross(z).normalized(),axis,z)).transposed().to_4x4();matrix.translation=p.head.copy();p.matrix=matrix;bpy.context.view_layer.update()
            ev=m.evaluated_get(bpy.context.evaluated_depsgraph_get());correction=.004-min(ev.data.vertices[i].co.z for i in ids)
            if abs(correction)<1e-6:break
            target.z+=correction


def pose(name,t,previous):
    r=work.rig();work.reset()
    pitch,yaw,roll,grip,e,h=parameters(name,t)
    standing_legs(name,t)
    work.rotate('body',pitch,roll=roll,yaw=yaw);work.rotate('head',-pitch*.9,roll=-roll*.6,yaw=-yaw*.4)
    bpy.context.view_layer.update();target,axis=tool_frame(e,h)
    lower='R' if name=='paddle_right' else 'L';upper='L' if lower=='R' else 'R'
    grips={lower:grip,upper:grip+axis*SPACING}
    errors=[work.solve_arm(side,grips[side],axis,ARM_LENGTH,hinge=True,
              elbow_pole=(-1.2,.55,-1.8) if side==lower else (-.25,.8,-2.)) for side in ('L','R')]
    target.translation=grips['R'];r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update()
    es,previous=wrists.relax_current_pose(r,previous);errors.extend(es.values());raw={}
    for side in ('L','R'):
        u,f,hand=(r.pose.bones[n+'.'+side] for n in ('upper_arm','forearm','hand'))
        U,F,H=u.scale.y,f.scale.y,hand.scale.copy()
        raw[f.name]={'authored_length_ratio':F};raw[hand.name]={'authored_palm_scale_'+c:H[i] for i,c in enumerate('xyz')}
        u.scale=(1,U,1);r.pose.bones['elbow_scale.'+side].scale=(1,1/U,1)
        f.scale=(1,U*F,1);r.pose.bones['wrist_scale.'+side].scale=(1,1/(U*F),1);hand.scale=H*(U*F)
        for b in (f,hand):
            for key,value in raw[b.name].items():b[key]=value
    bpy.context.view_layer.update()
    values={p.name:{'location':tuple(p.location),'rotation_quaternion':tuple(p.rotation_quaternion),'scale':tuple(p.scale)} for p in r.pose.bones}
    return values,raw,previous,max(errors)


def loop_endpoint(first,previous):
    end=copy.deepcopy(first)
    for name in end:
        q=Quaternion(end[name]['rotation_quaternion'])
        if q.dot(Quaternion(previous[name]['rotation_quaternion']))<0:q.negate()
        end[name]['rotation_quaternion']=tuple(q)
    return end


def fix_quaternion_seam(action):
    # q and -q describe the same orientation. An elbow plane may wind around
    # during the stroke; preserve its continuous sign at the repeat endpoint.
    groups={}
    for c in work.curves(action):
        if c.data_path.endswith('rotation_quaternion'):groups.setdefault(c.data_path,{})[c.array_index]=c
    for channels in groups.values():
        first=Quaternion([channels[i].evaluate(1) for i in range(4)])
        end=Quaternion([channels[i].evaluate(LENGTH+1) for i in range(4)])
        sign=-1 if first.dot(end)<0 else 1
        for c in channels.values():
            ks=c.keyframe_points
            if len(ks)<2:continue
            h0,h1=ks[-1].co.x-ks[-2].co.x,ks[1].co.x-ks[0].co.x
            d0=(ks[0].co.y-sign*ks[-2].co.y)/h0;d1=(ks[1].co.y-ks[0].co.y)/h1
            slope=0.
            if d0*d1>0:
                w0,w1=2*h1+h0,h1+2*h0;slope=(w0+w1)/(w0/d0+w1/d1)
            for k,d in ((ks[0],slope),(ks[-1],sign*slope)):
                k.handle_left.y=k.co.y-d*(k.co.x-k.handle_left.x)
                k.handle_right.y=k.co.y+d*(k.handle_right.x-k.co.x)
            c.update()


def fit_tool(a):
    r=work.rig();name=a.name.split('_build')[0];desired={};targets={};last=None
    for f in sorted({1+k*.25 for k in range(LENGTH*4+1)}|{1+t*LENGTH for t in PHASES}):
        wrists.select(r,a,f);*_,grip,e,h=parameters(name,(f-1)/LENGTH)
        target,_=tool_frame(e,h);hand=r.pose.bones['hand.R'];target.translation=(hand.head+hand.tail)*.5
        targets[f]=target;r.pose.bones['tool.R'].matrix=target;bpy.context.view_layer.update();p=r.pose.bones['tool.R'];q=p.rotation_quaternion.copy()
        if last and q.dot(last)<0:q.negate()
        last=q;desired[f]={'tool.R':dict(location=tuple(p.location),rotation_quaternion=tuple(q),scale=tuple(p.scale))}
    keys={1+t*LENGTH for t in PHASES}
    for _ in range(12):
        times=sorted(keys);samples=[desired[f] for f in times];samples[-1]=loop_endpoint(samples[0],samples[-2]);wrists.write_samples(r,a,times,samples,['tool.R']);fix_quaternion_seam(a)
        worst={};maxpos=maxangle=0.
        for f,target in targets.items():
            wrists.select(r,a,f);p=r.pose.bones['tool.R'];distance=(p.head-target.translation).length
            angle=p.matrix.to_quaternion().rotation_difference(target.to_quaternion()).angle;angle=min(angle,abs(math.tau-angle))
            maxpos=max(maxpos,distance);maxangle=max(maxangle,angle);error=max(distance/.0006,angle/math.radians(1.))
            if error>1.001 and f not in keys:
                i=next((i for i in range(len(times)-1) if times[i]<f<times[i+1]),None)
                if i is not None and (i not in worst or error>worst[i][0]):worst[i]=(error,f)
        if not worst:break
        keys.update(f for _,f in worst.values())
    assert maxpos<.00061 and maxangle<math.radians(1.01),(name,maxpos,maxangle)
    return len(keys)


class Geometry:
    def __init__(self):self.v=[];self.f=[];self.i=[]
    def face(self,indices,material=0):self.f.append(tuple(indices));self.i.append(material)
    def box(self,lo,hi,material=0):
        x,y,z=lo;X,Y,Z=hi;i=len(self.v);self.v.extend([(x,y,z),(X,y,z),(X,Y,z),(x,Y,z),(x,y,Z),(X,y,Z),(X,Y,Z),(x,Y,Z)])
        for face in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:self.face([i+n for n in face],material)
    def object(self,name,colors):
        data=bpy.data.meshes.new(name+'_geometry');data.from_pydata(self.v,[],self.f);data.update();obj=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(obj)
        for suffix,color in colors:
            mat=bpy.data.materials.get(name+'_'+suffix) or bpy.data.materials.new(name+'_'+suffix);mat.use_nodes=True;mat.diffuse_color=(*color,1)
            sh=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');sh.inputs['Base Color'].default_value=(*color,1);sh.inputs['Roughness'].default_value=.8;data.materials.append(mat)
        for p,i in zip(data.polygons,self.i):p.material_index=i
        obj['preview_only']=True;obj['preview_actions']=','.join(NAMES);obj['remove_after_user_approval']=True
        return obj


def add_previews():
    r=work.rig()
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    for name in ('PREVIEW_Canoe','PREVIEW_Raft','PREVIEW_Plank','PREVIEW_WoodBoat','PREVIEW_PaddleLeft','PREVIEW_PaddleRight'):
        if bpy.data.objects.get(name):bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
    g=Geometry();sections=[(-1.05,.035,.31,.12),(-.82,.26,.25,-.015),(-.30,.4,.24,-.06),(.50,.4,.24,-.06),(1.12,.24,.28,.025),(1.45,.025,.34,.18)]
    for x,w,top,bottom in sections:
        for inner in (False,True):
            if not inner:cross=[(-w,top),(-w*.78,bottom+.07),(0,bottom),(w*.78,bottom+.07),(w,top)]
            else:cross=[(-max(.006,w-.028),top),(-max(.004,w*.78-.024),bottom+.10),(0,bottom+.035),(max(.004,w*.78-.024),bottom+.10),(max(.006,w-.028),top)]
            g.v.extend((x,y+.01,z) for y,z in cross)
    for j in range(len(sections)-1):
        a,b=j*10,(j+1)*10
        for k in range(4):g.face((a+k,b+k,b+k+1,a+k+1),j%2);g.face((a+5+k,a+6+k,b+6+k,b+5+k),2)
        g.face((a,a+5,b+5,b),3);g.face((a+4,b+4,b+9,a+9),3)
    for j in (0,len(sections)-1):
        a=j*10
        for k in range(4):g.face((a+k,a+k+1,a+6+k,a+5+k),3)
    g.box((-.25,-.27,-.025),(.36,.29,0),3)
    g.box((-.69,-.265,.17),(-.65,.285,.205),3);g.box((.81,-.29,.18),(.85,.31,.215),3)
    colors_boat=[('Hull',(.29,.115,.035)),('Planks',(.40,.19,.065)),('Interior',(.48,.255,.10)),('Rim',(.20,.075,.018))]
    boat=g.object('PREVIEW_Canoe',colors_boat);boat['preview_vessel']='CANOE'
    plank=Geometry();plank.box((-.42,-.23,-.055),(.58,.25,0),0)
    plank.object('PREVIEW_Plank',colors_boat)['preview_vessel']='PLANK'
    raft=Geometry()
    for i in range(7):
        y=-.385+i*.112;raft.box((-.72,y,-.075),(.92,y+.103,0),i%2)
    for x in (-.51,.61):raft.box((x,-.40,-.12),(x+.065,.42,-.075),3)
    raft.object('PREVIEW_Raft',colors_boat)['preview_vessel']='RAFT'
    # A broader, flat-stern wooden boat shares the same standing-floor height.
    skiff=Geometry();skiff.v=[(x,y,z) for x,y,z in g.v];skiff.f=list(g.f);skiff.i=list(g.i)
    skiff.v=[(max(-.88,x),.01+(y-.01)*1.20,z) for x,y,z in skiff.v]
    skiff.object('PREVIEW_WoodBoat',colors_boat)['preview_vessel']='WOOD_BOAT'
    scene=bpy.context.scene
    if hasattr(scene,'hero_paddle_vessel'):scene.hero_paddle_vessel='RAFT'
    scene['hero_paddle_vessel_choice']='RAFT'
    paddle=Geometry();paddle.box((-.013,-1.00,-.013),(.013,.70,.013),0)
    paddle.box((-.075,.66,-.018),(.075,.70,.018),1)
    outline=[(-.015,-.95),(-.09,-1.07),(-.095,-1.33),(-.06,-1.40),(.06,-1.40),(.095,-1.33),(.09,-1.07),(.015,-.95)]
    i=len(paddle.v);n=len(outline)
    paddle.v.extend((x,y,z) for z in (-.016,.016) for x,y in outline)
    paddle.face(tuple(i+j for j in reversed(range(n))),1);paddle.face(tuple(i+n+j for j in range(n)),1)
    for j in range(n):paddle.face((i+j,i+(j+1)%n,i+(j+1)%n+n,i+j+n),0)
    colors=[('Shaft',(.33,.15,.045)),('BladeGrip',(.57,.30,.10))]
    for side,name in (('L','PREVIEW_PaddleLeft'),('R','PREVIEW_PaddleRight')):
        obj=paddle.object(name,colors);obj['preview_actions']='paddle_left' if side=='L' else 'paddle_right'
        obj.parent=r;obj.parent_type='BONE';obj.parent_bone='tool.R';obj.matrix_parent_inverse=Matrix.Identity(4)
        obj.location=(0,-r.data.bones['tool.R'].length-(SPACING if side=='L' else 0),0);obj['attachment_socket']='tool.R';obj['grip_offset_m']=-SPACING if side=='L' else 0.
    return boat


def build():
    r=work.rig();s=bpy.context.scene
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero2_before_standing_paddling.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    preserved={a.name:action_fingerprint(a) for a in bpy.data.actions if a.name not in NAMES}
    results={};drivers=list(r.animation_data.drivers);driven={(c.data_path,c.array_index) for c in drivers}
    for name in NAMES:
        old=bpy.data.actions.get(name);r.animation_data.action=None;samples=[];properties=[];previous={};reach=0.
        try:
            for c in drivers:c.mute=True
            for t in PHASES:
                values,raw,previous,e=pose(name,t,previous);samples.append(values);properties.append(raw);reach=max(reach,e)
        finally:
            for c in drivers:c.mute=False
        assert reach<.001,(name,'Grip unreachable',reach)
        for i in range(1,len(samples)):
            for n in samples[i]:
                q=Quaternion(samples[i][n]['rotation_quaternion'])
                if q.dot(Quaternion(samples[i-1][n]['rotation_quaternion']))<0:q.negate()
                samples[i][n]['rotation_quaternion']=tuple(q)
        samples[-1]=loop_endpoint(samples[0],samples[-2]);properties[-1]=copy.deepcopy(properties[0])
        a=bpy.data.actions.new(name+'_build');a.use_fake_user=True;a.use_frame_range=True;a.frame_start=1;a.frame_end=LENGTH+1
        a['loop_frames']=LENGTH;a['looping']=True;a['description']='分腿微屈站立，'+('左侧' if name.endswith('left') else '右侧')+'转腰伸桨、后拉发力、抬桨回收'
        a['stance']='standing';a['foot_plane_z']=0.;a['stance_width_m']=.28
        a['author_pose_times']=len(PHASES);a['grip_spacing']=SPACING;a['lower_grip_hand']='L' if name.endswith('left') else 'R'
        a['water_entry_frame']=1+.24*LENGTH;a['water_exit_frame']=1+.58*LENGTH
        times=[1+t*LENGTH for t in PHASES];r.animation_data.action=a;wrists.write_samples(r,a,times,samples,list(r.pose.bones.keys()));fix_quaternion_seam(a)
        for layer in a.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for c in list(bag.fcurves):
                        if (c.data_path,c.array_index) in driven:bag.fcurves.remove(c)
        for bone,props in properties[0].items():
            p=r.pose.bones[bone]
            for key,value in props.items():p[key]=value;p.keyframe_insert('["'+key+'"]',frame=1,group=bone)
        for c in work.curves(a):
            if c.data_path.endswith(']'):
                bone,key=c.data_path.split('"')[1::2];write_property(c,times,[v[bone][key] for v in properties],True)
        tool_times=fit_tool(a)
        for c in work.curves(a):
            ys=[k.co.y for k in c.keyframe_points]
            if len(ys)>1 and max(ys)-min(ys)<2e-6:c.keyframe_points.clear();c.keyframe_points.insert(1,ys[0]);c.update()
        if old:bpy.data.actions.remove(old)
        a.name=name;a.asset_mark();results[name]={'frames':LENGTH,'fps':s.render.fps,'duration':LENGTH/s.render.fps,'author_pose_times':len(PHASES),'tool_pose_times':tool_times,'keys':sum(len(c.keyframe_points) for c in work.curves(a)),'max_grip_reach_error_m':reach}
    assert all(action_fingerprint(bpy.data.actions[n])==h for n,h in preserved.items())
    add_previews();work.set_action(NAMES[0],1)
    import hero_action_preview as preview
    preview.sync_scene(s,force=True)
    stats=json.loads(s.get('animation_clips','{}'));stats.update(results);s['animation_clips']=json.dumps(stats)
    report={'clips':results,'preserved_actions':list(preserved),'boat_preview_only':True,'paddle_preview_only':True,
            'stance':'standing','foot_plane_z':0.,'supported_preview_vessels':['PLANK','RAFT','CANOE','WOOD_BOAT']}
    (ROOT/'hero2_paddling_report.json').write_text(json.dumps(report,indent=2))
    print('PADDLING_CREATED',json.dumps(report));return report


if __name__=='__main__':build()
