"""Check standing strokes, planted feet, grips and clearance on four vessels."""
from pathlib import Path
import json,math
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import hero1_work_actions as work
import hero_action_preview as preview
from create_hero2_paddling import NAMES,LENGTH,SPACING
from validate_hero1_till_preview import shaft_touches_palm


def validate(strict=True):
    r,m=work.rig(),work.mesh();s=bpy.context.scene
    def faces(prefixes):
        ids={v.index for v in m.data.vertices if any(m.vertex_groups[g.group].name.startswith(prefixes) for g in v.groups)}
        return [tuple(p.vertices) for p in m.data.polygons if all(i in ids for i in p.vertices)]
    armfaces=faces(('forearm.','hand.'));obstacles={n:faces((n,)) for n in ('body','head')}
    boatbvs={o.name:BVHTree.FromPolygons([o.matrix_world@v.co for v in o.data.vertices],[tuple(p.vertices) for p in o.data.polygons]) for o in s.objects if o.get('preview_vessel')}
    foot_ids={side:[v.index for v in m.data.vertices if v.co.z<.021 and any(m.vertex_groups[g.group].name=='shin.'+side for g in v.groups)] for side in ('L','R')}
    result={}
    for name in NAMES:
        paddle=bpy.data.objects['PREVIEW_PaddleLeft' if name.endswith('left') else 'PREVIEW_PaddleRight'];paddlefaces=[tuple(p.vertices) for p in paddle.data.polygons]
        report={'samples':LENGTH*4+1,'issues':{},'max_grip_line_error_m':0.,'max_grip_spacing_error_m':0.,'elbow_degrees':[0.,-180.],'max_wrist_swing_degrees':0.,'max_wrist_twist_degrees':0.,'blade_z_range':[100.,-100.],'blade_x_range':[100.,-100.],'body_pitch_range':[100.,-100.],'body_yaw_range':[100.,-100.]}
        first=None;blade_samples={};min_forearm_forward=1.;foot_starts={};foot_slide=0.;foot_z=[100.,-100.];knees=[180.,0.]
        for k in range(LENGTH*4+1):
            f=1+k*.25;work.set_action(name,f);preview.update_preview_objects(s,r);bpy.context.view_layer.update()
            vs=[v.co.copy() for v in m.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices]
            if first is None:first=vs
            ps=[paddle.matrix_world@v.co for v in paddle.data.vertices];pbv=BVHTree.FromPolygons(ps,paddlefaces)
            for label,fs in obstacles.items():
                bv=BVHTree.FromPolygons(vs,fs)
                if bv.overlap(BVHTree.FromPolygons(vs,armfaces)):report['issues'].setdefault('arm_'+label,[]).append(f)
                if bv.overlap(pbv):report['issues'].setdefault('paddle_'+label,[]).append(f)
            for boatname,boatbv in boatbvs.items():
                if boatbv.overlap(pbv):report['issues'].setdefault('paddle_'+boatname,[]).append(f)
            axis=(r.pose.bones['tool.R'].tail-r.pose.bones['tool.R'].head).normalized();hands={side:(r.pose.bones['hand.'+side].head+r.pose.bones['hand.'+side].tail)*.5 for side in ('L','R')}
            lower='L' if name.endswith('left') else 'R';upper='R' if lower=='L' else 'L';delta=hands[upper]-hands[lower]
            offset=-SPACING if name.endswith('left') else 0.
            for side in ('L','R'):
                if not shaft_touches_palm(r.pose.bones['tool.R'],r.pose.bones['hand.'+side],shaft_min=-.70+offset,shaft_max=.70+offset):
                    report['issues'].setdefault('hand_off_shaft_'+side,[]).append(f)
            report['max_grip_line_error_m']=max(report['max_grip_line_error_m'],delta.cross(axis).length)
            report['max_grip_spacing_error_m']=max(report['max_grip_spacing_error_m'],abs(delta.dot(axis)-SPACING))
            for side in ('L','R'):
                body=r.pose.bones['body'];forward=(body.matrix.to_quaternion()@body.bone.matrix_local.to_quaternion().inverted())@Vector((1,0,0))
                forearm=r.pose.bones['forearm.'+side]
                min_forearm_forward=min(min_forearm_forward,(forearm.tail-forearm.head).normalized().dot(forward))
                q=r.pose.bones['forearm.'+side].rotation_quaternion.to_euler('XYZ');angle=math.degrees(q.z)
                if not (-145<angle<0 and max(abs(q.x),abs(q.y))<.001):report['issues'].setdefault('elbow_'+side,[]).append(f)
                report['elbow_degrees']=[min(report['elbow_degrees'][0],angle),max(report['elbow_degrees'][1],angle)]
                sw,tw=r.pose.bones['hand.'+side].rotation_quaternion.normalized().to_swing_twist('Y')
                report['max_wrist_swing_degrees']=max(report['max_wrist_swing_degrees'],math.degrees(sw.angle));report['max_wrist_twist_degrees']=max(report['max_wrist_twist_degrees'],abs(math.degrees(tw)))
                for segment in ('upper_arm','forearm'):
                    scale=r.pose.bones[segment+'.'+side].scale
                    assert abs(scale.x-1)<1e-6 and abs(scale.z-1)<1e-6,('Arm thickens',name,f)
                knee=math.degrees(r.pose.bones['shin.'+side].rotation_quaternion.angle);knees=[min(knees[0],knee),max(knees[1],knee)]
                ankle=r.pose.bones['shin.'+side].tail.copy()
                if side not in foot_starts:foot_starts[side]=ankle
                foot_slide=max(foot_slide,math.hypot(ankle.x-foot_starts[side].x,ankle.y-foot_starts[side].y))
                z=min(vs[i].z for i in foot_ids[side]);foot_z=[min(foot_z[0],z),max(foot_z[1],z)]
                assert -.40<ankle.x<.54 and -.21<ankle.y<.23,('Foot outside smallest plank',name,side,f)
            assert .295<r.pose.bones['hips'].head.z<.302,('Standing hips collapse',name,f)
            blade=sum(ps[-16:],Vector())/16;blade_samples[f]=blade
            report['blade_z_range']=[min(report['blade_z_range'][0],blade.z),max(report['blade_z_range'][1],blade.z)]
            report['blade_x_range']=[min(report['blade_x_range'][0],blade.x),max(report['blade_x_range'][1],blade.x)]
            body=r.pose.bones['body'];rotation=body.matrix.to_quaternion()@body.bone.matrix_local.to_quaternion().inverted();yaw=math.degrees(rotation.to_euler('XYZ').z)
            up=(body.tail-body.head).normalized();pitch=math.degrees(math.atan2(up.x,up.z))
            for key,value in [('body_pitch_range',pitch),('body_yaw_range',yaw)]:report[key]=[min(report[key][0],value),max(report[key][1],value)]
        report['min_forearm_forward_dot']=min_forearm_forward
        report['max_foot_slide_m']=foot_slide;report['foot_clearance_m']=foot_z;report['knee_flex_degrees']=knees
        report['vessels_checked']=list(boatbvs)
        report['seam_position_error_m']=max((a-b).length for a,b in zip(first,vs))
        report['issues']={n:{'start':min(fs),'end':max(fs),'count':len(fs)} for n,fs in report['issues'].items()}
        if strict:
            assert not report['issues'],(name,report['issues'])
            assert min_forearm_forward>.08,('Elbow bends backwards instead of reaching forwards',name,min_forearm_forward)
            assert foot_slide<.003 and foot_z[0]>-.001 and foot_z[1]<.008,('Feet leave deck',name,foot_slide,foot_z)
            assert knees[1]<40,('Standing knees bend too far',name,knees)
            # The palm is 4 cm across and the shaft is 2.6 cm across; verify
            # actual shaft/cube contact above, as well as the grip-centre error.
            assert report['max_grip_line_error_m']<.025 and report['max_grip_spacing_error_m']<.022,(name,'hands leave paddle',report)
            assert report['seam_position_error_m']<1e-6
            assert report['max_wrist_swing_degrees']<35.1 and report['max_wrist_twist_degrees']<25.1
            assert report['blade_z_range'][0]<-.10 and report['blade_z_range'][1]>.18,('No dip/recovery',name,report)
            assert report['blade_x_range'][1]-report['blade_x_range'][0]>.5
            assert report['body_pitch_range'][1]-report['body_pitch_range'][0]>40
            assert report['body_yaw_range'][1]-report['body_yaw_range'][0]>50
        result[name]=report
    work.set_action(NAMES[0],1)
    for choice in ('PLANK','RAFT','CANOE','WOOD_BOAT'):
        s.hero_paddle_vessel=choice;preview.update_preview_objects(s,r)
        visible=[o for o in s.objects if o.get('preview_vessel') and not o.hide_viewport]
        assert len(visible)==1 and visible[0]['preview_vessel']==choice,('Vessel selector failed',choice)
    s.hero_paddle_vessel='RAFT'
    for name in ('till','harvest','idle'):
        work.set_action(name,1);preview.update_preview_objects(s,r)
        assert all(o.hide_viewport for o in s.objects if o.get('preview_vessel') or o.name in ('PREVIEW_PaddleLeft','PREVIEW_PaddleRight'))
    work.set_action(NAMES[0],1);preview.update_preview_objects(s,r)
    (Path(bpy.data.filepath).parent/'hero2_paddling_validation.json').write_text(json.dumps(result,indent=2))
    print('PADDLING_VALIDATED' if strict else 'PADDLING_DIAGNOSTIC',json.dumps(result));return result

if __name__=='__main__':validate()
