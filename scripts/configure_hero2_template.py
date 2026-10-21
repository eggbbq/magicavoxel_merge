"""Add preview-only tools and hero2's head clearance to the shared template.
Run after retarget_voxel_character.py; it does not modify any animation curves.
"""
from pathlib import Path
import json,sys
import bpy
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from retarget_voxel_character import action_fingerprint,export
import hero1_work_actions as work
import hero_action_preview as preview


def configure():
    s=bpy.context.scene;r=work.rig();m=work.mesh()
    assert Path(bpy.data.filepath).name=='hero2_animated.blend'
    if bpy.context.screen and bpy.context.screen.is_animation_playing:bpy.ops.screen.animation_cancel(restore_frame=False)
    backup=ROOT/'hero2_before_tool_previews.blend'
    if not backup.exists():bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
    fingerprints={a.name:action_fingerprint(a) for a in bpy.data.actions}
    if not bpy.data.objects.get('PREVIEW_Hoe'):
        objects_before=set(bpy.data.objects);actions_before=set(bpy.data.actions)
        with bpy.data.libraries.load(str(ROOT/'hero1_animated.blend'),link=False) as (src,dst):
            dst.objects=[n for n in src.objects if n.startswith('PREVIEW_')]
        for obj in dst.objects:
            s.collection.objects.link(obj)
            if obj.parent:
                bone=obj.parent_bone;obj.parent=r;obj.parent_type='BONE';obj.parent_bone=bone
        for obj in set(bpy.data.objects)-objects_before-set(dst.objects):bpy.data.objects.remove(obj,do_unlink=True)
        for action in set(bpy.data.actions)-actions_before:bpy.data.actions.remove(action)
    # The beard protrudes farther down/forward than hero1's hair. A 1-cm head
    # alignment correction clears the hoe follow-through without changing clips.
    if not s.get('hero2_head_clearance_v1'):
        for v in m.data.vertices:
            if any(m.vertex_groups[g.group].name=='head' for g in v.groups):v.co.x-=.01
        m.data.update();s['hero2_head_clearance_v1']=True;s['hero2_head_alignment_x_m']=-.01
    s['standard_animation_template']=True;s['animation_event_authoring']='engine'
    assert all(action_fingerprint(bpy.data.actions[n])==h for n,h in fingerprints.items())
    report_path=ROOT/'hero2_transfer_report.json'
    report=json.loads(report_path.read_text())
    report.update(head_alignment_x_m=-.01,standard_template=True,
                  preview_tools={o.name:o.get('preview_actions',o.get('preview_action')) for o in s.objects if o.get('preview_only')},
                  all_animation_curves_preserved=True)
    report_path.write_text(json.dumps(report,indent=2))
    export(ROOT/'hero2_animated.glb',True);export(ROOT/'hero2_skinned.glb',False)
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT');r.select_set(True);bpy.context.view_layer.objects.active=r
    work.set_action('idle',1);work.set_action('till',1);preview.sync_scene(s,force=True)
    bpy.ops.object.mode_set(mode='POSE');bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'hero2_animated.blend'))
    print('HERO2_TEMPLATE_CONFIGURED',json.dumps(report['preview_tools']))
    return report


if __name__=='__main__':configure()
