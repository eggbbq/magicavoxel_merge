"""Project-local Blender add-on: follow the active hero Action's playback range."""
bl_info = {'name':'Hero Action Preview','author':'Project tools','version':(1,0,0),
           'blender':(5,0,0),'category':'Animation',
           'description':'Match preview playback to the active Action on opted-in hero scenes'}

import math
import bpy
from mathutils import Matrix,Vector,Quaternion
from bpy.app.handlers import persistent

_state = {}


def update_preview_objects(scene,rig):
    action=rig.animation_data.action if rig.animation_data else None
    if not action:return
    frame=scene.frame_current+scene.frame_subframe
    for obj in scene.objects:
        only_actions=obj.get('preview_actions');only_action=obj.get('preview_action')
        if not (only_actions or only_action):continue
        hidden=action.name not in only_actions.split(',') if only_actions else action.name!=only_action
        if obj.get('preview_vessel') and not hidden:
            choice=getattr(scene,'hero_paddle_vessel',scene.get('hero_paddle_vessel_choice','RAFT'))
            hidden=obj['preview_vessel']!=choice
        if obj.get('preview_projectile') and not hidden:
            release=float(obj['release_frame']);end=float(obj['flight_end_frame']);reset=float(obj['reappear_frame'])
            hidden=end<frame<reset
            if release<=frame<=end and obj.get('release_direction') is not None:
                elapsed=(frame-release)/(scene.render.fps/scene.render.fps_base)
                direction=Vector(obj['release_direction']);velocity=direction*float(obj['flight_speed'])
                gravity=float(obj['flight_gravity'])
                position=Vector(obj['release_position'])+velocity*elapsed+Vector((0,0,-.5*gravity*elapsed*elapsed))
                tangent=(velocity+Vector((0,0,-gravity*elapsed))).normalized()
                rotation=direction.rotation_difference(tangent)@Quaternion(obj['release_rotation'])
                matrix=rotation.to_matrix().to_4x4();matrix.translation=position;obj.matrix_world=matrix
            else:
                socket=obj.get('attachment_socket','tool.R')
                obj.rotation_mode='QUATERNION';obj.rotation_quaternion=(1,0,0,0);obj.scale=(1,1,1)
                obj.location=(0,-rig.data.bones[socket].length,0)
        if obj.hide_viewport!=hidden:obj.hide_viewport=hidden
        if obj.hide_render!=hidden:obj.hide_render=hidden


@persistent
def _on_frame(scene,*_):
    if not scene.get('hero_action_preview_auto',False):return
    rig=scene.objects.get(scene.get('hero_action_preview_rig','Hero'))
    if rig:update_preview_objects(scene,rig)


def sync_scene(scene, force=False):
    if not scene.get('hero_action_preview_auto',False):
        return None
    rig=scene.objects.get(scene.get('hero_action_preview_rig','Hero'))
    if not rig or not rig.animation_data or not rig.animation_data.action:
        return None
    action=rig.animation_data.action
    # Preview props stay in the authoring file while awaiting user review.
    # Their visibility follows the associated Action without adding animation keys.
    update_preview_objects(scene,rig)
    start=math.floor(action.frame_range.x)
    count=action.get('loop_frames')
    # These clips include a duplicate endpoint. Excluding it avoids a one-frame pause.
    end=start+int(count)-1 if count else math.ceil(action.frame_range.y)
    key=(action.as_pointer(),start,end)
    old=_state.get(scene.as_pointer())
    if not force and old==key:
        return None
    _state[scene.as_pointer()]=key
    scene.frame_start=start
    scene.frame_end=end
    scene.frame_preview_start=start
    scene.frame_preview_end=end
    scene.use_preview_range=True
    scene.frame_set(start)
    for window in bpy.context.window_manager.windows:
        if window.scene!=scene:
            continue
        for area in window.screen.areas:
            if area.type=='DOPESHEET_EDITOR' and area.spaces.active.mode=='ACTION':
                region=next((r for r in area.regions if r.type=='WINDOW'),None)
                if region:
                    with bpy.context.temp_override(window=window,area=area,region=region):
                        if bpy.ops.action.view_all.poll():bpy.ops.action.view_all()
                area.tag_redraw()
    return (start,end)


def _tick():
    if bpy.app.background:
        return 1.
    for window in bpy.context.window_manager.windows:
        sync_scene(window.scene)
    return .1


@persistent
def _on_load(_):
    _state.clear()
    if _on_frame not in bpy.app.handlers.frame_change_post:bpy.app.handlers.frame_change_post.append(_on_frame)
    if not bpy.app.background and not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick,first_interval=.1,persistent=True)


def _vessel_changed(scene,context):
    scene['hero_paddle_vessel_choice']=scene.hero_paddle_vessel
    rig=scene.objects.get(scene.get('hero_action_preview_rig','Hero'))
    if rig:update_preview_objects(scene,rig)


class HERO_PT_vessel_preview(bpy.types.Panel):
    bl_label='划船预览'
    bl_idname='HERO_PT_vessel_preview'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='角色动画'

    @classmethod
    def poll(cls,context):return context.scene.get('hero_action_preview_auto',False) and 'paddle_left' in bpy.data.actions

    def draw(self,context):
        self.layout.prop(context.scene,'hero_paddle_vessel',text='船型')
        self.layout.label(text='共用左右站立划桨动作')


def register():
    if not hasattr(bpy.types.Scene,'hero_paddle_vessel'):
        bpy.types.Scene.hero_paddle_vessel=bpy.props.EnumProperty(name='预览船型',
            items=[('PLANK','木板','木板上的站立划桨',0),('RAFT','木筏','木筏上的站立划桨',1),
                   ('CANOE','独木舟','独木舟内站立划桨',2),('WOOD_BOAT','小木船','小木船内站立划桨',3)],
            default='RAFT',update=_vessel_changed)
    if not getattr(bpy.types,'HERO_PT_vessel_preview',None):bpy.utils.register_class(HERO_PT_vessel_preview)
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    _on_load(None)


def unregister():
    if bpy.app.timers.is_registered(_tick):bpy.app.timers.unregister(_tick)
    if _on_frame in bpy.app.handlers.frame_change_post:bpy.app.handlers.frame_change_post.remove(_on_frame)
    if _on_load in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(_on_load)
    _state.clear()
    if getattr(bpy.types,'HERO_PT_vessel_preview',None):bpy.utils.unregister_class(HERO_PT_vessel_preview)
    if hasattr(bpy.types.Scene,'hero_paddle_vessel'):del bpy.types.Scene.hero_paddle_vessel
