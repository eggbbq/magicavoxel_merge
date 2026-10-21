# 游戏导出与 Laya 使用

## 精简副本

保留作者文件，用副本做合并和清理。记录必要网格、骨架、动作白名单及参考顶点；整理前后统一坐标比较。顶点重排时可双向比较最近点距离，同时检查蒙皮、UV 与材质，不能把形状接近当作所有属性正确。

合并同材质网格时，把各部件变换统一到骨架空间，保留 UV、面与权重。骨架 rest 数据与网格同时平移后，动画局部通道通常可保持，但仍要逐动作验证。

删除无关道具、包装空对象、道具驱动与属性动画，不删除必要的骨架祖先。只在确认材质没使用时删除 UV 层。打包实际使用的贴图，避免相对路径失效。

## Blender 5.2 导出选项

以下在本案例中验证过，使用前检查目标版本导出器的实际 RNA 属性：

```python
bpy.ops.export_scene.gltf(
    filepath=destination, export_format='GLB', use_selection=True,
    export_animations=True, export_animation_mode='ACTIONS',
    export_frame_range=False, export_anim_slide_to_zero=True,
    export_force_sampling=True, export_frame_step=1,
    export_optimize_animation_size=True, export_reset_pose_bones=True,
    export_skins=True, export_def_bones=False, export_leaf_bone=False,
    export_armature_object_remove=True, export_rest_position_armature=True,
    export_morph=False, export_morph_animation=False,
    export_cameras=False, export_lights=False, export_extras=False,
    export_yup=True, export_materials='EXPORT', export_image_format='AUTO',
    export_draco_mesh_compression_enable=False,
    export_meshopt_compression_enable=False,
)
```

- 不按当前场景范围裁剪，避免 Run 的短播放范围截断 Sleep/Idle。
- 每个动作保留首帧和重复周期终点，导出时间从零开始。
- `export_def_bones=False` 保留没有直接权重但控制子骨骼的髋骨。
- 单一根骨架、变换已处理时可移除 armature 对象包装，不能为省节点改变动画坐标系。
- 本例让 identity 变换的 mesh 与骨架在 Blender 中独立，移除骨架包装后 glTF 根节点为 mesh 和根骨骼。一个 mesh 加七根骨骼共八个 node；不是其他角色的固定标准。
- `NODE_SKINNED_MESH_NON_ROOT` 警告父节点变换不能按普通网格方式作用于蒙皮。应理顺坐标和根节点，不仅为消除警告强行删父节点。
- 禁用压缩扩展是此案例的兼容性选择，目标引擎确认支持后可另行启用。
- 作者稀疏关键帧与导出采样并不冲突，按实际体积和运行效果评估。

保存预览 `.blend` 时仅选骨架，设置 POSE、明显可见的 Action/Slot 和对应范围。导出前切回 OBJECT，选择必要骨架与网格。不能用给 mesh 分配骨架 Action 的方式“绑定动画”。

## 验证等级

1. 随附脚本：检查 glTF/GLB 层级、蒙皮和片段元数据，不是完整格式验证器。
2. Khronos glTF Validator：报告实际标准错误与警告。
3. Blender 新场景回导：核对片段与时长，逐动作比较多帧蒙皮位置。回导可能生成 Icosphere 骨骼显示对象，不是 GLB 混进了道具。
4. 实际引擎运行：检查 Animator、材质、方向和动作切换；需要真实项目，不能由前三级推断通过。

交付时写明执行到哪一级。

## LayaAir 3

按目标项目版本核对官方说明：

- [模型资源设置，支持 GLB/glTF](https://layaair.com/3.x/doc-en/IDE/assets/model/readme.html)
- [模型动画导入与 AnimatorController](https://layaair.com/3.x/doc-en/3D/useModel/readme.html)

通常将 GLB 放进 `assets`，拖入 Scene3D，给根节点 Animator 指定 AnimationController，把模型内片段拖进状态机。默认 Idle，按用途设置循环；`animator.play(stateName)` 使用状态机状态名。

导出的模型、骨骼和权重已完成绑定，状态机负责选择动作。换成另一套骨架需要结构一致或重定向。GLB 不通用保存所有引擎状态机、循环配置、道具可见性驱动或 Blender 任意自定义属性。

原地行走/奔跑的地图位移由游戏控制器负责。删除眼皮仍可保留睡觉姿态，但没有该道具闭眼效果；删除武器仍可保留角色挥砍动作。
