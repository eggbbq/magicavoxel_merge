# Blender 动作实现与诊断

## 坐标和评估

检查 mesh 变换、骨架 matrix_world、bone.matrix_local 和 pose bone 局部通道。体素部件原点可能与关节不同。

骨架对象坐标与角色坐标一致时，角色轴到骨骼局部旋转的关系为：

```python
rest = pose.bone.matrix_local.to_quaternion()
pose.rotation_mode = 'QUATERNION'
pose.rotation_quaternion = rest.inverted() @ desired_rotation @ rest
pose.location = rest.inverted() @ desired_translation
```

骨架对象有额外旋转/缩放时先转换到骨架空间；上例不是任意世界坐标变换的完整公式。

改变动作/帧后更新依赖图，读取 evaluated mesh 顶点及其 matrix_world。每段参考动画开始前复位骨骼，避免缺失通道使前一动作污染验证结果。

## Blender 5.x Action Slots

先检查运行版本的实际 bpy 属性。此次 Blender 5.2 验证过的写法：

```python
rig.animation_data_create()
rig.animation_data.action = action
rig.animation_data.action_slot = action.slots[0]  # 先确认是该骨架对应的 slot
curves = [curve for layer in action.layers for strip in layer.strips
          for bag in strip.channelbags for curve in bag.fcurves]
```

多骨架/多 slot 文件按目标 ID 选择 slot，不能直接采用第一个。

缺失复位通道分别按位置 `(0,0,0)`、四元数 `(1,0,0,0)`、缩放 `(1,1,1)` 补常量首帧。只插入缺失分量，不覆盖已有真实关键帧。

## 循环曲线与变速

24 fps、16 帧周期播放 1–16；第 17 帧为下周期首帧，导出时长 16/24 秒。播放到 36 再回 1 会错拍。

对称步伐每周期有两次腾空，检查两半周期。`max(0, sin(2*phase))` 有分段尖点，可能造成步内起伏不匀；`sin(phase)**2` 能提供对称平滑起伏，但仍需与支撑脚/摆腿相位同步，不能仅替换公式就认定不穿地。

逐帧根据“当前最低脚”修正髋部，在支撑脚交替时可能引入速度突变。优先设计连续髋部曲线与支撑阶段，在接触点校准脚底，再密集抽样检查接触与地面间距。

相同端点值不保证相同速度。采用周期感知手柄，或为首尾 FREE 手柄设置相同周期导数。Hermite 段的手柄横距为段长/3，纵距为导数乘横距；不等间距关键点分别计算左右段长。

比较 `(f(start+eps)-f(start))/eps` 与 `(f(end)-f(end-eps))/eps`，并检查实际骨骼/网格。曲线数值误差不等于可见空间误差。

统一变速对关键点与两个手柄都应用 `new_time=start+(old_time-start)*factor`，再同步播放范围。只修改场景 End 不会给动作变速。

## 动作辨识度

- 待机：轻微呼吸和头部变化，脚底稳定。
- 行走：交替接触与通过，手脚反相。
- 奔跑：前倾、支撑/腾空、摆腿回收；频率、步幅与起伏分别调整。
- 跳跃：蓄力、离地、顶点、落地缓冲与恢复。
- 睡觉：整体躺姿、轻呼吸；是否需要闭眼由用户范围决定。
- 挥砍：预备、蓄力、击打、回收；动作与预览树/武器分别管理。

直臂直腿的简化骨架可用缩短/抬升摆动腿暗示收腿，但真实屈膝/屈肘需要相应关节与蒙皮；按动作需求选择骨架，增骨后还需验证旧动作复位。

## “所有动画不动”的检查顺序

1. 实际打开的文件是否正确，是否仍在查看旧文件？
2. 活动对象是骨架还是 mesh？Action 是否误挂到 mesh，曲线却指向 `pose.bones[...]`？
3. 骨架 Action/Slot 是否有效，是否仍绑定微动的 Idle？
4. 是否真正播放、范围是否对应、骨架是否为 POSE 而非 REST？
5. Armature 修改器是否启用、指向正确，权重和顶点组是否存在？
6. 曲线是否控制正确骨骼，有无 NLA/驱动/约束干扰？

修复后逐动作读取变形网格验证。最后仅选骨架、进入姿态模式、保存 Action/Slot/范围，再开始现场播放。
