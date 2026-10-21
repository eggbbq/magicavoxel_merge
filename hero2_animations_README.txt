hero2 标准动画模板

烧烤 grilling
192 帧 / 48 fps（4 秒）独立循环，第 193 帧重复起点，预览 1–192。
右手前伸烤串，来回翻面；左手轻扇火，随后抬起烤串查看，再放回火堆上方。
角色有俯身和抬头配合，双脚固定；手肘沿正常方向弯曲，手腕扭转受限，手臂粗细保持不变。
14 个全身主要姿态时间点；工具挂点单独保留 31 个补偿时间点，保证握柄与手掌贴合。
PREVIEW_Skewer 是实际握在右手的简易肉菜串，经 tool.R 跟随；PREVIEW_Campfire 为预览石圈木柴火堆。
两者只在 grilling 显示，只保留在作者文件，不进入角色 GLB。游戏中挂烤串时让握点沿模型局部 Y=0，串尖朝局部 +Y。
角色正前方 +X，火堆预览中心约 (0.72, 0.10, 0) 米；按实际场景火堆高度与位置安排角色站位。
原有 23 个动作曲线与蒙皮不变。脚本 scripts/create_hero2_grilling.py；检查 scripts/validate_hero2_grilling.py。
检查结果 hero2_grilling_validation.json，现场备份 hero2_before_grilling_live.blend。

虚弱喘息 exhausted_breathing
144 帧 / 48 fps（3 秒），两次喘息，第 145 帧重复起点，预览 1–144。
低头弓背、双手松垂；吸气时抬肩抬胸，呼气时上身下沉，呼气时间较长。
身体前倾 33–43°，头部整体下倾 49–65°，肩部上下变化约 5.6 厘米；双脚保持固定，膝盖轻屈。
保留 9 个主要姿态时间点、264 个标量关键帧（含所有常量复位通道），手腕中立、手臂原长度和粗细不变。
原有 22 个动作曲线不变。脚本 scripts/create_hero2_exhausted_breathing.py，检查 scripts/validate_hero2_exhausted_breathing.py。
现场备份 hero2_before_exhausted_breathing_live.blend；检查结果 hero2_exhausted_breathing_validation.json。

站立划船 paddle_left / paddle_right
两个独立循环动作，各 96 帧 / 48 fps（2 秒），第 97 帧重复起点，预览 1–96。
双脚分开约 28 厘米并略微前后错开，膝盖约 17–36° 微屈；脚底统一对齐 Z=0 承载平面，髋部约 0.298 米，小幅转移重心。
前探、转腰、后仰拉桨、抬桨回收；躯干前后摆幅约 42°、水平转动约 56°。
沿用已修正的手肘方向：上臂向下靠近身体，前臂向前握桨；手腕保持受限扭转。
每段保留 15 个全身姿势时间点；左侧工具补偿 30 个时间点 / 894 个标量关键帧，右侧 36 个时间点 / 918 个标量关键帧。

预览船型：木板、木筏、独木舟、小木船。右侧 N 侧栏 → 角色动画 → 划船预览 → 船型，可切换；默认木筏。
四种预览共用 Z=0 站立平面；独木舟与小木船留出中央站立位置，木板与木筏使用平面甲板。
同一套左右站立动作不依赖座椅。游戏中把角色根节点放在船内的站立挂点，使脚底对齐承载面。
实际船型的船宽、船舷及水面相对甲板高度差异较大时，按实际尺寸适配桨的长度或挂点偏移；四种简易预览的脚底稳定性和船桨避让均已检查。
船桨比坐姿版向下延长 30 厘米，保持上手握点间距不变；脚底基准下方约 0.21 米仍可划动，回收时桨叶抬起。
船与桨只在 paddle_left / paddle_right 显示，其他动作自动隐藏；游戏角色 GLB 不包含这些预览道具。
船桨通过 tool.R 跟随握点；游戏左侧桨模型沿挂点局部 Y 偏移 -0.64 米，右侧偏移 0 米，与预览一致。
入水参考帧 24.04，离水参考帧 56.68；音效与特效事件在引擎配置。
原有其他 20 个动作的曲线、时长及蒙皮保持原样，没有增加骨骼。
脚本 scripts/create_hero2_paddling.py；检查 scripts/validate_hero2_paddling.py，结果 hero2_paddling_validation.json。
坐姿版现场备份 hero2_before_standing_paddling_live.blend。

hero2_animated.blend：标准作者模板，包含 18 根骨骼、缝合后的蒙皮和 24 个独立动作，以及仅供预览的临时工具。
hero2_animated.glb：共享动画资源，包含角色、骨架、蒙皮、材质贴图和全部 24 个动作；不含临时工具。
hero2_skinned.glb：轻量角色资源，只含模型、骨架、蒙皮和材质贴图，不含动画，可在引擎内绑定共享动画。
character_animation_template.json：项目默认模板、片段长度与循环设置。
原始 hero2.blend 和 hero1_animated.blend 均保持原文件内容。

原模型正面为 -Y，已绕 Z 轴旋转 90°，统一为 +X 前进、+Z 向上；身体位置对齐模板。
手脚各段长度保持不变，横向位置微调 1–3 厘米对齐公共骨架。
只缝合肘、腕、膝的内部关节，肩部和髋部保持分件关系。
复用原有骨架、12 条手臂长度补偿驱动及全部动画曲线，动作未重新制作。
为避让翻土随挥中的胡须，头部对齐位置沿 X 向后调整 1 厘米，外形和动画曲线不变。

临时预览道具：
till：锄头。
harvest：长柄镰刀。
chop：斧头和正前方树干。
fishing / fishing_pull：鱼竿。
javelin_throw：标枪，出手后独立飞行并隐藏，重播后重新显示。
在 Action Editor 中切换动作，道具显示和播放范围自动同步。工具挂在 tool.R，游戏导出不包含工具。

游戏中的音效和特效在引擎内制作，通过动画事件在打击/出手时触发。
标准 GLB 不自动转换 Blender 时间标记为 Unity/Laya 原生动画事件。未在本文件里添加音效、粒子或引擎专用事件。
Unity/Laya 内配置事件与播放控制后，各同骨架角色共用动画资源。
此项目只验证了 Blender 和 GLB；未在实际 Unity/Laya 项目内运行。

生成脚本：
Blender -b --python scripts/retarget_voxel_character.py -- hero2.blend --template hero1_animated.blend --forward=-Y
Blender -b hero2_animated.blend --python scripts/configure_hero2_template.py

后续同结构角色默认使用 character_animation_template.json 指向的 hero2_animated.blend；先根据实际模型正面指定 --forward。
例：Blender -b --python scripts/retarget_voxel_character.py -- hero3.blend --forward=-Y

验证报告：hero2_transfer_report.json、hero2_animation_validation.json、hero2_animated_gltf_validation.json、hero2_skinned_gltf_validation.json。
