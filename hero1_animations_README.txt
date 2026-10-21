hero1 动画

标枪投掷 javelin_throw（整段重做）
120 帧 / 48 fps（2.5 秒），游戏设为单次。自然站立 → 后拉蓄力 → 跨步转髋出手 → 随挥放臂 → 收脚直立。
以肩关节和正常单轴肘屈曲制作手臂，全段长度/粗细均为原始比例，手腕保持中立，不使用长臂 IK 拉扯。
收势从第 77 帧起肘部不超过约 7°，第 105 帧完整回到自然站立并保持至第 121 帧；站立时肘部约 5°、双臂接近垂直、双膝伸直。
起止角色姿态一致，可在 Blender 重播 1–120；开始时持枪，出手后空手收势。
后仰约 23°，出手后前倾约 38°，腰部摆动约 100°；右手前后行程约 1.2 米。前脚抬高约 10 厘米、后腿约 13 厘米。
18 个全身姿势时间点（含脚底接触校正）；工具补偿精简后共 28 个时间点、913 个标量关键帧。
保留 PREVIEW_Javelin 简易标枪，持枪时挂在 tool.R；第 43 帧出手并独立飞行，飞出视野后隐藏，重播到起点再显示。
游戏在第 43 帧（片段时间 0.875 秒）触发武器脱手/发射；预览飞行效果和标枪不包含在角色 GLB 中。
角色 GLB 包含角色、贴图、材质、蒙皮与 20 个动作；原有其他 19 个动作的关键帧、手柄和时长完整保留。
当前默认预览 javelin_throw。重做前现场备份 hero1_before_throw_rebuild_live.blend；重建 scripts/create_hero1_javelin_throw.py。
检查 hero1_javelin_throw_validation.json：握枪、关节、身体/头部避让、脚底接触、跨步支撑、自然站立收势、脱手和显示切换。

长柄镰刀收割 harvest
新增 harvest，72 帧 / 48 fps（1.5 秒）原地循环，第 73 帧复制第 1 帧。
9 个全身姿势时间点；已精简工具挂点补偿，合计 27 个时间点、640 个标量关键帧。
双手前后握长柄，分腿微屈，髋部与腰部转动带动低位横扫；扫完后抬刀回收，首尾衔接。
双手在上版基础上再前伸 4 厘米、降低 6 厘米，相较初版累计前伸 9 厘米；手臂仅沿长度方向舒展，粗细和手掌尺寸保持原样。
扫割阶段刀刃离地约 6–12 厘米，横向摆动约 104°；脚间距约 34 厘米，膝盖约 24–41° 屈伸。
新增 PREVIEW_Scythe 简易长柄镰刀，挂在 tool.R，只有 harvest 显示，保留供用户预览；角色 GLB 不包含镰刀或其他预览道具。
复用现有 18 根骨骼，骨架、网格、权重和材质不变，原有 18 个动作的关键帧与手柄保持原样；当前共 19 个动作。
选择 harvest 可预览收割；在 Action Editor 切换其他动作时，播放范围和预览道具自动跟随。
新增前现场备份 hero1_before_harvest_live.blend；重建脚本 scripts/create_hero1_harvest.py。
前伸修改前的现场备份 hero1_before_forward_harvest_live.blend。
本次降低并继续前伸之前的现场备份 hero1_before_lower_forward_harvest_live.blend。
检查结果 hero1_harvest_validation.json，含握柄、关节、穿模、刀刃高度、脚底接触和循环；导出仍为纯角色骨骼动画。

翻土关键帧精简
till 时间轴关键点由 97 处降到 26 处，总标量关键帧由 2001 个降到 990 个（约减少 51%）。
保留 16 个角色姿势时间点和原有 72 帧 / 48 fps（1.5 秒）节奏，合并常量通道、压缩锄头挂点曲线。
577 个抽样姿态对比：角色网格差异小于 0.001 毫米，锄头顶点最大轨迹差异约 7.2 毫米。
握柄、关节、落点、穿模和循环检查通过；其余 17 个动作的关键帧与手柄保持原样。
备份 hero1_before_till_key_reduction_live.blend；脚本 scripts/simplify_hero1_till_keys.py；报告 hero1_till_key_reduction.json。
选择 till 时显示锄头并预览翻土。

垂钓与拉杆优化
fishing 双手降低约 10 厘米、适当前伸到腰腹前方，保持轻微呼吸等待；掌心高度约 0.55–0.59 米。
fishing_pull 与 fishing 第 1 帧姿势完全一致，结束回到同一姿势；等待呼吸任意相位与起点的最大网格差异约 5 毫米。
拉杆最高掌心高度约 1.14 米，鱼竿最高倾角约 82°，身体后仰约 18°；相较上一版抬手提高约 20 厘米。
拉杆从 64 帧延长到 76 帧（48 fps，1.33 秒到约 1.58 秒，增加约 19%），游戏中设为单次播放后回到 fishing。
肘部使用正常单轴屈曲，手腕保持受限扭转；只沿手臂长度方向伸展，手臂粗细和手掌尺寸不变。
等待 9 个全身姿态时间点，拉杆 12 个全身姿态时间点；保留稀疏作者关键帧，工具挂点另作必要补偿。
修改前现场备份 hero1_before_fishing_refinement_live.blend；重建脚本 scripts/refine_hero1_fishing.py。
检查结果 hero1_fishing_refinement_validation.json，已验证完整动作中的握点、关节、穿模和首尾衔接。
在 Action Editor 选择 fishing_pull 或 fishing 可预览钓鱼动作；本次钓鱼制作保留其余 16 个动作，包括已认可的 chop。
保留简易鱼竿 PREVIEW_FishingRod（竿柄、卷线轮、分段细杆），挂在 tool.R 骨骼上，便于检查握杆与拉杆动作。
鱼竿仅在 fishing 和 fishing_pull 显示，切换其他动作自动隐藏；仅供 Blender 预览，不加入角色 GLB。
鱼竿脚本 scripts/preview_hero1_fishing_rod.py，握柄/穿模/显示切换检查 hero1_fishing_rod_preview_validation.json。

伐木斧头预览（待确认）
chop 改为砍正前方树干：侧后方蓄力、平斧砍入、卡住停顿、向后拔出、回收。
52 帧 / 48 fps（约 1.08 秒）循环，11 个全身姿势时间点，斧头补偿 22 个时间点，总标量关键点 710。
握法为右手靠斧头、左手靠柄尾，两手沿斧柄前后相隔约 19 厘米；整段验证双臂无相交、左右手顺序不交换。
侧后方收斧角度增加到约 205°，砍入时为 45°，挥砍弧度约 160°（上一版约 130°）；身体由后仰 14°/转腰 74°快速转为前倾 24°/反向转腰 24°。
蓄力后用 4 帧（约 0.083 秒）快速挥砍，击中停住 10 帧（约 0.21 秒），短暂停顿后用 4 帧迅速向后拔斧，再用 12 帧（0.25 秒）回收。
斧刃停在树干前表面内约 2.2 厘米，向后拔出约 6.5 厘米；挥砍和拔斧速度约为上一版 1.9 倍。
节奏脚本 scripts/retime_hero1_chop_rhythm.py 仅重排原有关键帧与曲线手柄，不增加关键帧；原姿势/轨迹变化小于 0.002 毫米。
双腿分开，脚间距约 34 厘米；下肢随蓄力、挥砍、拔斧转移重心，髋部转动约 22°、左右移动约 3.8 厘米。
膝盖约 23–40° 屈伸，击中时下沉，拔斧时回弹；脚底已校准到地面，插值中的脚部水平偏移不超过约 3.4 毫米。
已验证接触停顿、拔出、握柄、平持、手腕/手肘、身体/头部碰撞以及循环，其余 17 个动作保留。
关键帧精简脚本 scripts/simplify_hero1_chop_keys.py，重建 chop 时也会自动执行精简。
精简前备份 hero1_before_chop_key_reduction.blend；对比结果 hero1_chop_key_reduction.json。
保留低位前伸的起始、砍入和拔斧姿势，蓄力时向侧后方收手，为加大的水平挥砍避让身体；双手高度约 0.51–0.60 米。
只沿长度方向伸展手臂，粗细与手掌尺寸保持不变；前方预览树干相应前移 6 厘米，维持原接触深度和拔出距离。
保留 PREVIEW_Axe 简易斧头，挂在 tool.R；新增 PREVIEW_Trunk 标出正前方接触位置。
斧头和树干仅在 chop 显示，切换 till 显示原简易锄头。GLB 不包含这些道具。
本次加快挥砍和拔斧前的现场备份 hero1_before_faster_chop_rhythm_live.blend。
扩大挥砍弧度前的现场备份 hero1_before_larger_chop_arc_live.blend。
降低和前伸持斧位置之前的现场备份 hero1_before_lower_forward_chop_live.blend。
加入下肢跟随之前的备份 hero1_before_chop_lower_body_follow.blend。
扩大握距前备份 hero1_before_wider_chop_grip.blend；改成前后握斧前备份 hero1_before_power_chop_grip.blend。
加入树干与接触停顿之前的备份 hero1_before_trunk_contact_chop.blend。
脚本 scripts/preview_hero1_chop.py；上一版竖劈备份 hero1_before_horizontal_chop.blend。
斧头放平、加大弧度之前的横劈版备份 hero1_before_flat_wide_chop.blend。
降低持斧高度之前的版本备份 hero1_before_lower_chop.blend。
最初举斧修改前备份 hero1_before_chop_axe_preview.blend。
验证 hero1_chop_preview_validation.json、hero1_chop_axe_report.json。

手臂粗细修正
全部 18 个动作的上臂、前臂恢复原始横截面，只沿骨骼长度方向伸展。
举手高度、肘部与手掌位置、工具挂点轨迹保持不变；身体、头部、腿部和片段时长不变。
新增左右肘部、腕部共 4 根非变形缩放补偿骨骼，用于避免长度缩放传递后产生剪切。
骨骼总数现为 18；这些辅助骨骼没有网格权重，属于必要的关节控制节点。
作者文件使用简单表达式驱动保持原有稀疏插值，GLB 已烘焙成普通骨骼动画，无需游戏执行驱动。
脚本 scripts/fix_hero1_arm_thickness.py，修正前备份 hero1_before_arm_thickness_fix.blend。
检查结果 hero1_arm_thickness_validation.json；已比较 1410 个姿势的关节、手掌和工具位置。

手腕统一修正
全部 18 个动作已检查：手掌相对前臂的弯曲限制为 35°，轴向扭转限制为 25°。
手腕仍使用原焊接网格；修正后接缝横截面最小保留约 92.6%，避免被大角度旋转挤成细脖子。
原有身体、头部、腿部曲线和片段时长保持不变；手臂配合重新求解，主要姿势的手掌中心保持不变。
手腕这一步未增加网格、材质或骨骼。行走、奔跑、跳跃、待机等原本无需修正的手腕保持原样。
手臂关键姿势数量保持原数量；tool.R 使用独立的自适应补偿曲线以保留原工具轨迹。
工具朝向由挂点补偿，避免用手掌大角度扭转来控制工具。
修正前备份：hero1_before_wrist_relaxation.blend。
脚本：scripts/relax_hero1_wrists.py（现有文件已应用，重复运行会拒绝）。
检查结果：hero1_animation_validation.json、hero1_wrist_relaxation_report.json、hero1_wrist_preservation_validation.json。

当前工具预览（待确认）
hero1_animated.blend 保留简易锄头、斧头、鱼竿、镰刀和标枪，当前默认播放 javelin_throw；预览道具按动作自动显示。
切换到 till 时显示锄头，其他动作自动隐藏，等待用户确认后再删除。
持久工具挂点 tool.R，父骨骼 hand.R，原点在右手掌中心；含缩放补偿骨骼共 18 根。
锄头沿挂点局部 +Y 延伸，刃口朝局部 -X；Blender 骨骼父级以骨尾为参考，
预览模型沿局部 Y 偏移 -0.04 米以对齐骨头原点，游戏骨骼原点挂接时无需这个补偿。
等待用户确认后才删除锄头，tool.R 将保留用于游戏挂接工具。
hero1_animated.glb 已同步更新：18 根骨骼（含 tool.R 及 4 根缩放补偿骨骼）、18 个动作，不包含锄头。
锄头挂点验证见 hero1_tool_preview_validation.json。
翻土低位双臂长度增加 25%，举高时进一步伸展；双手超过头顶、双肘只保留约 25–30° 屈曲。
手掌尺寸基本不变，其他动作保持原比例。
平时握柄降低到腰腹附近，蓄力时明显抬锄到头顶上方，身体后仰约 32°，再快速下挖。
上臂从肩部明显抬起，肘部朝外下侧、沿单一轴向屈曲，前臂配合握柄；下挖时身体前倾最多 42°。
翻土仍为 1.5 秒循环，使用 9 个主要姿势及 7 个挥动握持校正姿势，共 16 个时间点。
翻土握持、碰撞、落点与循环验证见 hero1_till_arm_validation.json。

hero1_animated.blend：可编辑骨骼和稀疏关键帧，当前默认预览 javelin_throw，标枪只在此动作中显示。
hero1_animated.glb：模型、贴图、材质、蒙皮和二十个动作，供游戏导入。
原始 hero1.blend 未覆盖；现场制作前备份为 hero1_before_animation.blend。

Blender 预览
1. 打开 hero1_animated.blend，按空格播放/暂停。
2. 下方 Action Editor 的动作名称下拉框选择 idle、walk、walk_fast、run、jump、sleep、fishing、fishing_pull、chop、till、harvest、javelin_throw、sitting、eating、clutch、crouching、hands_over_head、hold_up、stomach_pain 或 surrendering。
3. 已在这台电脑启用项目的 Hero Action Preview 辅助工具。切换动作后会自动更新
   场景和预览范围，并回到首帧；所有动作均从第 1 帧开始。
   预览使用 Frame Dropping，尽量按真实时间播放，避免卡顿时把动作整体拖慢。
   不要把骨架动作分配给 HeroMesh；活动对象应为骨架 Hero，处于 Pose Mode。

动作        预览帧范围    时长（48 fps）
idle        1–144        3.00 秒，循环
walk        1–64         1.33 秒，循环，原地行走
walk_fast   1–40         0.83 秒，循环，保留上一版低抬腿的快步动作
run         1–25         0.52 秒，循环，保留认可姿势，速度降低 20%
jump        1–96         2.00 秒，蓄力、跳起、落地和恢复
sleep       1–192        4.00 秒，躺姿呼吸循环
fishing     1–144        3.00 秒，低位前伸双手、轻微呼吸的垂钓等待循环
fishing_pull 1–76        1.58 秒，衔接垂钓、抬高拉杆、后仰收力、回到等待，单次
chop        1–52         1.08 秒，大幅蓄力、快速砍入前方树干、短暂停顿、快速拔出、回收循环
till        1–72         1.50 秒，翻土抬锄、下压、回拉循环
harvest     1–72         1.50 秒，长柄镰刀低位横扫收割、抬刀回收循环
javelin_throw 1–96       2.00 秒，标枪后拉、跨步蹬地、转腰过顶投出、前倾收势，单次
sitting     1–144        3.00 秒，坐姿保持、轻微呼吸循环，游戏中配合座椅
eating      1–120        2.50 秒，左手托食、右手取食送入口边、回收循环
clutch      1–144        3.00 秒，双手捂腹、微微蜷缩与呼吸循环
crouching   1–96         2.00 秒，蹲姿保持、前倾平衡与呼吸循环
hands_over_head 1–144    3.00 秒，双手举高、肘部外展、掌心向前的投降/防护循环
hold_up     1–120        2.50 秒，双掌向上、向前上方托举的承重循环
stomach_pain 1–48        1.00 秒，弓身屈腿、右手捂腹、左手垂下，全身发抖
surrendering 1–48        1.00 秒，低头蹲下、双手抱头，全身发抖

循环动作的 End+1 帧与首帧相同，导出时保留这一帧。
jump 的第 97 帧为恢复完成的重复终点，Blender 预览 1–96。fishing_pull 的第 77 帧为复位终点，Blender 预览 1–76。两者在游戏中均设为不循环。
为精确减速 20%，作者文件使用 48 fps 时间基准。关键帧时间坐标相应调整，
关键姿势数量不增加，除 run 外所有动作的实际时长保持不变。
角色前进方向：Blender +X，Z 向上；GLB 按 glTF Y 向上导出。

新增疼痛与害怕动作
stomach_pain：右手固定在腹部、左手自然垂下，弓身屈腿，每秒约 3 次小幅颤动。
surrendering：低头蹲下，双手跟随头部抱住头侧，每秒约 4 次小幅颤动。
两段保持循环，分别 13、17 个姿势时间点（含重复终点）；没有增加骨骼或道具。
原有 16 个动作保留。

举手动作
hands_over_head 和 hold_up 各 9 个关键姿势时间点（含重复终点）。
前者双手在头侧上方、掌心朝前；后者双手向前上方托举、掌心朝上。
两段均使用单向屈肘、轻微呼吸或承重起伏；没有新增道具或骨骼。
原有 14 个动作保留。

生活动作
此前新增 sitting、eating、clutch、crouching，每段各 9 个姿势时间点（含重复终点）。
复用现有骨架、蒙皮和 tool.R 挂点；本轮开始前的 10 个动作不作改动。
sitting 是坐姿保持而非坐下过渡，适合游戏中配合座椅；crouching 是蹲姿保持。
eating 不含食物或餐具；clutch 根据确认制作成捂腹动作。

工作动作
垂钓、拉杆、伐木各 9 个关键姿势时间点，翻土为 16 个（含挥动校正点与重复终点）。
已有六段动作的关键帧、插值手柄和时长完全保留。
已用临时鱼竿测试双手握持和拉杆路径，测试后删除；GLB 只含角色，待确认锄头仍留在作者文件中。
游戏中把鱼竿、斧头、锄头挂在 tool.R；工具杆沿挂点局部 +Y。
鱼竿动作的左手握点在右手前方 0.08 米。
斧头和锄头同样在游戏内挂接；GLB 不包含这些道具。

绑定与资源
18 根骨骼：髋部、身体、头部、左右上臂/前臂/手掌、左右大腿/小腿、tool.R 工具挂点，
以及左右 elbow_scale、wrist_scale 共 4 根非变形缩放补偿骨骼。
GLB：1 个蒙皮网格、1 个材质、内嵌贴图；没有树、武器、相机、灯光或导入包装节点。
作者 .blend 另保留待确认锄头、斧头和接触参考树干，它们有独立材质，不随角色导出。
手肘、手腕和膝盖已移除内部端面并焊接为连续网格，局部支撑环使用过渡权重。
肩部和髋部仍保持原分件；其余区域保留刚性权重。跳跃时双臂向外侧抬起。
Blender 作者文件保留稀疏关键帧，GLB 按帧采样，以便游戏兼容。
行走、奔跑各 8 个主要姿势加重复终点；跳跃为 11 个主要姿势加 3 个仅调整髋部高度的接触校正点。
行走手臂只前后摆动。行走大腿摆幅不超过 18°、屈膝不超过 26°；
walk_fast 大腿摆幅不超过 30°、屈膝不超过 40°；
新 run 身体前倾 34–40°，上臂相对身体前摆 125°、后摆 55°，步频为 walk_fast 的 1.6 倍。
大腿前摆最多 50°、后蹬最多 72°，后侧收脚时屈膝最多 80°，支撑时仅轻微屈膝。
大腿和膝盖仍保持在髋部以下，避免重新抬入上半身。

项目脚本（在项目目录运行）
新四动作构建：scripts/hero1_life_actions.py（需要原 10 个动作和现有骨架；已有同名动作时会拒绝重复构建）。
只导出角色：
/Applications/Blender.app/Contents/MacOS/Blender -b hero1_animated.blend --python scripts/export_hero1_actor.py
验证当前 18 动作及工具挂点回导：
/Applications/Blender.app/Contents/MacOS/Blender -b hero1_animated.blend --python scripts/validate_hero1_animations.py
新增举手动作脚本：scripts/hero1_raised_hands_actions.py（需要已有 14 动作；同名动作存在时会拒绝重复构建）。
新增疼痛/投降动作脚本：scripts/hero1_distress_actions.py（需要已有 16 动作；同名动作存在时会拒绝重复构建）。
本轮前备份：hero1_before_distress_actions.blend。

验证结果见 hero1_animation_validation.json 和 hero1_gltf_validation.json。
已验证 Blender 网格运动、首尾接缝、动作切换及 GLB 回导；未在实际 Laya 项目中运行。
导入 Laya 后为 Animator 配置十八个同名状态；地图位移由游戏控制器负责。

自动预览范围辅助工具
源码在 scripts/hero_action_preview.py，通过 Blender 用户 Add-ons 目录中的软链接启用。
它仅作用于带 hero_action_preview_auto 标记的场景，不更改其他文件或 Python 安全设置。
这台电脑重新启动 Blender 后仍会启用；若把项目复制到别的电脑，需要在 Blender 中
安装并启用这个 .py 附加组件，才会自动跟随动作更新播放范围和预览道具可见性。
