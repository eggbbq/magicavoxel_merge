hero1 — LayaAir 3 角色动画资源

使用文件：hero1_laya.glb
编辑源文件：hero1_laya.blend
重新导出脚本：scripts/export_hero1_laya.py

GLB 已包含角色网格、贴图、材质、骨骼、蒙皮权重和 7 个动作。
动画已经绑定到这个角色，无需在 Laya 中重新绑骨骼。

在 Blender 中预览
  在大纲中选中 Hero（骨架图标），进入姿态模式，然后在动作编辑器切换动作。
  动作应绑定到 Hero 骨架上；HeroMesh 的用途是显示蒙皮模型。
  文件默认选中骨架并显示 Hero_Run，按空格播放。
  切换动作时，对应播放范围为：
    待机 1–72；行走 1–24；两种奔跑 1–16；跳跃 1–48；睡觉 1–96；挥砍 1–36。

资源结构
  1 个网格，1 个材质，1 张内嵌 PNG，7 根骨骼。
  GLB 中只有 8 个节点：HeroMesh 与 hips/body/head/arm.L/arm.R/leg.L/leg.R。
  六个角色身体部件已合并；每个顶点只受一根骨骼影响。
  没有树、斧头、眼皮道具、相机、灯光、导入空节点或道具控制曲线。
  睡觉保留身体动作，移除附加眼皮，因此没有道具闭眼效果。
  砍树保留角色挥砍动作，武器应由游戏另行挂接。

在 LayaAir 3 IDE 中使用
  1. 将 hero1_laya.glb 放进项目的 assets 目录，等待导入。
  2. 将模型拖进 Scene3D，展开资源查看其动画片段。
  3. 为角色根节点的 Animator 指定一个 AnimationController。
     如果模型根节点没有 Animator，手动添加一个。
  4. 将需要的动画片段拖进 Animator 状态机窗口，建立状态。
     把 Hero_Idle 设为默认状态，再按需要建立转换。
  5. 保存为预制体后，在游戏代码中获取该 Animator，按状态名播放。

若保留以下状态名，可这样调用（animator 为角色根节点上的 Laya.Animator）：
  animator.play("Hero_Idle");
  animator.play("Hero_Run_Exaggerated");
  animator.play("Hero_Jump");
播放参数是 AnimatorController 中的状态名；重命名状态后同步修改代码。

动作                     时长       用途
Hero_Idle                3.000 秒   待机
Hero_Walk_InPlace        1.000 秒   行走
Hero_Run                 0.667 秒   原来满意的平稳奔跑
Hero_Run_Exaggerated     0.667 秒   大幅跨步、腾空的夸张奔跑
Hero_Jump                2.000 秒   跳跃
Hero_Sleep               4.000 秒   睡觉身体动作
Hero_ChopTree            1.500 秒   挥砍动作

循环与移动
  在 Laya 的动画/状态设置中开启待机、行走、奔跑、睡觉循环。
  跳跃通常关闭循环；挥砍按游戏逻辑决定是否循环。
  行走和奔跑为原地动画，角色在地图中的移动由游戏控制器负责。
  模型按 glTF 标准 Y 向上导出，贴图已嵌入，无需另外拷贝 PNG。

验证
  Khronos glTF Validator：0 错误、0 警告、0 提示。
  Blender 回导：7 个动作、35 个姿态，最大位置误差约 0.00000025。
  尚未在实际 Laya 项目中运行验证。

官方导入说明
  https://layaair.com/3.x/doc-en/IDE/assets/model/readme.html
  https://layaair.com/3.x/doc-en/3D/useModel/readme.html
