# hero1 案例

以下是已完成工作的参数与脚本索引，不是其他角色的固定配方。原项目 `magicavoxel_merge`；项目脚本包含具体对象名，复用前检查并映射，技能不依赖它仍位于原机器路径。

## 流程

1. 复用 `btp_vox.cli` 将 VOX 转成 glTF，输出当前目录，保留六个身体部件、贴图和 UV。
2. 修复扁平化节点的重复父引用：重新挂接后解除旧父 children，验证所有节点父节点唯一，再实际导入 Blender。
3. 用七根骨骼刚性蒙皮，左右共享 mesh 先复制，避免顶点组互相覆盖。
4. 分别制作行走、待机、跳跃、睡觉、两版奔跑与挥砍。
5. 调整播放范围、周期速度、起伏曲线及常量复位通道。
6. 合并单网格，移除道具和空节点，打包 GLB，标准校验与回导比对，再说明 Laya 状态机配置。

## 骨架和动作

```text
hips（无直接权重，负责整体控制）
├─ body
│  ├─ head
│  ├─ arm.L
│  └─ arm.R
├─ leg.L
└─ leg.R
```

| Action | 播放范围 | 时长（24 fps） |
|---|---|---|
| Hero_Idle | 1–72 | 3 秒 |
| Hero_Walk_InPlace | 1–24 | 1 秒 |
| Hero_Run | 1–16 | 0.667 秒 |
| Hero_Run_Exaggerated | 1–16 | 0.667 秒 |
| Hero_Jump | 1–48 | 2 秒 |
| Hero_Sleep | 1–96 | 4 秒 |
| Hero_ChopTree | 1–36 | 1.5 秒 |

各动作另有 N+1 帧作为重复终点。跳跃在游戏中通常单次播放。

## 已验证的经验

- Run 是 16 帧，但预览范围沿用挥砍的 36 帧，导致首尾跳步。
- 把摆幅、起伏全部压低，稳定了却缺少奔跑感；需要从轨迹连续性和腾空姿态处理。
- 曾缩短到 12 帧，用户反馈晃眼；统一变速回 16 帧后认可。这个频率只适用于本次反馈。
- 用户认可的 Hero_Run 保留，再另建 Hero_Run_Exaggerated。后者加强前倾、跨步、摆臂和腾空，保持相同频率。
- 旧 Walk 缺少完整复位通道，从 Sleep 切换后可残留躺姿。导出副本补齐缺失的常量通道。
- “所有动作失效”的现场状态：活动对象 HeroMesh，网格误挂 Hero_Run 且无有效 slot，真正 Hero 骨架仍绑 Idle。修复为只选 Hero、POSE、正确 Action/Slot 与播放范围，逐动作检查真实顶点运动。

## 产物与验证

`hero1_laya.glb` 约 100 KB，1 mesh、1 材质、内嵌 PNG、7 根骨骼、7 片段，共 8 个 node。作者网格 680 顶点、340 三角形。

移除树、斧头、眼皮、相机、灯光、旧空节点和道具控制通道，保留身体动作。

标准校验 0 错误、0 警告；7 片段 35 个回导姿态最大位置误差约 2.5e-7。未据此声称已在实际 Laya 项目运行通过。

## 项目脚本索引

- `export_hero1.sh`：VOX 导出命令。
- `scripts/animate_hero1.py`：骨架与行走。
- `scripts/animate_hero1_idle.py`、`animate_hero1_jump.py`：待机、跳跃。
- `scripts/simplify_hero1_jump.py`：关键帧精简与误差对比。
- `scripts/animate_hero1_more.py`：早期睡觉、奔跑、挥砍与道具。
- `scripts/refine_hero1_run.py`、`retime_hero1_run.py`：平滑循环与统一变速。
- `scripts/animate_hero1_run_exaggerated.py`：新增夸张版本。
- `scripts/export_hero1_laya.py`：精简、合并、补复位与导出。
- `scripts/validate_hero1_laya.py`：回导与蒙皮顶点对比。

这些脚本带有 hero1 专用的名字、关节、文件和坐标假设，不能直接运行在任意新角色上。
