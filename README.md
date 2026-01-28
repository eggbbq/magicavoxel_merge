# MagicaVoxelMegre 使用说明（MagicaVoxel → GLB）
废话: 该程序完全有windsurf编写。起因是我最近正在制作体素风格的游戏。目前最好用的体素建模软件依然是MagicaVoxel。
但是MagicaVoxel 不提供模型导出优化，很难直接用于游戏资源。在我陆陆续续尝试了无数的体素网格优化工具之后, 我花了很多时间去给开源作者测试bug， 提出修复建议，问题始终得不到很好的解决。
于是我就决定用windsurf帮我编写一个工具，事实证明windsurf工作得很出色，它只花费了几个小时（其中大部分时间花费在我逐步拆解需求，和测试软件），并且它做得非常出色，我从头到位并没有编写一句代码。

大纲: 该程序只导出glb(gltf2.0 的二进制文件)，因为我是做游戏的，gltf2.0 是事实上的标准。OBJ格式因为它有缺陷，所以就不管它。cocos creator 3x 完美的支持gltf导入。Unity 默认不支持，但是只需要下载一个插件即可玩么支持gltf文件导入。FBX由于生成麻烦所以也放弃了。实在需要OBJ/FBX 借助Blender 转换即可。


本工程用于将 MagicaVoxel 的 `.vox` 文件导出为 glTF 2.0 `.glb`，并提供：

- `palette` / `atlas` 两种贴图策略
- `atlas-style baked`（烘焙型 atlas：尽量少面数）
- 坐标轴与左右手系转换（适配 Cocos Creator / Blender 等）
- 可选顶点焊接（weld）
- 可选导出“平均法线”到顶点属性（`COLOR_0` 或 `TANGENT`）
- 批量转换脚本与 Blender 无界面重拓扑脚本

---

## 1. 快速开始

### 1.1 单文件转换

在工程根目录运行：

```bash
python -m magicavoxel_merge input.vox output.glb
```

### 1.3 一条完整调用示例（atlas + baked + by-model，推荐）

下面是一条包含常用参数的完整命令，你可以直接复制后按需改动：

```bash
python -m magicavoxel_merge \
  input.vox output.glb \
  --mode atlas \
  --atlas-style baked \
  --atlas-layout by-model \
  --merge-strategy maxrect \
  --atlas-texel-scale 1 \
  --atlas-pad 0 \
  --atlas-inset 0 \
  --baked-dedup \
  --cull-mv-z 0 \
  --no-atlas-square \
  --axis y_up \
  --handedness left \
  --scale 0.02 \
  --center \
  --weld \
  --avg-normals-attr none
```

如果你希望导出外置 PNG（而不是内嵌在 glb 中），加：

```bash
--texture-out output.png
```

### 1.2 批量转换（推荐）

编辑根目录的 `batch_convert.sh`，设置 `IN_DIR / OUT_DIR / JOBS` 和导出参数，然后运行：

```bash
./batch_convert.sh
```

脚本会自动并行处理目录下所有 `.vox`。

---

## 2. 导出模式说明：`palette` vs `atlas`

### 2.1 `--mode palette`（调色板模式）

特点：

- 纹理为 **1D 调色板**：`256 x 1`（一条线）
- 每个面通过 UV 采样对应颜色
- 纹理非常小，适合纯色/颜色索引工作流

示例：

```bash
python -m magicavoxel_merge \
  input.vox output_palette.glb \
  --mode palette \
  --axis y_up \
  --handedness left \
  --scale 0.02 \
  --center \
  --weld
```

### 2.2 `--mode atlas`（图集模式）

特点：

- 自动生成 atlas 图集纹理
- 可选择 `solid` 或 `baked` atlas 风格

示例：

```bash
python -m magicavoxel_merge \
  input.vox output_atlas.glb \
  --mode atlas \
  --atlas-style baked \
  --atlas-texel-scale 1
```

---

## 3. Atlas 风格：`solid` vs `baked`（重点）

### 3.1 `--atlas-style solid`

- 每个合并后的 quad 区域填充 **单一颜色**
- 合并通常会受到“颜色变化”影响：颜色越碎，越难合并 → 面数更高

```bash
--atlas-style solid
```

### 3.2 `--atlas-style baked`（烘焙型 atlas，推荐）

- 合并几何时 **忽略颜色**：同平面/同朝向会尽量合并成更大的四边面
- 颜色细节 **烘焙进纹理像素**
- 通常能显著降低面数（尤其在颜色变化很碎的模型上）

```bash
--atlas-style baked
```

### 3.3 `--atlas-texel-scale`

控制每个 voxel 对应的纹理像素密度：

- `1`：最省纹理（通常也够用）
- `2/4`：更清晰，但 atlas 更大

```bash
--atlas-texel-scale 1
```

### 3.4 `--atlas-pad` / `--atlas-inset`

用于减少接缝、闪烁、串色：

- `--atlas-pad`：每个 chart 周围额外 padding（像素）
- `--atlas-inset`：UV 向内收，减少 minification 时采到边缘的风险

推荐起步值：

```bash
--atlas-pad 4 --atlas-inset 2
```

### 3.5 `--atlas-layout`

- `global`：全局打包（所有模型的 charts 混合打包）
- `by-model`：先每个模型内部打包，再把模型块打包到总图集（更稳定、也更符合按模型组织）

推荐：

```bash
--atlas-layout by-model
```

### 3.6 `--merge-strategy`

控制网格合并策略：

```bash
--merge-strategy greedy|maxrect
```

- `greedy`：默认策略，速度更快
- `maxrect`：更激进的矩形覆盖策略，通常 quad 分布更均匀（可能更慢）

该选项同时适用于 `palette` 与 `atlas` 两种模式。

---

### 3.7 `--cull-mv-faces`

在 MagicaVoxel 坐标系下剔除多个朝向面：

```bash
--cull-mv-faces top,bottom
```

支持的值：

- `top`（等价于 `+z`）
- `bottom`（等价于 `-z`）
- `+x` / `-x` / `+y` / `-y` / `+z` / `-z`

你也可以按 MagicaVoxel 的 `z` 平面裁剪底面：

```bash
--cull-mv-z 0
```

该裁剪使用 MagicaVoxel 的局部坐标来判断，并且发生在 `--center` 之前，因此无论是否使用 `--center` 都会保持一致。

当你传入更大的阈值（例如 `--cull-mv-z 10`），会剔除更高处、且满足 `z <= 10` 的底面。

在 `atlas` 模式下，会同步减少 atlas 图块（不生成/不打包被剔除面的图块）。

如果你只想按朝向剔除而不关心阈值，使用 `--cull-mv-faces bottom` 即可。

在 `atlas` 模式下，该选项会同时剔除对应面的图块生成与打包，从而让 atlas 贴图自动变小。

在 `atlas-style baked` 下，如果多个 quad 的纹理区块内容完全一致，会自动复用同一个图块并只打包一次，从而进一步减少 atlas 面积。

你也可以通过开关控制该行为：

```bash
--baked-dedup
--no-baked-dedup
```

默认开启（等价于 `--baked-dedup`）。

如果你希望贴图保持 2 的幂但不强制正方形，可加：

```bash
--no-atlas-square
```

在 `atlas-layout by-model` 下，打包会尽量避免出现单方向过度增长的长条 atlas，从而提高填充率（不旋转图块）。

如果你想撤回到默认行为，删除该参数或改回：

```bash
--merge-strategy greedy
```

---

## 4. 坐标轴 / 左右手系（适配引擎）

### 4.1 `--axis`

- `y_up`：导出为 glTF 常用 Y-up（推荐）
- `identity`：不做轴转换（调试用途）

```bash
--axis y_up
```

### 4.2 `--handedness`

- `right`：右手系
- `left`：左手系

```bash
--handedness left
```

如果你发现模型方向/镜像异常，优先检查 `--axis` 与 `--handedness` 组合。

---

## 5. 尺寸与居中

### 5.1 `--scale`

针对 Cocos Creator 常用：

```bash
--scale 0.02
```

### 5.2 `--center` / `--center-bounds`

- `--center`：按体素尺寸中心居中
- `--center-bounds`：按几何 bounds 居中

两者互斥。

---

## 6. 顶点焊接（`--weld`）与法线

### 6.1 `--weld` 做什么

`--weld` 用于合并重复顶点，降低顶点数。

### 6.2 硬边与平滑问题

本工程的焊接逻辑会将合并 key 设为：

- `position + normal + uv`

因此：

- 不会跨硬边（不同 normal）错误合并
- 不会把 voxel 的硬边“焊平滑”

推荐在大多数情况下开启：

```bash
--weld
```

---

## 7. 导出“平均法线”到顶点属性（写入 COLOR_0 / TANGENT）

### 7.1 适用场景

如果你希望：

- 保持当前 `NORMAL`（用于渲染硬边/正确光照）
- 同时额外携带一份“按位置聚合的平均法线”（用于你自己的 shader/特效/调试）

可以启用该选项。

### 7.2 参数

```bash
--avg-normals-attr none|color|tangent
```

- `none`：不导出（默认）
- `color`：写入 `COLOR_0`（推荐，兼容性最好）
- `tangent`：写入 `TANGENT`（不推荐，但可用）

示例（推荐写到 `COLOR_0`）：

```bash
--avg-normals-attr color
```

写入规则：

- `color`：将平均法线 `[-1,1]` 映射到 `[0,1]`（`rgb = n*0.5+0.5`，`a=1`）
- `tangent`：写入 `xyz=n`，`w=1`

---

## 8. 外置贴图输出（可选）

默认贴图嵌入 `.glb` 内。

如果要输出外置 PNG：

```bash
--texture-out /path/to/output.png
```

在 `batch_convert.sh` 里可以用环境变量控制：

```bash
TEXTURE_OUT=1 ./batch_convert.sh
```

脚本会把纹理输出为 `OUT_DIR/<stem>.png`。

---

## 9. 批量转换脚本：`batch_convert.sh`

### 9.1 你需要配置的变量

打开 `batch_convert.sh`，编辑：

- `IN_DIR`：输入目录（包含 `.vox`）
- `OUT_DIR`：输出目录（写入 `.glb` 和可选 `.png`）
- `JOBS`：并行数量

以及导出参数：

- `MODE`：`atlas` / `palette`
- `ATLAS_STYLE`：`baked` / `solid`
- `ATLAS_TEXEL_SCALE`
- `ATLAS_PAD` / `ATLAS_INSET`
- `ATLAS_LAYOUT`
- `AXIS` / `HANDEDNESS`
- `AVG_NORMALS_ATTR`：`none` / `color` / `tangent`

### 9.2 脚本运行时输出

- 若没有 `.vox`：会提示 `No .vox files found...`
- 若有 `.vox`：会提示转换数量与关键模式信息


## 10. 常见问题排查

### 10.1 脚本“没有输出/看起来没导出”

- 检查 `IN_DIR` 下是否真的有 `.vox`
- 脚本会显示 `.vox` 数量；若为 0，会提示 `No .vox files found...`

### 10.2 palette 模式颜色不对

- 当前 palette 贴图为 `256x1`（一条线），UV 也按 1D 采样
- 如果仍不对，优先确认引擎是否正确加载了 glb 内嵌纹理或外置纹理

### 10.3 法线变平滑

- 如果你开启 `--weld`，焊接不会跨法线合并（已包含 normal 在 key）
- 若你自己在引擎端重新计算法线或启用平滑组，也会导致平滑效果

### 10.4 Atlas 接缝/闪烁

- 增大 `--atlas-pad`
- 增大 `--atlas-inset`
- 若纹理太小，可提高 `--atlas-texel-scale`

---

## 11. 参数速查表

本章节按实际 CLI 参数逐项列出：所有可选值 + 默认值 + 作用说明。

### 11.1 位置参数

- `input`
  - 含义：输入 `.vox` 文件路径
- `output`
  - 含义：输出 `.glb` 文件路径

### 11.2 通用参数

- `--mode palette|atlas`（默认：`palette`）
  - 含义：选择贴图/导出模式
  - `palette`：256x1 调色板纹理
  - `atlas`：生成 atlas 图集纹理

- `--merge-strategy greedy|maxrect`（默认：`greedy`）
  - 含义：quad 合并策略
  - `greedy`：更快
  - `maxrect`：更激进，通常 quad 更规整（可能更慢）

- `--scale <float>`（默认：`1.0`）
  - 含义：导出缩放

- `--axis y_up|identity`（默认：`y_up`）
  - 含义：坐标轴转换
  - `y_up`：输出为 glTF 常用 Y-up（推荐）
  - `identity`：不做轴转换（调试用途）

- `--handedness right|left`（默认：`right`）
  - 含义：输出左右手系

- `--center` / `--center-bounds`（默认：都不启用）
  - 含义：将模型平移到原点附近
  - 注意：两者互斥，只能二选一

- `--weld`（默认：关闭）
  - 含义：焊接顶点（减少重复顶点）

- `--avg-normals-attr none|color|tangent`（默认：`none`）
  - 含义：将“平均法线”写入顶点属性
  - `none`：不写入
  - `color`：写入 `COLOR_0`
  - `tangent`：写入 `TANGENT`

- `--flip-v`（默认：关闭）
  - 含义：翻转 UV 的 V 方向（某些引擎/贴图约定需要）

- `--texture-out <path>`（默认：不导出外置纹理）
  - 含义：导出外置 PNG 贴图文件

- `--preserve-transforms` / `--no-preserve-transforms`（默认：开启 `--preserve-transforms`）
  - 含义：是否保留 `.vox` 场景树中的模型平移/层级变换信息

### 11.3 MV 坐标系裁剪

- `--cull-mv-faces <csv>`（默认：不裁剪）
  - 含义：在 MagicaVoxel 坐标系下按“朝向”剔除面
  - 取值：
    - `top`（等价 `+z`）
    - `bottom`（等价 `-z`）
    - `+x` / `-x` / `+y` / `-y` / `+z` / `-z`
  - 示例：`--cull-mv-faces bottom`

- `--cull-mv-z <float>`（默认：不裁剪）
  - 含义：按 MagicaVoxel 局部坐标 `z <= 阈值` 剔除底面（`-z` 方向）
  - 特性：裁剪发生在 `--center` 之前，因此无论是否 `--center` 都保持一致；在 `atlas` 模式下会同步减少 atlas 图块
  - 示例：`--cull-mv-z 0`

### 11.4 atlas 模式参数（仅 `--mode atlas` 生效）

- `--atlas-style solid|baked`（默认：`solid`）
  - 含义：atlas 风格
  - `solid`：每个合并后的 quad 用单一颜色填充
  - `baked`：将颜色烘焙进纹理像素，合并几何时尽量不受颜色影响（更少面数）

- `--atlas-texel-scale <int>`（默认：`1`）
  - 含义：每个 voxel 对应的纹理像素密度（越大越清晰，但 atlas 越大）

- `--atlas-pad <int>`（默认：`2`）
  - 含义：每个图块周围额外 padding（像素）

- `--atlas-inset <float>`（默认：`1.5`）
  - 含义：UV inset（避免采样到边缘外侧造成串色/接缝）

- `--atlas-layout global|by-model`（默认：`global`）
  - 含义：atlas 的打包布局
  - `global`：所有模型的图块全局打包
  - `by-model`：先每个模型内部打包，再把“模型块”打到总 atlas

- `--atlas-square` / `--no-atlas-square`（默认：开启 `--atlas-square`）
  - 含义：是否强制 atlas 更倾向正方形
  - 说明：你需要 pow2 但不需要正方形时，建议 `--no-atlas-square`

- `--baked-dedup` / `--no-baked-dedup`（默认：开启 `--baked-dedup`）
  - 含义：仅在 `--atlas-style baked` 下生效
  - `--baked-dedup`：相同的纹理区块只打包/烘焙一次，其他 quad 复用同一块（减少 atlas 面积）
  - `--no-baked-dedup`：关闭 baked 纹理块去重

---

## 12. 推荐组合（直接复制）

### 12.1 Cocos Creator：最少面数（推荐）

```bash
python -m magicavoxel_merge \
  input.vox output.glb \
  --mode atlas \
  --merge-strategy maxrect \
  --atlas-style baked \
  --atlas-texel-scale 1 \
  --atlas-pad 4 \
  --atlas-inset 2 \
  --atlas-layout by-model \
  --axis y_up \
  --handedness left \
  --scale 0.02 \
  --center \
  --weld
```

### 12.2 需要携带平均法线数据（写入 COLOR_0）

```bash
python -m magicavoxel_merge \
  input.vox output.glb \
  --mode atlas \
  --merge-strategy maxrect \
  --atlas-style baked \
  --avg-normals-attr color \
  --scale 0.02 \
  --axis y_up \
  --handedness left \
  --center \
  --weld
```
