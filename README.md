# btp_vox 使用文档（MagicaVoxel .vox → glTF2 / GLB）

`btp_vox` 是一个将 MagicaVoxel 的 `.vox` 文件转换为 glTF 2.0 的工具。

名称含义：

- `btp`：Block-Topology Preservation（块拓扑保留的体素面片化思路）
- `vox`：输入来源是 MagicaVoxel 的 `.vox`

目标（面向游戏工作流）：

- 输出 **glTF2**（`.glb` 或 `.gltf + .bin + .png`）
- 自动生成 **atlas 纹理**（PNG），并为每个模型写出 **UV 信息 JSON**（可选）
- 支持 **多模型 / 场景树（scene graph）** 的 `.vox`
- 导出按场景树实例输出 mesh（避免丢失 VOX 的 instancing 引用）

---

## 1. 安装 / 运行方式

在仓库根目录运行（示例）：

```bash
python -m btp_vox.cli --input input.vox --output output.glb
```

说明：

- `btp_vox` 是本仓库中的一个 Python 包模块
- 不需要额外的安装步骤时，直接用 `python -m ...` 调用即可

---

## 2. 快速开始

### 2.1 单文件导出（GLB）

```bash
python -m btp_vox.cli \
  --input input.vox \
  --output output.glb \
  --format glb \
  --tex-out output.png \
  --vertex-color \
  --scale 0.02 \
  --pivot center \
  --tex-layout global \
  --tex-pot \
  --tex-fmt auto
```

#### 2.1.1 角色模型导出（推荐使用 bottom_center）

```bash
python -m btp_vox.cli \
  --input character.vox \
  --output character.glb \
  --format glb \
  --tex-out character.png \
  --vertex-color \
  --scale 0.02 \
  --pivot bottom_center \
  --tex-layout global \
  --tex-pot \
  --tex-fmt auto
```

### 2.2 单文件导出（glTF：外置 .bin + .png）

```bash
python -m btp_vox.cli \
  --input input.vox \
  --output output.gltf \
  --format gltf \
  --tex-out output.png \
  --vertex-color \
  --scale 0.02 \
  --pivot center \
  --tex-layout global \
  --tex-pot \
  --tex-fmt auto
```

说明：

- `--format gltf` 会写出：
  - `output.gltf`
  - `output.bin`
  - `output.png`（atlas 纹理）

### 2.3 批量转换（推荐）

使用仓库内脚本：`batch_convert.sh`

1. 修改脚本顶部：

- `DIR_IN`：输入 `.vox` 所在目录
- `DIR_OUT`：输出目录
- `JOBS`：并发数

2. 运行：

```bash
./batch_convert.sh
```

提示：如果你的引擎/Shader 需要 `COLOR_0` 顶点色通道（用于传递自定义数据等），请在脚本的参数列表中加入 `--vertex-color`。

---

## 3. 输出内容说明

### 3.1 输出文件

根据 `--format`：

- `--format glb`
  - 输出：`*.glb`
- `--format gltf`
  - 输出：`*.gltf + *.bin + *.png`

如果指定 `--uv-out`：

- 会额外输出一个 JSON，记录每个模型在 atlas 中的 UV 区域（便于外部工具调试）

### 3.2 导出层级结构（节点树）

当前导出遵循以下结构（满足引擎侧更直观的层级需求）：

- 顶层：多个“父节点”（对应 `.vox` 场景中的节点）
- 每个父节点下：一个或多个“mesh 子节点”（挂 mesh）

伪结构示例：

```text
<Scene Roots>
|__ ParentNodeA
|    |__ MeshChild1
|__ ParentNodeB
     |__ MeshChild2
```

#### 父节点命名规则

父节点优先使用你在 MagicaVoxel 里设置的节点名称：

- 若 `nSHP` 节点本身有自定义名：使用该名字
- 若 `nSHP` 节点名是默认值（如 `shp_3`），则向上查找最近的 `nTRN` 祖先：
  - 若该 `nTRN` 有自定义名：使用该名字
- 如果仍然没有任何自定义名：才会显示为 `shp_<id>` 风格

#### Mesh 子节点命名规则

Mesh 子节点名称使用 `.vox` 的 `model` 名称（即 `scene.models[model_id].name`）。

---

## 4. 坐标、缩放、Pivot

### 4.1 `--scale`

`--scale` 是**统一缩放系数**：

- mesh 顶点会乘以 `scale`
- 节点 `translation` 也会乘以 `scale`

因此你可以把 `0.02` 作为常见的“体素到米（或引擎单位）”的缩放。

### 4.2 `--pivot`

可选：

- `corner`：原点在模型角落（默认）
- `center`：原点在模型几何中心
- `bottom_center`：原点在模型底部中心（适合角色模型）

说明：

#### `corner`
- 原点位于模型的最小坐标点（左下角）
- 适用于需要精确控制模型边界的情况

#### `center` 
- 原点位于模型的几何中心（X、Y、Z 轴都居中）
- 最常用的 pivot 模式，适合大多数场景

#### `bottom_center`
- 原点位于模型底部中心（X、Z 轴居中，Y 轴在最低点）
- **特别适合角色模型**，便于地面放置和动画控制
- 实现原理：
  1. 顶点向上移动 `half_height`（使 pivot 位于底部）
  2. 节点向下移动 `half_height`（补偿视觉位置）
- **保证**：模型视觉位置与 `center` 模式完全一致，只有 pivot 位置不同

#### 技术细节
- 该选项通过调整 mesh 顶点位置实现 pivot 变更
- 同时调整节点 translation 以保持视觉位置不变
- 支持复杂的节点层次结构（包括没有几何体的父节点）
- 坐标系转换：从 MagicaVoxel Z-up 自动转换为 Unity Y-up
- **自动地面平移**：使用 `bottom_center` 时会自动将所有模型向下平移，使最低的模型贴着地面（y=0）

#### 地面平移功能
当使用 `--pivot bottom_center` 时，系统会：
1. 找到所有模型中底面最接近 y=0 的模型
2. 计算需要向下移动的距离（使其底面贴着 y=0）
3. 所有模型都应用相同的平移，保持相对位置不变
4. 输出日志显示平移距离，便于调试

这样确保所有角色模型都正确地站在地面上，便于场景布置。

#### Plat-t 模型特殊处理
**重要**：无论用户选择什么 pivot 模式，**模型名称以 `-plat-t` 结尾的模型都会强制使用 `top_center` pivot**

**原因**：
- `plat-t` 模型只包含顶面四边形
- 使用 `top_center` pivot 更符合顶面的逻辑
- 确保地面平移功能正确计算最低点

**检测规则**：
- 仅检测模型名称是否以 `-plat-t` 结尾（不区分大小写）
- 其他模型（如 `-plat-f`）保持用户选择的 pivot 不变

---

## 5. Atlas / 纹理相关参数

### 5.1 `--tex-layout`

- `global`
  - 将所有模型的图块全局一起打包
- `by-model`
  - 先模型内部打包，再把每个模型块打到总 atlas

### 5.2 `--tex-pot`

强制 atlas 宽高为 2 的幂（power-of-two），利于部分引擎纹理压缩与 mipmap。

### 5.4 `--tex-out` / `--tex-fmt`

- `--tex-out <path>`：将 atlas 纹理写出为外部 PNG 文件（不再内嵌进 `.glb`）。
- `--tex-fmt auto|rgba|rgb`：控制输出纹理是否包含透明通道。
  - `auto`（默认）：若整张 atlas 没有透明像素则输出 RGB，否则输出 RGBA。
  - `rgba`：强制输出 RGBA。
  - `rgb`：强制输出 RGB（会丢弃透明）。

### 5.3 `--tex-pad` / `--tex-inset`

- `--tex-pad <int>`：图块间 padding（texel）
- `--tex-inset <float>`：UV 内缩，降低串色风险

如果你看到贴图边缘有“串色/闪烁”，建议增大这两个值。

---

## 6. 调试与排查

### 6.1 打印 VOX 场景树

```bash
python -m btp_vox.cli --input input.vox --output output.glb --print-nodes
```

会把 scene graph 节点信息打印到 stderr，便于排查：

- 节点命名是否读取正确
- 是否存在多模型引用、缺失模型等

### 6.2 导出变换调试 JSON

```bash
python -m btp_vox.cli \
  --input input.vox --output output.glb \
  --debug-transforms-out transforms.json
```

用于在导出前/后坐标系转换阶段输出变换数据，排查坐标问题。

---

## 7. CLI 参数速查

必填参数：

- `--input <path>`：输入 `.vox`
- `--output <path>`：输出 `.glb` 或 `.gltf`

常用参数：

- `--format glb|gltf`：输出格式
- `--scale <float>`：统一缩放
- `--pivot corner|bottom_center|center`：pivot 模式
- `--uv-flip-v`：翻转 UV 的 V
- `--uv2`：导出第二套 UV（`TEXCOORD_1`，由 `TEXCOORD_0` 复制，用于 lightmap 烘焙）
- `--uv2-mode copy|lightmap`：UV2 生成方式（默认 copy；lightmap 会生成不重叠的 UV2，用于真正的 lightmap 烘焙）
- `--vertex-color`：导出顶点色通道（`COLOR_0`，默认填充白色，可用于 shader 传递数据）
- `--no-merge-nodes`：不做“两层节点扁平化”，按 VOX 原始 scene graph（`nTRN/nGRP/nSHP`）导出节点
- `--character-apart`：角色导出模式：在原始 VOX 层级下保留各部件（各 model）为独立 mesh（等价于启用 `--no-merge-nodes`）。同时会尽量折叠/隐藏 VOX scene graph 中的自动包装节点（如 `trn_*/grp_*/shp_*`），让导出层级更接近 MagicaVoxel 视图结构
- `--character-flat`：配合 `--character-apart` 使用：以每个角色根节点为单位，把其下所有部件扁平到一层（全部成为该角色节点的直接子节点）
- `--vox-view`：折叠/移除 VOX 自动包装节点（如 `trn_*/grp_*/node_*`），并把变换下推到子节点，使导出层级尽量与 MagicaVoxel 视图一致（`--character-apart/--character-flat` 默认启用该折叠行为）
- `--uv-out <path>`：输出 UV JSON
- `--tex-out <path>`：输出 atlas PNG 文件
- `--tex-fmt auto|rgba|rgb`：纹理透明通道控制
- `--tex-layout by-model|global`
- `--tex-reuse-subrects`：复用重复子块纹理（小块可映射到大块子区域），减小 atlas 面积（默认开启）
- `--no-tex-reuse-subrects`：关闭子块复用，回退到每个 quad 独立打包
- `--tex-compress-solid-quads`：若一个平面的颜色块是单色，则压缩为 `1x1` 像素再参与 atlas 打包
- `--tex-pot`
- `--tex-pad <int>`
- `--tex-inset <float>`
- `--tex-texel-scale <int>`
- `--plat-top-cutout`：贴片裁切模式（每个 model 输出 1 个顶面四边形 + alpha clip）
- `--plat-cutoff <float>`：贴片裁切的 alpha cutoff（默认 0.5）
- `--plat-suffix [str]`：名称以该后缀结尾的 model 自动用贴片裁切方式导出（默认 `-cutout`，也可直接写 `--plat-suffix` 不带参数）

`--plat-suffix` 支持简写：

- `--plat-suffix plat-t`：等价于后缀 `-plat-t`（top +Z cutout）
- `--plat-suffix plat-f`：等价于后缀 `-plat-f`（front +Y cutout）

特殊后缀：

- `-plat-t`：按 top(+Z) 贴片裁切（单 quad + alpha clip）
- `-plat-f`：按 front(+Y) 贴片裁切（单 quad + alpha clip）

示例：

```bash
# 自动识别后缀：-cutout / -plat-t / -plat-f
python -m btp_vox.cli --input input.vox --output output.glb

# 全局强制所有 model 用贴片裁切（top +Z）
python -m btp_vox.cli --input input.vox --output output.glb --plat-top-cutout

# 把默认触发后缀从 -cutout 改成 -mycut
python -m btp_vox.cli --input input.vox --output output.glb --plat-suffix -mycut
```

不常用/调试参数：

- `--print-nodes`
- `--debug-transforms-out <path>`

---

## 8. 常见问题

### 8.1 父节点为什么显示成 `shp_XX`？

说明该节点在 VOX 里没有有效的自定义名（或者名字写在 `nTRN` 上但没有对应关系）。

建议：

- 在 MagicaVoxel 的 Scene Graph 中给对象/组设置名称
- 重新导出并使用 `--print-nodes` 检查节点名

### 8.2 调整 `--scale` 后位置不对？

当前导出规则是：

- 顶点和节点 translation 都会乘 `scale`

因此在 `--scale` 改变时，整体应当等比例变化，不应出现“越远误差越大”。

如果出现异常：

- 用 `--debug-transforms-out` 导出调试数据
- 检查引擎导入时是否又做了额外缩放

### 8.3 `bottom_center` 模式下模型浮空或位置错误？

**问题现象**：使用 `--pivot bottom_center` 后，模型看起来浮空或位置不正确。

**解决方案**：
- 确保使用的是最新版本的 btp_vox（已实现完整的 bottom_center 支持）
- 检查模型是否正确导出：pivot 应该在模型底部中心
- 如果仍有问题，尝试：
  1. 对比 `--pivot center` 的输出位置是否正确
  2. 检查 Unity/引擎导入设置是否正确

**技术说明**：`bottom_center` 模式通过双重调整确保正确性：
1. 顶点向上移动 `half_height`（pivot 在底部）
2. 节点向下移动 `half_height`（视觉位置不变）

### 8.4 不同 pivot 模式如何选择？

**`corner`**：
- 适用于需要精确控制模型边界的场景
- 如建筑块、道具等需要精确对齐的物体

**`center`**：
- 最通用的选择，适合大多数场景
- 如环境物体、静态道具等

**`bottom_center`**：
- **强烈推荐角色模型使用**
- 便于地面放置、动画控制、物理模拟
- 如 NPC、玩家角色、可移动的物体等
