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
  --scale 0.02 \
  --pivot center \
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

- `corner`
- `bottom_center`
- `center`

说明：

- 该选项会改变 mesh 的局部原点（通过对顶点做平移实现）
- 导出的节点层级仍保持正确的世界坐标关系（以当前实现为准）

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
- `--uv-out <path>`：输出 UV JSON
- `--tex-out <path>`：输出 atlas PNG 文件
- `--tex-fmt auto|rgba|rgb`：纹理透明通道控制
- `--tex-layout by-model|global`
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


