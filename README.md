# btp_vox

`btp_vox` 用来把 MagicaVoxel 的 `.vox` 文件转换成 glTF 2.0 资源。

当前仓库的主能力：

- 输出 `.glb`，或 `.gltf + .bin + .png`
- 生成 atlas 纹理，并可额外导出每个模型的 UV 矩形 JSON
- 支持多模型、场景树和按场景实例导出
- 支持 cutout、UV2、顶点色和多种 atlas 打包选项

## 依赖

```bash
pip install -r requirements.txt
```

也可以在仓库根目录直接运行：

```bash
python -m btp_vox.cli --help
```

## 快速开始

导出单个 `.glb`：

```bash
python -m btp_vox.cli \
  --input input.vox \
  --output output.glb \
  --format glb \
  --scale 0.02 \
  --pivot bottom_center \
  --tex-layout global \
  --tex-pot
```

导出单个 `.gltf`：

```bash
python -m btp_vox.cli \
  --input input.vox \
  --output output.gltf \
  --format gltf \
  --tex-out output.png \
  --uv-out output_uv.json \
  --scale 0.02 \
  --pivot bottom_center
```

输出规则：

- `--format glb`：默认写出单个 `.glb`；如果指定 `--tex-out`，atlas 会写成外部 PNG，不再嵌入 GLB。
- `--format gltf`：写出 `.gltf + .bin`；如果没有指定 `--tex-out`，会在输出目录自动生成同名 `.png`。
- `--uv-out`：额外写出 UV 矩形 JSON，便于调试或后处理。

## 批量转换

仓库内提供了一个项目内使用的批处理脚本 [`batch_convert.sh`](/Users/graylian/workspace/magicavoxel_merge/batch_convert.sh)。

脚本保留了当前仓库作者的默认目录，但支持通过环境变量覆盖：

```bash
DIR_IN=/path/to/vox \
DIR_OUT=/path/to/output \
JOBS=8 \
./batch_convert.sh
```

脚本会：

- 扫描 `DIR_IN` 下所有 `.vox`
- 依据文件名是否以 `-plat.vox` 结尾选择不同导出参数
- 并行输出 `.gltf + .bin + .png + _uv.json`

## 导出行为

默认模式下，导出器会：

- 按场景里的 shape 实例生成 mesh，避免同一个 model 被多次引用时丢实例
- 对节点层级做简化，自动折叠无意义的 wrapper 节点
- 把坐标转换成导出目标使用的 `Y-up` 左手系

如果你需要更接近原始 VOX 层级的结果，可以使用：

- `--no-merge-nodes`：尽量保留原始 scene graph
- `--character-apart`：保留角色部件为独立 mesh
- `--character-flat`：在 `--character-apart` 的基础上，把每个角色根下的部件扁平到一层

## 常用参数

基础参数：

- `--input <path>`：输入 `.vox`
- `--output <path>`：输出 `.glb` 或 `.gltf`
- `--format glb|gltf`
- `--scale <float>`：统一缩放
- `--pivot corner|center|bottom_center`
- `--cull <letters>`：按面裁剪，字母含义为 `t=+Z b=-Z l=-X r=+X f=+Y k=-Y`

UV / 顶点数据：

- `--uv-out <path>`：导出 UV 矩形 JSON
- `--uv-flip-v`：翻转 V
- `--uv2`：导出 `TEXCOORD_1`
- `--uv2-mode copy|lightmap`：UV2 直接复制 UV0，或为 lightmap 生成不重叠布局
- `--vertex-color`：导出白色 `COLOR_0`

层级控制：

- `--no-merge-nodes`
- `--character-apart`
- `--character-flat`

纹理 / atlas：

- `--tex-out <path>`：输出 atlas PNG
- `--tex-fmt auto|rgba|rgb`
- `--tex-style baked|solid`
- `--tex-layout by-model|global`
- `--tex-pad <int>`
- `--tex-inset <float>`
- `--tex-texel-scale <int>`
- `--tex-square`
- `--tex-pot`
- `--tex-fixed-size <width>x<height>`
- `--tex-tight-blocks`
- `--tex-reuse-subrects`
- `--no-tex-reuse-subrects`
- `--tex-compress-solid-quads`
- `--face-alias-uv-remap`
- `--no-face-alias-uv-remap`

cutout：

- `--plat-top-cutout`：把每个 model 导出成单张裁切四边形
- `--plat-cutoff <float>`：cutout 的 alpha cutoff
- `--plat-suffix [suffix]`：名字以该后缀结尾的 model 自动走 cutout；默认后缀是 `-cutout`

调试：

- `--print-nodes`
- `--debug-transforms-out <path>`

## 模型命名约定

面纹理复用可以直接写在模型名里：

- 语法：`模型名@组1@组2...`
- 每组第一个字符是目标面，后续字符表示复用该目标面的面
- 示例：`cube@lrfk@tb`

上面的例子表示：

- `r/f/k` 复用 `l`
- `b` 复用 `t`

cutout 的常见后缀：

- `-cutout`：走默认 cutout 规则
- `-plat-t`：按 top 方向导出单张 cutout quad
- `-plat-f`：按 front 方向导出单张 cutout quad

## 已知情况

- `--vox-view` 这个参数目前仍保留在 CLI 里做兼容，但当前版本不会在默认自动 wrapper 折叠之外再增加额外行为；不建议依赖它来区分导出结果。
- `bottom_center` 只负责把 pivot 放到底部中心并保持模型视觉位置，不会额外把整个场景自动吸附到 `y=0`。
- 名称以 `-plat-t` 结尾的模型会走特殊 cutout 处理，这会影响其几何 pivot 计算方式。

## 排查

打印节点树：

```bash
python -m btp_vox.cli --input input.vox --output output.glb --print-nodes
```

导出变换调试 JSON：

```bash
python -m btp_vox.cli \
  --input input.vox \
  --output output.glb \
  --debug-transforms-out transforms.json
```
