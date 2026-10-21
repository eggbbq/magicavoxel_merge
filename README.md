# btp_vox

`btp_vox` 用来把 MagicaVoxel 的 `.vox` 文件转换成 glTF 2.0 资源。

当前仓库的主能力：

- 输出 `.glb`，或 `.gltf + .bin + .png`
- 生成 atlas 纹理，并可额外导出每个模型的 UV 矩形 JSON
- 支持多模型、场景树和按场景实例导出
- 支持 cutout、UV2、顶点色和多种 atlas 打包选项
- 支持输出 Y-up 右手系或兼容旧资源的 Y-up 左手系
- 提供绘画用 `paint` UV 展开、SVG 参考图和详细布局 JSON，方便二次绘制与烘焙
- 支持对加工后的 glTF/GLB 独立重新 layout，进行区块复用、相似合并、纯色压缩和密集装箱

纹理二次加工使用两个脚本：`test_cub_export.sh` 导出加工版，绘制完成后运行 `test_cub_relayout.sh` 生成发布版。脚本参数逐行排列，可直接编辑；完整流程和配置见下文“用于二次绘制的 UV 展开”及“二次加工后重新打包模型和纹理”。原有 VOX 导出默认仍使用 `atlas` 模式。

## 依赖

```bash
pip install -r requirements.txt
```

也可以在仓库根目录直接运行：

```bash
python -m btp_vox.cli --help
```

查看加工后重排工具的参数：

```bash
python3 -m btp_vox.repack --help
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

批处理脚本当前默认开启 `--tex-reuse-subrects`；如果你想临时关闭，可以显式指定：

```bash
BTP_VOX_TEX_REUSE_SUBRECTS=0 ./batch_convert.sh
```

如果希望 `--tex-reuse-subrects` 真正复用“大块里的子块”，不要把 `BTP_VOX_REUSE_SUBRECT_LIMIT` 设成正数；`0` 表示不因为 quad 总量过大而整批关闭子矩形复用，这也是当前脚本默认值。

当前 atlas 复用不只会匹配“完全同方向”的子矩形，也会尝试左右翻转、上下翻转和转置后的子矩形；命中后会自动在导出的 UV 里补偿方向，所以像地形边条、拐角条带这类纹理通常能明显减少冗余块。

当前 `--tex-inset` 默认值是 `0`。如果你的目标引擎强制改成线性采样、开启 mipmap，或者出现 atlas 边缘串色，再按需把 `--tex-inset` 调大。

脚本会：

- 扫描 `DIR_IN` 下所有 `.vox`
- 依据文件名是否以 `-plat.vox` 结尾选择不同导出参数
- 这是仓库内批处理脚本的兼容约定；如果直接给 `.vox` 里的 model 命名，推荐优先使用下文的 `@` / `#` 规则
- 并行输出 `.gltf + .bin + .png + _uv.json`

## 导出行为

默认模式下，导出器会：

- 按场景里的 shape 实例生成 mesh，避免同一个 model 被多次引用时丢实例
- 对节点层级做简化，自动折叠无意义的 wrapper 节点
- 默认把坐标转换成符合 glTF 规范、可直接用于 LayaAir 的 `Y-up` 右手系
- 如需兼容旧版本导出的方向，可使用 `--handedness left` 输出 `Y-up` 左手系

如果你需要更接近原始 VOX 层级的结果，可以使用：

- `--no-merge-nodes`：保留原始 scene graph，包括 nTRN / nGRP / nSHP 包装节点（单独使用，不叠加角色层级简化选项）
- `--character-apart`：保留角色部件为独立 mesh
- `--character-flat`：在 `--character-apart` 的基础上，把每个角色根下的部件扁平到一层

导出 `hero1.vox` 并保留 MagicaVoxel 中命名的角色分组：

```bash
./export_hero1.sh
```

脚本使用 `--character-apart`，保留 `hero1 → hand-right / hand-left → upper-arm / forearm / palm` 等层级。输出文件写入当前工作目录。

切换 `corner`、`center`、`bottom_center` 只改变 mesh 的局部锚点，部件的世界位置保持一致。导出器按 VOX 网格原点 `floor(SIZE / 2)` 对齐，并在 mesh 节点处补偿一次锚点偏移；空白边距、奇数尺寸、旋转和重复实例都参与计算。

## 常用参数

基础参数：

- `--input <path>`：输入 `.vox`
- `--output <path>`：输出 `.glb` 或 `.gltf`
- `--format glb|gltf`
- `--handedness right|left`：输出坐标系手性；默认 `right`，`left` 用于兼容旧资源
- `--scale <float>`：统一缩放
- `--pivot corner|center|bottom_center`
- `--cull <letters>`：按面裁剪，字母含义为 `t=+Z b=-Z l=-X r=+X f=+Y k=-Y`

UV / 顶点数据：

- `--uv-out <path>`：导出 UV 矩形 JSON
- `--uv-mode atlas|paint`：默认 `atlas` 保留原有打包；`paint` 生成用于二次绘制的独立展开
- `--uv-template-out <path.svg>`：导出带名称和面方向的透明 SVG 绘画参考图，需要 `paint`
- `--uv-islands-out <path.json>`：导出绘画布局和各四边形的像素坐标，需要 `paint`
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
- `--tex-reuse-subrects`：复用重复子矩形；除原样匹配外，也会尝试翻转 / 转置后的子块，并自动修正 UV 方向
- `--no-tex-reuse-subrects`
- `--tex-compress-solid-quads`
- `--face-alias-uv-remap`
- `--no-face-alias-uv-remap`

cutout：

- `--plat-top-cutout`：把每个 model 导出成单张裁切四边形
- `--plat-cutoff <float>`：cutout 的 alpha cutoff
- `--plat-suffix [suffix]`：名字以该后缀结尾的 model 自动走 cutout；默认后缀是 `-cutout`
- 如果是通过模型名控制行为，推荐优先使用后文的 `@t` / `@f` / `@cutout`，`--plat-suffix` 更适合兼容旧资源

调试：

- `--print-nodes`
- `--debug-transforms-out <path>`

## 用于二次绘制的 UV 展开

完整工作流程：

1. 在 MagicaVoxel 中建模，保存 `.vox`。
2. 运行 `test_cub_export.sh`，生成模型和便于加工的展开贴图。
3. 在绘画软件或 Blender 中绘制、烘焙颜色贴图，保存加工成果。
4. 运行 `test_cub_relayout.sh`，以加工后的模型和贴图生成新的紧凑资源。

`test_cube.vox` 对应的两个脚本：

| 阶段 | 脚本 | 输入 → 输出 |
|---|---|---|
| 导出加工版 | `./test_cub_export.sh` | `test_cube.vox` → `exports/test_cube_paint/` 中的模型、展开贴图和参考图 |
| 绘制后重新 layout | `./test_cub_relayout.sh` | 加工版模型和画好的贴图 → `exports/test_cube_runtime/` 中的新模型和紧凑贴图 |

两个脚本之间，在绘画软件中修改 `exports/test_cube_paint/test_cube_paint.png` 并保存。重排脚本显式读取这张贴图，原加工版文件保持不变。如果在 Blender 中修改了网格或 UV，重排时应同时提供重新导出的模型和对应贴图；贴图必须与输入模型的 UV 匹配。

加工版用于持续编辑，发布版用于运行时加载。发布版可能让多个面共用纹理、让纯色面的 UV 收缩到一个采样点；后续修改请回到加工版，再次重排生成发布版。

`--uv-mode paint` 是可选模式。不传此参数时，原有 atlas、复用、压缩、命名规则和批处理脚本的行为保持不变。

直接使用仓库中的 `test_cube.vox`：

```bash
./test_cub_export.sh
```

脚本可以从任意目录执行，默认输出到仓库的 `exports/test_cube_paint/`：

- `test_cube_paint.gltf`、`test_cube_paint.bin`：模型，引用同目录 PNG
- `test_cube_paint.png`：可直接修改的颜色贴图
- `test_cube_paint_uv.svg`：与 PNG 同尺寸的透明参考图，作为绘画软件的参考图层，不要合并进最终贴图
- `test_cube_paint_islands.json`：部件、展开片、面方向、深度及四边形像素坐标
- `test_cube_paint_uv.json`：保持原格式的每个模型 UV 包围矩形

两个脚本均使用逐行排列的 `args=(...)` 参数列表，直接打开 `.sh` 修改。输入、输出路径在脚本顶部的 `in_vox` / `in_model` / `in_tex` / `out_dir` 等变量中修改。

两个脚本当前采用的默认配置：

| 配置 | `test_cub_export.sh` | `test_cub_relayout.sh` |
|---|---|---|
| 输入 | `test_cube.vox` | 加工版 `.gltf` 和 PNG |
| 输出目录 | `exports/test_cube_paint/` | `exports/test_cube_runtime/` |
| 像素密度 | 每体素面单位 `8×8` 像素 | 保留输入像素，纯色块可折叠 |
| 边距 | `--tex-pad 0` | `--tex-pad 0` |
| POT | 开启 `--tex-pot` | 使用 `--no-tex-pot` |
| 区块复用 | 关闭，保持可独立绘制 | 开启相似合并，阈值 `0.98`、最大通道色差 `16` |
| 纯色压缩 | 关闭 | 开启 |
| 装箱 | 按模型集中排布 | `dense` 密集装箱 |

这是示例脚本的配置，不等于 CLI 默认值。例如两个 CLI 的边距默认均为 `2`；重排 CLI 默认不合并区块、不压缩纯色，使用 `shelf` 装箱。

导出脚本默认每体素面单位为 `8×8` 像素、外边距为 `0`，并开启 POT。加工版不默认增加外边距，避免十字展开宽度从 `512` 增加到 `520` 后，被 POT 扩展到 `1024`。例如在参数列表中编辑：

```bash
  --tex-texel-scale    8     # 改为 1 使用原始像素密度
  --tex-pad            0     # 修改展开片外边距
  --tex-pot                  # 需要非 POT 时，注释这一整行
```

其他资源可修改脚本顶部的 `in_vox` 路径；输出文件名可在 `out_model` / `out_tex` 处修改。通用 CLI 示例：

```bash
python3 -m btp_vox.cli \
  --input input.vox \
  --output paint.gltf \
  --format gltf \
  --uv-mode paint \
  --character-apart \
  --tex-texel-scale 8 \
  --tex-pad 0 \
  --tex-pot \
  --tex-out paint.png \
  --uv-template-out paint_uv.svg \
  --uv-islands-out paint_islands.json
```

绘画模式的布局规则：

- 每个 model 单独展开并集中排布，外包围面的相邻表面保持六面十字布局。方盒的排列为 `l / f / r / k`，`t` 在 `f` 上方，`b` 在 `f` 下方；字母使用 MagicaVoxel 模型空间，含义与 `--cull` 一致。
- 同方向但处于不同深度的凹槽、台阶或内部表面另建平面展开片，防止投影重叠。同一平面的四边形按空间位置连接，缺口留空。
- 不复用相同、镜像或子矩形纹理，不压缩纯色面；模型名中的面复用规则（例如 `@lrfk@tb`）也会被忽略。不同 shape 对同一 model 的引用会获得独立绘画区域，方便单独绘制左右肢体。
- 原始体素颜色直接写入新布局，网格形状、顶点、法线、坐标系、锚点和节点变换保持原流程；cutout 和剔面仍然有效。
- `--tex-texel-scale 8` 表示每个体素面单位占 `8×8` 像素；CLI 默认仍为 `1`，专用脚本默认 `8`。`--tex-pad` 控制展开片外的颜色扩边；相连的面之间没有间隔。
- 支持 `--tex-pot`、`--tex-square`、`--tex-fixed-size`、纹理格式和 UV2。固定尺寸不够时会报错，不会自动缩小绘制分辨率。
- 本模式固定使用原色烘焙与按模型分组；`--tex-style`、`--tex-layout`、`--tex-tight-blocks`、纹理复用、纯色压缩、face alias 和 `--tex-inset` 参数不参与绘画布局。旧 `atlas` 模式继续使用这些参数。

SVG 与详细 JSON 均使用 PNG 的左上角像素坐标，四边形坐标按原始顶点顺序排列。`--uv-flip-v` 仍仅翻转导出的网格 UV，元数据中会记录该设置，参考图继续对应实际 PNG。固定模型和配置的输出是确定的；改变模型尺寸、可见表面或实例数量后布局可能变化，应重新导出参考图。

POT 仅约束画布宽高，不保证 UV 填满画布。16×16×16 方盒的六个独立表面共需 `6×16×16=1536` 个像素，无法恰好填满面积也是 2 的幂的画布。当前连续十字展开为 4×3 个面：每体素 1 像素、无边距时为 `64×48`，POT 后为 `64×64`；每体素 8 像素则为 `512×384`，POT 后为 `512×512`。保留连续展开本身也会留下空白，第 4 步再紧凑打包。

绘制时保留 PNG 的尺寸和文件名，保存后重新载入模型贴图即可；另存为其他文件时修改重排脚本的 `in_tex`。重新运行第 2 步会覆盖同名加工版输出，绘制成果请另存备份。

当前 `paint` 使用外包围盒六面展开和不同深度的分平面展开。复杂模型的凹槽、窗洞、台阶可能被分成较多布局组，同组也可能包含不相连的面；它还不是按拓扑邻接关系选择接缝、尽量连接相邻表面的通用展开算法。“沿接缝生成连续 UV 岛”目前尚未实现，需要这类布局时可在 Blender 中编辑，并使用新的模型/贴图执行第 4 步。

## 二次加工后重新打包模型和纹理

完整工作流程为：MagicaVoxel 建模 → `--uv-mode paint` 导出加工版 → 绘制/烘焙颜色贴图 → 独立重排工具生成发布版。

第 4 步只读取**加工后的 `.gltf` / `.glb` 和实际颜色贴图**，不读取 `.vox`，也不需要之前的 UV JSON。CLI 未传 `--texture` 时使用模型引用的贴图；示例脚本显式指定 `in_tex`，可修改为另外保存的绘制结果，也可以注释 `--texture` 整行改为自动读取。

木箱专用脚本：

```bash
# 非 POT，默认；输出到 exports/test_cube_runtime/
./test_cub_relayout.sh

# POT：在脚本中将 --no-tex-pot 改为 --tex-pot 后，再执行同一脚本
# 另存的加工模型和贴图：修改脚本顶部的 in_model 和 in_tex 路径
```

与示例脚本优化设置对应的通用命令：

```bash
python3 -m btp_vox.repack \
  --input edited.gltf \
  --texture finished.png \
  --output runtime/model.gltf \
  --tex-pad 0 \
  --no-tex-pot \
  --merge-similar \
  --similarity-threshold 0.98 \
  --similarity-max-error 16 \
  --compress-solid-patches \
  --pack dense \
  --report-out runtime/repack_report.json
```

输出为新的 `.gltf + .bin + .png`；输出扩展名为 `.glb` 时，纹理会嵌入新 GLB。`--tex-out` 可以指定外部 PNG 路径，对 GLB 则是额外输出一份贴图副本。

工具依据三角形位置和 UV，沿几何折角与 UV 接缝拆分纹理区域，打散加工版十字展开的空白；直接复制像素并更新 UV，保留非纯色区块的原像素密度。需要在折角处分离共用顶点时，也会复制对应的权重、其他顶点属性和形变数据。节点变换、骨骼和动画继续沿用。加工版文件不会被覆盖；源文件与输出路径冲突会报错。再次运行会覆盖同名发布版输出。

- `--tex-pot` / `--no-tex-pot`：选择 POT / 非 POT，默认非 POT。
- `--tex-pad`：纹理块四周的像素边距，默认 2；复制原贴图中对应的邻接像素，保留原边缘的线性采样颜色。
- `--deduplicate`：可选，复用最终像素和边距都完全相同的纹理块；脚本中取消这一行的注释即可开启。
- `--merge-similar`：合并相似区块，也尝试旋转和翻转，自动修正 UV。只比较尺寸相同或旋转后尺寸相同的区块，不缩放图案。
- `--similarity-threshold 0.98`：最低相似度，定义为 `1 - RGB 平均绝对误差 / 255`。越高越严格，设为 `1` 时只接受完全匹配；这不是感知相似度或 SSIM。
- `--similarity-max-error 16`：每个 RGB 通道允许的最大差值，用于保护平均误差可能掩盖的小图案。alpha 必须逐像素完全相同，边距也参与比较。每块直接与保留的原始代表块比较，避免链式合并累积误差。
- `--compress-solid-patches`：把颜色和 alpha（包括边距）都一致的纯色区块压成单像素，UV 改为采样该像素中心；保留几何、权重和动画。有渐变、烘焙阴影或细节的区块不会这样压缩。
- `--pack dense|shelf`：`dense` 使用 MaxRects 尝试多种宽度与排序，保留快速装箱作为面积上限；`shelf` 使用原有快速装箱。

在 `test_cub_relayout.sh` 的参数列表中直接编辑，例如：

```bash
  --tex-pad                 0
  --no-tex-pot                     # 改为 --tex-pot 开启 POT
  --merge-similar                  # 注释整行关闭相似合并
  --similarity-threshold    0.98
  --similarity-max-error    16
  --compress-solid-patches         # 注释整行关闭纯色压缩
  --pack                    dense
```

`test_cub_relayout.sh` 默认启用相似合并、纯色压缩和 `dense` 装箱；Python API / CLI 的这些优化默认关闭，原有调用行为不变。`--merge-similar` 本身已经包含完全匹配复用，不必同时加 `--deduplicate`。两者都关闭时，每个区块独立排布。

如果希望保留旋转/翻转复用、纯色压缩与密集装箱，同时禁止近似颜色替换，在脚本中改成：

```bash
  --merge-similar
  --similarity-threshold    1
  --similarity-max-error    0
  --compress-solid-patches
  --pack                    dense
```

相似合并会选用一个现有区块作为代表，可能改变细微颜色；它不修改代表块，也不通过均值合成一张新图。完全匹配的旋转/翻转复用与纯色压缩可以保持原始采样颜色；近似替换产生的细节差异不能靠修改 UV 恢复，应提高阈值或回到加工版重新导出。

### 纯色单像素采样与边缘串色

纯色压缩后，整个面的所有 UV 顶点都固定到代表像素的中心：

```text
u = (像素 x + 0.5) / 贴图宽度
v = (像素 y + 0.5) / 贴图高度
```

因此面内部插值后仍指向同一像素。若两个颜色像素横向组成一张 `2×1` 贴图，它们的采样中心分别是 `u=0.25` 和 `u=0.75`，不会让纯色面的 UV 横跨两像素之间的 `u=0.5` 分界。

VOX 导出器写入的是最近邻采样（glTF 的 `magFilter/minFilter=9728`），不请求 mipmap 采样。重排工具保留输入模型的采样设置，不会强制把 Blender 或其他软件另存的材质改为最近邻。当前示例脚本的“零边距＋像素中心采样”应与最近邻设置一起使用。

若目标引擎改用线性过滤、mipmap 或其他过滤方式，应同时验证扩边和远距离显示效果。`--tex-pad 2` 表示核心四周各增加两像素；符合纯色压缩条件的 `1×1` 核心连同边距会占 `5×5` 像素。边距来自原贴图的邻接像素；若边距含不同颜色或 alpha，该区块不会被当作纯色折叠。固定两像素边距不能保证任意 mipmap 级别都安全，也不能仅靠 UV 中心对齐保证所有引擎设置下无串色。

### 重排报告与结果检查

`--report-out` 输出 JSON，常用字段如下；相似合并、纯色压缩的专属字段只在对应选项开启时出现。

| 字段 | 含义 |
|---|---|
| `source_size` / `packed_size` | 源贴图和输出贴图的宽、高 |
| `area_reduction_percent` | 相对源贴图的画布面积变化，正数表示减少，负数表示增大 |
| `patches` / `unique_patches` | 拆分后的区块总数 / 实际保留并装箱的区块数 |
| `packing` / `rect_fill_percent` | 装箱方式 / 包含边距的区块矩形占用率 |
| `exact_merged_patches` / `similar_merged_patches` | 完全匹配复用 / 有颜色差异的近似复用数量 |
| `transformed_merged_patches` / `similarity_matches` | 使用旋转或翻转的匹配数量 / 各匹配的代表块、变换、相似度和误差 |
| `lossy` / `max_rgb_error` / `max_mean_rgb_error` | 是否发生近似颜色替换 / 匹配块的最大通道色差 / 最大平均色差，比较包含边距 |
| `solid_compressed_patches` | 被折叠的原始纯色区块数，可能随后共用同一个代表像素 |
| `solid_texels_before` / `solid_texels_after` | 纯色区块折叠前后的核心像素总数，尚未扣除区块复用 |
| `solid_texels_saved_before_deduplication` | 纯色折叠节省的核心像素数，不等于最终画布减少的面积 |
| `triangles` / `duplicated_vertices` | 参与重排的纹理三角形数量 / 因 UV 接缝新增的顶点数量 |
| `resampled` | 是否进行了图像插值缩放；当前为 `false`，不表示必然没有近似颜色替换 |

工具自动输出这些结构和区块比较指标，不会自动在 Blender 或游戏引擎中渲染验证。`max_rgb_error=0` 表示本次匹配没有颜色差异；它不等同于所有过滤模式、缩放距离和引擎配置下的渲染误差都为 0。

一次包含 158 个区块的复杂资源实测中，整块合并版本保留 49 个区块，贴图为 `1216×1632`；增加纯色压缩与密集装箱后保留 31 个区块，贴图为 `1440×704`，相对上一版面积再减少 `48.92%`。该次矩形装箱占用率为 `96.21%`，未发生近似颜色替换，31,600 个表面颜色采样的最大 RGBA 误差为 0。该结果属于特定输入的验证示例，不是每个模型都能达到的固定比例。

### 支持范围与兼容性

当前尚未复用大块内部的子矩形，也不将非矩形 UV 岛的空洞用于交错装箱；高矩形占用率不代表所有像素都被实际表面使用，也不代表已达到理论最小面积。

目前支持三角形网格、0–1 范围 UV、共用一张颜色贴图的材质，包括颜色贴图中烘焙好的光照/AO。材质可指定 `TEXCOORD_0` 或其他已有 UV 集作为颜色贴图坐标；重排更新这个 UV 集，其他顶点属性、其他 UV、蒙皮和形变数据按对应顶点保留。

法线、粗糙度等多贴图联动重排、范围外的重复 UV、纹理变换和压缩模型需先处理；不支持的输入会明确报错。POT 或边距设置会影响最终面积，输入已经紧凑时不保证继续缩小。

VOX 导出的 `atlas` 模式已有子矩形复用能力；加工后使用的 `btp_vox.repack` 是另一条流程，目前仅提供上述整块合并和纯色压缩，不会因为 VOX 导出器支持某个选项就自动支持它。原有导出参数、模型命名规则和批处理脚本保持原有用途。

## 模型命名约定

推荐规则：

- `@`：用于 plat / cutout 模式
- `#`：用于局部剔面
- 旧写法 `-plat-*`、`-cutout`、`-cull-*` 仍可用，但建议新资源不要再继续使用

面纹理复用可以直接写在模型名里：

- 语法：`模型名@组1@组2...`
- 每组第一个字符是目标面，后续字符表示复用该目标面的面
- 示例：`cube@lrfk@tb`
- 注意：`@t`、`@f`、`@cutout` 现在保留给 plat / cutout 命名，不参与 face alias

上面的例子表示：

- `r/f/k` 复用 `l`
- `b` 复用 `t`

plat / cutout 推荐命名：

- `模型名@cutout`：走默认 cutout 规则
- `模型名@t`：按 top 方向导出单张 cutout quad
- `模型名@f`：按 front 方向导出单张 cutout quad
- 推荐新资源统一使用这套 `@...` 写法，不再推荐继续扩展 `-plat-*` / `-cutout`

按模型名局部剔面推荐写成 `#` 后缀：

- 语法：`模型名#<letters>`
- 示例：`wall#lr`、`roof@t#b`
- 字母含义与 `--cull` 相同：`t=+Z b=-Z l=-X r=+X f=+Y k=-Y`
- 模型名剔面会和全局 `--cull` 叠加，只影响当前 model
- 推荐新资源统一使用这套 `#...` 写法，不再推荐继续扩展 `-cull-<letters>`

可以组合使用：

- `cube@t@lrfk#b`：按 top cutout 导出，`r/f/k` 复用 `l`，并额外剔除 `b`
- `tree_1#tblr`：只保留基础模型名 `tree_1`，并剔除 `t/b/l/r`
- `rock_big@f#k`：按 front cutout 导出，并剔除 `k`

兼容性：

- 旧写法 `-plat-t`、`-plat-f`、`-cutout`、`-cull-<letters>` 仍然可用
- 导出后的 mesh / node / UV 名称会自动去掉这些规则标记，只保留基础名；例如 `tree_1#tblr` 会导出成 `tree_1`
- 也就是说，规则字符只参与导出行为控制，不应该成为最终资源名的一部分

## 已知情况

- `--vox-view` 这个参数目前仍保留在 CLI 里做兼容，但当前版本不会在默认自动 wrapper 折叠之外再增加额外行为；不建议依赖它来区分导出结果。
- `bottom_center` 只负责把 pivot 放到底部中心并保持模型视觉位置，不会额外把整个场景自动吸附到 `y=0`。
- 名称为 `@t`（或兼容旧写法 `-plat-t`）的模型会走特殊 cutout 处理，这会影响其几何 pivot 计算方式。

## 排查

Blender 中看不到贴图：

- 通过“文件 → 导入 → glTF 2.0”导入 `.gltf` / `.glb`，确认场景中选中的是导入的模型。
- 切换到视图右上角的“材质预览”；默认实体着色不会显示图像纹理。
- 默认 `.gltf` 输出的 `.gltf`、`.bin`、`.png` 保存在同一目录；移动模型时保持引用的相对路径有效。重排 CLI 的 `--tex-out` 指向其他目录时，也应保留该相对路径关系。
- 修改磁盘 PNG 后，在 Blender 中重新加载对应图像；检查图像纹理节点是否连接到材质的 Base Color。
- 检查像素效果时确认图像插值为 Closest；引擎中的过滤、mipmap 和边距也需与资源配置一致。

运行现有检查：

```bash
python3 -m unittest discover -s tests -v
bash -n test_cub_export.sh test_cub_relayout.sh
```

相关测试位于 `tests/test_paint_uv.py`、`tests/test_repack.py` 和 `tests/test_scene_transforms.py`，覆盖展开颜色、接缝、旋转/翻转 UV、相似度约束、alpha、纯色中心采样、密集装箱、输入文件保护，以及几何、骨骼、动画和形变数据的保留。这些检查不代替目标游戏引擎验证。

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
