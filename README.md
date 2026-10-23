# btp_vox

`btp_vox` 用来把 MagicaVoxel 的 `.vox` 文件转换成 glTF 2.0 资源。

当前仓库的主能力：

- 输出 `.glb`，或 `.gltf + .bin + .png`
- 生成 atlas 纹理，并可额外导出每个模型的 UV 矩形 JSON
- 支持多模型、场景树和按场景实例导出
- 支持 cutout、UV2、顶点色和多种 atlas 打包选项
- 支持输出 Y-up 右手系或兼容旧资源的 Y-up 左手系
- 提供绘画用 `paint` UV 展开，可沿真实表面公共边连接 UV 岛，导出 SVG 参考图和详细布局 JSON，方便二次绘制与烘焙
- 支持对加工后的 glTF/GLB 独立重新 layout，进行区块复用、相似合并、纯色压缩和密集装箱
- 支持多个 glTF/GLB 合并颜色贴图，保留各模型为独立 glTF，共用一张 PNG，并同步更新 UV

纹理二次加工使用两个脚本：`test_cub_export.sh` 导出加工版，绘制完成后运行 `test_cub_relayout.sh` 生成发布版。脚本参数逐行排列，可直接编辑；完整流程和配置见下文“用于二次绘制的 UV 展开”及“二次加工后重新打包模型和纹理”。原有 VOX 导出默认仍使用 `atlas` 模式。

按用途选择脚本：

| 用途 | 脚本 | 结果 |
|---|---|---|
| 从 VOX 导出方便绘制的模型和 UV | `./test_cub_export.sh` | 加工版模型、独立绘画贴图、SVG 和 UV JSON |
| 将一个加工后的模型重新 layout | `./test_cub_relayout.sh` | 新的发布版模型和紧凑贴图 |
| 将多个模型的颜色贴图合并 | `./merge_gltf_textures.sh` | 各自独立的新 glTF，共用一张 PNG，自动更新 UV |

多模型合并可以直接读取已绘制完成的加工版模型，也可以读取已有发布版模型，无需先对每个模型单独重排。日常编辑保留源模型和源贴图，发布时按需要运行单模型重排或多模型合并。

供 AI 或自动化脚本接入的精简调用约定见 [`docs/AI_USAGE.md`](docs/AI_USAGE.md)，包含入口选择、CLI / Python API、参数对应关系、输出约定和验证方式；根目录 [`AGENTS.md`](AGENTS.md) 提供仓库内 AI 工作入口。

## 目录与归档

根目录保留 `btp_vox/`、`tests/`、`docs/`、转换与纹理处理脚本，以及对应的示例输入和 `exports/` 输出。`hero1/hero2` 的基础 `.vox`、`.gltf`、`.bin`、`.png` 和 UV JSON 继续作为导出与合并示例资源，模型、缓冲和贴图需配套保留。

角色动画、Blender 工程及备份、预览、验证报告、船模型、动画模板、历史 `old/` 和动画技能已移至 `bk/`，内部保留原目录结构。基础角色输入在 `bk/` 中另存一份，方便后续独立管理；两处副本修改后不会自动同步。文件迁移和副本清单见 [`bk/archive_manifest.json`](bk/archive_manifest.json)。

动画脚本现在位于 `bk/scripts/`，运行历史命令时先进入 `bk/`。归档中的旧报告、说明或 Blender 工程可能仍记录原来的绝对路径，重新运行前检查外部资源引用与输出位置。

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

查看多模型贴图合并参数：

```bash
python3 -m btp_vox.merge_textures --help
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
- `--paint-unwrap box|surface`：绘画展开算法；CLI 默认 `box` 保持旧布局，`surface` 按表面共边关系展开，需要 `--uv-mode paint`；专用导出脚本使用 `surface`
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
- `test_cube_paint_islands.json`：部件、展开片、面方向、深度及四边形像素坐标；`surface` 还记录连接边、折边和接缝
- `test_cube_paint_uv.json`：保持原格式的每个模型 UV 包围矩形

两个脚本均使用逐行排列的 `args=(...)` 参数列表，直接打开 `.sh` 修改。输入、输出路径在脚本顶部的 `in_vox` / `in_model` / `in_tex` / `out_dir` 等变量中修改。

两个脚本当前采用的默认配置：

| 配置 | `test_cub_export.sh` | `test_cub_relayout.sh` |
|---|---|---|
| 输入 | `test_cube.vox` | 加工版 `.gltf` 和 PNG |
| 输出目录 | `exports/test_cube_paint/` | `exports/test_cube_runtime/` |
| 展开方法 | `--paint-unwrap surface`，沿真实公共边连接 | 使用输入模型的 UV 读取贴图 |
| 像素密度 | 每体素面单位 `8×8` 像素 | 保留输入像素，纯色块可折叠 |
| 边距 | `--tex-pad 0` | `--tex-pad 0` |
| POT | 开启 `--tex-pot` | 开启 `--tex-pot`，可改为 `--no-tex-pot` |
| 区块复用 | 关闭，保持可独立绘制 | 开启相似合并，阈值 `0.98`、最大通道色差 `25` |
| 纯色压缩 | 关闭 | 开启 |
| 装箱 | 按模型集中排布 | `dense` 密集装箱 |

这是示例脚本的配置，不等于 CLI 默认值。例如两个 CLI 的边距默认均为 `2`；重排 CLI 默认不合并区块、不压缩纯色，使用 `shelf` 装箱。

导出脚本默认每体素面单位为 `8×8` 像素、外边距为 `0`，并开启 POT。加工版不默认增加外边距，避免十字展开宽度从 `512` 增加到 `520` 后，被 POT 扩展到 `1024`。例如在参数列表中编辑：

```bash
  --paint-unwrap       surface  # 改为 box 使用原包围盒/深度投影
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
  --paint-unwrap surface \
  --character-apart \
  --tex-texel-scale 8 \
  --tex-pad 0 \
  --tex-pot \
  --tex-out paint.png \
  --uv-template-out paint_uv.svg \
  --uv-islands-out paint_islands.json
```

绘画模式支持两种展开算法，均按 model 分组排布：

- `surface`：先沿真实公共边连接共面四边形，再以公共边为折线，逐步把相邻平面摊到同一个 UV 岛。共边可只占长边的一部分，支持贪心合并网格的 T 形接点；仅点接触的区域不会连接。窗洞、台阶及不同朝向/深度的表面，只要相邻且展开后不重叠，就能留在同一岛内。
- `surface` 的平面区域可有凹形轮廓或孔洞；碰撞检查使用实际四边形占用区域。放置后会重叠的面另开接缝，非流形边上连接关系不唯一的部分也切开。每个生成的岛都通过真实公共边连续相连，不会把不相连的表面仅因方向或深度相同而归为一岛。
- 展开只做整像素的平移和 90° 旋转，不拉伸、不改变像素密度、不镜像外向表面；不改变网格拓扑。完整六面方盒仍保留熟悉的十字布局：`l / f / r / k`，`t` 在 `f` 上方，`b` 在 `f` 下方。字母使用 MagicaVoxel 模型空间，含义与 `--cull` 一致。
- `box`：保留原来的外包围盒六面展开；不同深度的凹槽、台阶或内部表面按方向和深度分别投影。同组可能包含互不相连的面。旧 CLI 调用不加 `--paint-unwrap` 时继续使用此方法，方便复现旧布局。

两种算法的共同规则：
- 不复用相同、镜像或子矩形纹理，不压缩纯色面；模型名中的面复用规则（例如 `@lrfk@tb`）也会被忽略。不同 shape 对同一 model 的引用会获得独立绘画区域，方便单独绘制左右肢体。
- 原始体素颜色直接写入新布局，网格形状、顶点、法线、坐标系、锚点和节点变换保持原流程；cutout 和剔面仍然有效。
- `--tex-texel-scale 8` 表示每个体素面单位占 `8×8` 像素；CLI 默认仍为 `1`，专用脚本默认 `8`。`--tex-pad` 控制展开片外的颜色扩边；相连的面之间没有间隔。
- 支持 `--tex-pot`、`--tex-square`、`--tex-fixed-size`、纹理格式和 UV2。固定尺寸不够时会报错，不会自动缩小绘制分辨率。
- 本模式固定使用原色烘焙与按模型分组；`--tex-style`、`--tex-layout`、`--tex-tight-blocks`、纹理复用、纯色压缩、face alias 和 `--tex-inset` 参数不参与绘画布局。旧 `atlas` 模式继续使用这些参数。

SVG 与详细 JSON 均使用 PNG 的左上角像素坐标，四边形坐标按原始顶点顺序排列。`surface` 的 SVG 用橙色实线标记切开的接缝、蓝色虚线标记已展开连接的折边，细线标记原四边形边界。JSON 顶层 `unwrap` 记录算法；每个岛的 `connections` 记录连接的两个四边形、像素端点及是否为折边，`seams` 记录切开的边段，也支持部分边连接。

`--uv-flip-v` 仍仅翻转导出的网格 UV，元数据中会记录该设置，参考图继续对应实际 PNG。固定模型和配置的输出是确定的；改变模型尺寸、可见表面、实例数量或展开算法后布局可能变化，应重新导出参考图。已有加工贴图应继续与原模型配套，不能直接替换为另一算法生成的模型。

POT 仅约束画布宽高，不保证 UV 填满画布。16×16×16 方盒的六个独立表面共需 `6×16×16=1536` 个像素，无法恰好填满面积也是 2 的幂的画布。当前连续十字展开为 4×3 个面：每体素 1 像素、无边距时为 `64×48`，POT 后为 `64×64`；每体素 8 像素则为 `512×384`，POT 后为 `512×512`。保留连续展开本身也会留下空白，第 4 步再紧凑打包。

绘制时保留 PNG 的尺寸和文件名，保存后重新载入模型贴图即可；另存为其他文件时修改重排脚本的 `in_tex`。重新运行第 2 步会覆盖同名加工版输出，绘制成果请另存备份。

当前仓库的复杂 `test_cube.vox` 包含房屋和木箱。相同的每体素 8 像素、无边距、POT 配置下，`box` 生成 48 个投影组和 `4096×8192` 贴图；`surface` 生成 6 个连续 UV 岛（房屋 5 个、木箱 1 个）和 `4096×4096` 贴图，面积减少 50%。逐面核对原始颜色和像素密度，并检查无表面重叠、岛内连通及几何保留；输出仍可直接交给第 4 步重新 layout。

`surface` 使用确定性的贪心接缝选择，目标是便于绘制的连续岛，不保证最少岛数、最短接缝或最小画布。复杂曲面会留下必要的接缝；连续展开及 POT 也仍可能留白。算法适用于本工具生成的轴对齐体素四边形；如需手工指定接缝，可在 Blender 中继续编辑，并以匹配的新模型/贴图执行第 4 步。

## 二次加工后重新打包模型和纹理

完整工作流程为：MagicaVoxel 建模 → `--uv-mode paint` 导出加工版 → 绘制/烘焙颜色贴图 → 独立重排工具生成发布版。

第 4 步只读取**加工后的 `.gltf` / `.glb` 和实际颜色贴图**，不读取 `.vox`，也不需要之前的 UV JSON。CLI 未传 `--texture` 时使用模型引用的贴图；示例脚本显式指定 `in_tex`，可修改为另外保存的绘制结果，也可以注释 `--texture` 整行改为自动读取。

木箱专用脚本：

```bash
# 按脚本中的 POT 配置导出到 exports/test_cube_runtime/
./test_cub_relayout.sh

# 脚本当前开启 POT；需要非 POT 时，将 --tex-pot 改为 --no-tex-pot
# 另存的加工模型和贴图：修改脚本顶部的 in_model 和 in_tex 路径
```

与示例脚本优化设置对应的通用命令：

```bash
python3 -m btp_vox.repack \
  --input edited.gltf \
  --texture finished.png \
  --output runtime/model.gltf \
  --tex-pad 0 \
  --tex-pot \
  --merge-similar \
  --similarity-threshold 0.98 \
  --similarity-max-error 25 \
  --compress-solid-patches \
  --pack dense \
  --report-out runtime/repack_report.json
```

输出为新的 `.gltf + .bin + .png`；输出扩展名为 `.glb` 时，纹理会嵌入新 GLB。`--tex-out` 可以指定外部 PNG 路径，对 GLB 则是额外输出一份贴图副本。

工具依据三角形位置和 UV，沿几何折角与 UV 接缝拆分纹理区域，打散加工版十字展开的空白；直接复制像素并更新 UV，保留非纯色区块的原像素密度。需要在折角处分离共用顶点时，也会复制对应的权重、其他顶点属性和形变数据。节点变换、骨骼和动画继续沿用。加工版文件不会被覆盖；源文件与输出路径冲突会报错。再次运行会覆盖同名发布版输出。

- `--tex-pot` / `--no-tex-pot`：选择 POT / 非 POT，默认非 POT。
- `--tex-pad`：纹理块四周的像素边距，默认 2；复制原贴图中对应的邻接像素，保留原边缘的线性采样颜色。
- `--deduplicate`：可选，复用最终像素和边距都完全相同的纹理块；当前脚本已开启。`--merge-similar` 也包含完全匹配去重，无需同时开启。
- `--merge-similar`：合并相似区块，也尝试旋转和翻转，自动修正 UV。只比较尺寸相同或旋转后尺寸相同的区块，不缩放图案。
- `--similarity-threshold 0.98`：最低相似度，定义为 `1 - RGB 平均绝对误差 / 255`。越高越严格，设为 `1` 时只接受完全匹配；这不是感知相似度或 SSIM。
- `--similarity-max-error 16`：每个 RGB 通道允许的最大差值，用于保护平均误差可能掩盖的小图案。alpha 必须逐像素完全相同，边距也参与比较。每块直接与保留的原始代表块比较，避免链式合并累积误差。
- `--compress-solid-patches`：把颜色和 alpha（包括边距）都一致的纯色区块压成单像素，UV 改为采样该像素中心；保留几何、权重和动画。有渐变、烘焙阴影或细节的区块不会这样压缩。
- `--pack dense|shelf`：`dense` 使用 MaxRects 尝试多种宽度与排序，保留快速装箱作为面积上限；`shelf` 使用原有快速装箱。

在 `test_cub_relayout.sh` 的参数列表中直接编辑，例如：

```bash
  --tex-pad                 0
  --tex-pot                        # 改为 --no-tex-pot 关闭 POT
  --merge-similar                  # 注释整行关闭相似合并
  --similarity-threshold    0.98
  --similarity-max-error    25
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

## 多个 glTF 共用一张纹理

入口为 Python 程序 [`btp_vox/merge_textures.py`](btp_vox/merge_textures.py) 和调用脚本 [`merge_gltf_textures.sh`](merge_gltf_textures.sh)。程序额外生成新的 glTF，并将新模型的顶点 UV 更新到公共贴图；源 glTF、源缓冲和源贴图保持原样。

推荐工作流程：

1. 保留各模型的源 `.gltf` / `.glb` 及其绘制贴图。若在 Blender 中修改了网格或 UV，先导出与最终贴图匹配的模型。
2. 修改脚本顶部的 `in_models`，列出要合并的模型；模型引用的颜色贴图就是输入图像，支持外部图像和 GLB 内嵌图像。
3. 按需设置 POT、相似度、边距和纯色压缩，然后运行脚本。
4. 在 Blender 或引擎中加载输出目录里的新 glTF，并将其 `.bin` 和公共 PNG 一起作为发布资源。
5. 后续绘制回到源模型和源贴图，完成后再次运行合并脚本，更新整批发布版资源。

运行脚本：

```bash
./merge_gltf_textures.sh
```

脚本顶部的 `in_models=(...)` 是输入模型列表，默认使用仓库中的 `hero1.gltf` 和 `hero2.gltf`。可以逐行增加、删除或修改文件路径；`out_dir` 和 `out_tex` 分别控制输出目录和公共 PNG 路径。参数继续使用逐行排列的 `args=(...)`，便于直接修改。

默认输出到 `exports/shared_atlas/`：

```text
hero1.gltf / hero1.bin
hero2.gltf / hero2.bin
shared_atlas.png
merge_report.json
```

新模型中参与合并的颜色贴图均引用同一张公共 PNG，各自的 UV 已按新区块位置和旋转/镜像变换更新。模型分别保存，几何、层级、材质属性、骨骼、动画及形变数据按原模型保留；只在 UV 接缝需要拆分顶点时复制相应的顶点属性。输入文件不会被覆盖。不同目录中的同名模型会自动使用 `name.gltf`、`name_2.gltf` 等名称，报告中记录输入与输出的对应关系。

### 新 glTF 与 UV 的关系

| 项目 | 处理方式 |
|---|---|
| 源 `.gltf` / `.glb`、缓冲和贴图 | 保留，供后续编辑及重新生成使用 |
| 新 `.gltf` | 每个输入模型各生成一份，引用对应的新缓冲和公共 PNG |
| 新 `.bin` 中的顶点 UV | 按新区块位置和画布尺寸重新计算，并补偿区块旋转、镜像 |
| 区块复用 | 多个模型表面可以使用公共贴图中的同一区域 |
| 纯色压缩 | 对应 UV 定位到保留像素的中心 |
| 颜色贴图使用的 UV 集 | 根据材质选择更新 `TEXCOORD_0` 或其他 UV 集，其他 UV 集按原顶点保留 |

区块在图集里旋转或镜像后，模型上显示的图案方向仍与输入一致。相似合并可能带来阈值允许的颜色变化；只允许完全匹配时不会进行近似颜色替换。

使用输出目录里的新 glTF 加载公共贴图。源模型的 UV 对应源贴图，直接给源模型替换公共 PNG 会产生错位。移动输出资源时，保留 glTF 到 `.bin` 和公共 PNG 的相对路径；再次运行相同配置会更新同名输出。改变模型列表或绘制内容后，布局可能变化，应一起更新本批新模型和公共贴图。

### 配置与命令

脚本中的模型列表示例：

```bash
in_models=(
  "$SCRIPT_DIR/hero1.gltf"
  "$SCRIPT_DIR/hero2.gltf"
  # "$SCRIPT_DIR/another_model.glb"  # 需要时取消注释或继续添加路径
)
```

通用命令：

```bash
python3 -m btp_vox.merge_textures \
  --input model_a.gltf model_b.gltf model_c.glb \
  --output-dir exports/shared_atlas \
  --tex-pot \
  --tex-pad 0 \
  --merge-similar \
  --similarity-threshold 0.98 \
  --similarity-max-error 16 \
  --compress-solid-patches \
  --pack dense
```

- 输入支持 `.gltf`、`.glb`，也支持一个模型引用多张颜色贴图；输出统一为独立 `.gltf + .bin`，共享外部 PNG。
- 区块从所有输入模型共同收集、统一装箱，因此相同或相似区块可以跨模型复用。支持现有的 8 种旋转/镜像匹配及 UV 修正、alpha 完全匹配约束、纯色压缩和密集装箱。
- `--tex-pot` 开启 POT，`--no-tex-pot` 关闭；`--tex-out` 可指定公共 PNG 路径，模型中保存相对引用。
- 示例脚本默认开启相似合并、纯色压缩、POT 和 `dense` 装箱，边距为 `0`。CLI / Python API 默认关闭这些优化，边距为 `2`、装箱为 `shelf`，可按需开启。
- 如果只允许无颜色变化的复用，可将相似度设为 `1`、最大色差设为 `0`，同时保留旋转/镜像比较；只开 `--deduplicate` 时只比较原方向的完全相同区块。
- 支持范围与单模型重排一致：三角网格、0–1 范围 UV、Base Color 贴图。法线/粗糙度等多通道贴图、纹理变换、范围外重复 UV 和压缩模型需先处理。不支持的输入会报错。

例如，保留旋转/镜像匹配且只合并完全相同的区块：

```bash
  --merge-similar
  --similarity-threshold   1
  --similarity-max-error   0
```

`--merge-similar` 已包含完全匹配去重，无需额外开启 `--deduplicate`。相似度采用 `1 - RGB 平均绝对误差 / 255`，最大通道色差和 alpha 一致性会同时检查；参与比较的是像素图案，不仅是平均颜色。普通区块需尺寸相同或旋转后尺寸相同，不缩放图案。

在 Python 中调用：

```python
from btp_vox.merge_textures import merge_textures

report = merge_textures(
    ["hero1.gltf", "hero2.gltf"],
    "exports/shared_atlas",
    pad=0,
    pot=True,
    merge_similar=True,
    similarity_threshold=0.98,
    similarity_max_error=16,
    compress_solid=True,
    packing="dense",
)
print(report["packed_size"], report["cross_model_merged_patches"])
```

### 报告与验证

报告包含各模型输出路径、贴图尺寸、区块总数、保留数、`cross_model_merged_patches`（跨模型共用的区块数），以及相似匹配的误差和变换。`similarity_matches` 中的 `model` / `owner_model` 对应 `models` 数组的下标。输入贴图总面积按原图像内容去重后统计；POT、边距及输入原布局都会影响面积变化，不保证每批资源都缩小。

仓库中 `hero1.gltf` 和 `hero2.gltf` 的脚本配置实测：两张输入贴图分别为 `32×32` 和 `16×32`，输出共用一张 `16×16` POT 贴图，610 个区块保留 18 个，其中 52 个区块跨模型复用，本次没有近似颜色替换。

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
bash -n test_cub_export.sh
bash -n test_cub_relayout.sh
bash -n merge_gltf_textures.sh
```

相关测试位于 `tests/test_paint_uv.py`、`tests/test_repack.py` 和 `tests/test_scene_transforms.py`，覆盖展开颜色、岛内连通、T 形接点、窗洞/台阶/非流形边、接缝元数据、旋转/翻转 UV、相似度约束、alpha、纯色中心采样、密集装箱、输入文件保护，以及几何、骨骼、动画和形变数据的保留；也覆盖连续展开后绘制贴图再重排的流程。这些检查不代替目标游戏引擎验证。

`tests/test_merge_textures.py` 另外覆盖跨模型共享、8 种变换后的 UV、不同尺寸及多张源贴图、GLB 输入、动画蒙皮、UV 集选择、同名文件、路径中的空格、原文件保护和确定性输出。

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
