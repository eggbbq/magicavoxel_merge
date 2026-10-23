# AI 调用指南

此文档用于 AI、自动化脚本及工具集成，描述可直接调用的程序接口。详细算法和使用说明见 [README](../README.md)。

仓库根目录维护 VOX 导出、绘画 UV、单模型重排和多模型纹理合并；角色动画、Blender 工程、预览、报告、船模型和动画技能归档在 `bk/`，不属于这些工具的运行依赖。根目录 `hero1/hero2` 基础模型及配套资源仍供示例使用，`bk/` 内的同名文件为独立副本。迁移清单见 [`bk/archive_manifest.json`](../bk/archive_manifest.json)；调用工具时沿用根目录入口，不使用归档动画脚本。

## 1. 按任务选择入口

| 用户目标 | CLI 模块 | Python 入口 | 示例 SH |
|---|---|---|---|
| VOX 导出绘画用模型、连续 UV 岛和贴图 | `btp_vox.cli`，`--uv-mode paint --paint-unwrap surface` | `btp_vox.pipeline.convert` | `test_cub_export.sh` |
| 加工完成后，重排一个模型及其最终贴图 | `btp_vox.repack` | `btp_vox.repack.repack` | `test_cub_relayout.sh` |
| 多个模型共享一张颜色贴图 | `btp_vox.merge_textures` | `btp_vox.merge_textures.merge_textures` | `merge_gltf_textures.sh` |

多模型合并可直接读取加工完成的模型，不要求先逐个执行 `repack`。单模型和多模型重排仅依赖模型与最终贴图，不读取 VOX 或 UV JSON。普通 VOX 导出默认仍为旧 `atlas` 模式，绘画模式需要显式选择。

## 2. 执行环境与路径

运行环境要求 Python >=3.10。从 Git 安装需要 `git`，并且远程仓库已经包含 `pyproject.toml` 打包配置：

```bash
python3 -m pip install "btp-vox @ git+https://github.com/eggbbq/magicavoxel_merge.git"
```

在仓库根目录也可执行 `python3 -m pip install .`，开发时使用 `python3 -m pip install -e .`。依赖从 `requirements.txt` 自动安装；发行包名为 `btp-vox`，导入名为 `btp_vox`。

安装后可从任意目录调用，不需要 `PYTHONPATH`：

| 安装后的命令 | 模块入口 |
|---|---|
| `btp-vox` | `python3 -m btp_vox.cli` 或 `python3 -m btp_vox` |
| `btp-vox-repack` | `python3 -m btp_vox.repack` |
| `btp-vox-merge-textures` | `python3 -m btp_vox.merge_textures` |

三者均支持 `--help`，模块参数与命令参数相同，Python API 导入路径不变。发行包只安装 `btp_vox` 工具代码，不含示例资源、`.sh`、测试或归档；输入路径由调用方提供。需要示例脚本时另行克隆仓库。

未安装而直接从其他目录调用源码时，将仓库根目录加入 `PYTHONPATH` 或 Python 搜索路径；三个示例 SH 已处理此项。模块使用相对导入，应通过 `python3 -m btp_vox.<模块>` 执行。

自动化优先传入明确的输入、输出路径。示例 SH 默认读取仓库内资源，重新运行会更新同名输出；运行前读取其实际 `in_*`、`out_*` 和 `args`，不要假定用户仍使用初始配置。VOX 导出前创建目标目录；重排和合并程序会创建输出目录。

## 3. 可复制的 CLI 调用

以下命令中的路径是占位示例，需要替换。输出目录使用独立的 `exports/ai_job/`。

### 导出便于绘制的连续 UV

```bash
mkdir -p exports/ai_job/paint
python3 -m btp_vox.cli \
  --input input.vox \
  --output exports/ai_job/paint/model.gltf \
  --format gltf \
  --uv-mode paint \
  --paint-unwrap surface \
  --character-apart \
  --handedness right \
  --scale 0.02 \
  --pivot bottom_center \
  --tex-texel-scale 8 \
  --tex-pad 0 \
  --tex-pot \
  --tex-out exports/ai_job/paint/model.png \
  --uv-template-out exports/ai_job/paint/model_uv.svg \
  --uv-islands-out exports/ai_job/paint/model_islands.json
```

`surface` 沿实际公共边展开，重叠处切开接缝；保持像素密度和几何。`box` 是原包围盒/深度投影，用于复现旧布局。SVG 和岛 JSON 使用 PNG 左上角像素坐标；`--uv-flip-v` 只翻转网格 UV，参考图仍对应 PNG。

### 单模型重排

```bash
python3 -m btp_vox.repack \
  --input edited.gltf \
  --texture finished.png \
  --output exports/ai_job/runtime/model.gltf \
  --tex-pad 0 \
  --tex-pot \
  --merge-similar \
  --similarity-threshold 0.98 \
  --similarity-max-error 16 \
  --compress-solid-patches \
  --pack dense \
  --report-out exports/ai_job/runtime/report.json
```

未传 `--texture` 时读取模型引用的图像；传入的最终贴图必须与输入模型的 UV 匹配。此入口要求模型共用一张 Base Color 图像。

### 多模型共享纹理

```bash
python3 -m btp_vox.merge_textures \
  --input model_a.gltf model_b.gltf model_c.glb \
  --output-dir exports/ai_job/shared \
  --tex-pad 0 \
  --tex-pot \
  --merge-similar \
  --similarity-threshold 0.98 \
  --similarity-max-error 16 \
  --compress-solid-patches \
  --pack dense
```

每个输入模型可以引用多张颜色图像。此入口读取模型所引用的图像，没有单模型入口的 `--texture` 覆盖参数；另存的绘制贴图应先正确绑定到输入模型。输出统一为各自独立的新 `.gltf + .bin`，共享 `shared_atlas.png`；默认报告为 `merge_report.json`。同名模型依序使用 `name.gltf`、`name_2.gltf` 等名称，以报告中的输入输出映射为准。

## 4. Python API

单模型重排与多模型合并均返回报告字典；路径可用字符串或 `Path`。

```python
from btp_vox.repack import repack
from btp_vox.merge_textures import merge_textures

report = repack(
    "edited.gltf", "exports/ai_job/runtime/model.gltf",
    texture="finished.png",
    pad=0, pot=True,
    merge_similar=True, similarity_threshold=0.98, similarity_max_error=16,
    compress_solid=True, packing="dense",
    report_out="exports/ai_job/runtime/report.json",
)

batch_report = merge_textures(
    ["model_a.gltf", "model_b.glb"], "exports/ai_job/shared",
    pad=0, pot=True,
    merge_similar=True, similarity_threshold=0.98, similarity_max_error=16,
    compress_solid=True, packing="dense",
)
```

VOX 导出入口为 `convert(input_vox, output_glb, *, output_format="glb", options=...)`，返回 `None`；输出 glTF 时需显式传入 `output_format="gltf"`。主要配置放在 `PipelineOptions` 和 `AtlasOptions`：

```python
from pathlib import Path
from btp_vox.pipeline import AtlasOptions, PipelineOptions, convert

Path("exports/ai_job/paint").mkdir(parents=True, exist_ok=True)
convert(
    "input.vox", "exports/ai_job/paint/model.gltf",
    output_format="gltf",
    uv_template_out="exports/ai_job/paint/model_uv.svg",
    uv_islands_out="exports/ai_job/paint/model_islands.json",
    options=PipelineOptions(
        uv_mode="paint", paint_unwrap="surface", character_apart=True,
        handedness="right", scale=0.02, pivot="bottom_center",
        atlas=AtlasOptions(texel_scale=8, pad=0, pot=True),
    ),
)
```

## 5. 参数语义与默认值

以下为重排/多模型合并的 CLI 与 API 默认值，示例 SH 的配置可能不同：

| CLI | Python 关键字 | 默认值 / 含义 |
|---|---|---|
| `--tex-pad` | `pad` | `2`；复制源图像邻接像素作为边距 |
| `--tex-pot` / `--no-tex-pot` | `pot` | `False`；POT 约束画布宽高，不保证填满 |
| `--deduplicate` | `deduplicate` | `False`；仅原方向的完全匹配去重 |
| `--merge-similar` | `merge_similar` | `False`；含完全匹配，并比较 8 种旋转/镜像组合 |
| `--similarity-threshold` | `similarity_threshold` | `0.98`；`1 - RGB 平均绝对误差 / 255`，范围 0–1 |
| `--similarity-max-error` | `similarity_max_error` | `16`；RGB 单通道最大差值，范围 0–255 |
| `--compress-solid-patches` | `compress_solid` | `False`；纯色核心和边距都一致时折叠为单像素 |
| `--pack` | `packing` | `"shelf"`；`"dense"` 使用更充分的装箱搜索 |
| `--tex-out` | `texture_out` | 可指定输出 PNG 路径 |
| `--report-out` | `report_out` | 指定报告路径；批量合并默认会写报告，单模型需显式指定 |

只允许无颜色变化的复用，并保留旋转/镜像比较时，使用 `merge_similar=True, similarity_threshold=1, similarity_max_error=0`。单独传阈值不会开启相似合并。`merge_similar=True` 已包含完全匹配，无需同时开启 `deduplicate`。

alpha 和边距也参与匹配，alpha 必须完全相同；最大通道差值用于约束平均误差可能掩盖的小图案。普通区块仅比较尺寸相同或旋转后尺寸相同的图案，不缩放。每块直接与保留的代表块比较，不做误差累积的链式替换。

## 6. 输出与资源约束

- 所有 glTF 重排/合并都会生成新模型，并重算颜色贴图使用的 UV 集。图集中的位置、旋转、镜像及纯色像素中心采样均有对应的 UV 修正；原模型不能直接改用公共 PNG。
- 新 `.gltf` 引用新 `.bin` 和 PNG；移动或发布时保持相对路径有效。多模型合并保留独立模型，不合并场景、网格或材质，也不保证减少 draw call。
- 几何、节点层级和变换、骨骼、动画及形变数据保留；必要的接缝顶点拆分会复制相应属性。其他 UV 集保留，不会自动重新烘焙光照贴图。
- `repack` 输出 `.glb` 时嵌入纹理，`texture_out` 是额外 PNG 副本；VOX 导出指定 `.glb` 的 `texture_out` 时使用外部纹理。两条入口的行为不同。
- 保留源模型及最终绘制贴图用于后续编辑。改变批量输入或源贴图后，重新生成并一起发布本批模型、缓冲和公共 PNG；同名输出会被更新。
- 重排/合并拒绝覆盖输入模型、缓冲或图像。输出应与所有输入资源分开；VOX 导出会写入指定路径，需自行选择合适的输出目录。

支持三角形网格、0–1 范围 UV、普通 PNG/JPEG Base Color 图像。法线/粗糙度等多通道贴图联动、范围外重复 UV、纹理变换和压缩模型不受支持，遇到报错先检查输入条件，不将它们解释为已成功处理。当前没有发布阶段的子矩形复用，也没有利用 UV 岛空洞的交错装箱。

## 7. 结果检查与代码验证

自动化读取报告字典或 JSON，避免依赖控制台文案：

- `models[].input/output`：批量合并的输入输出映射。
- `packed_size`、`patches`、`unique_patches`：尺寸与合并规模；`cross_model_merged_patches` 是跨模型复用数量。
- 开启相似合并时，读取 `lossy`、`max_rgb_error`、`similar_merged_patches` 和 `similarity_matches`；批量匹配记录中的 `model` / `owner_model` 对应 `models` 数组下标。
- `max_rgb_error=0` 仅说明区块匹配没有颜色替换，不等于所有渲染器及过滤方式下均无误差；`resampled=False` 表示没有对图案缩放重采样，纯色折叠仍可减少纹理采样区域。

确认新模型、缓冲和 PNG 引用可读取，按目标渲染器检查实际效果。过滤、mipmap、边距与源资源的采样约定应一致。

修改实现时按范围运行现有测试；仅编辑文档不需要重新导出用户资源：

```bash
python3 -m unittest discover -s tests -p 'test_merge_textures.py' -v
python3 -m unittest discover -s tests -v
bash -n test_cub_export.sh
bash -n test_cub_relayout.sh
bash -n merge_gltf_textures.sh
```

相关实现：`paint_uv.py` / `surface_uv.py`（绘画 UV）、`repack.py`（区块提取、匹配与 UV 修正）、`merge_textures.py`（跨模型汇总与输出）。新增参数或改变默认值时，同步更新本指南和 README。
