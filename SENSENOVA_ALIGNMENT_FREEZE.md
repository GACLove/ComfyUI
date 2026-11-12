# SenseNova-U1 推理对齐冻结说明

状态：**冻结用于自检**。验证日期：2026-11-12。

本分支保留 ComfyUI 与 SenseNova-U1 官方仓库推理结果对比所需的实现和已验证配置。它用于复现、诊断和自检，不作为普通 ComfyUI workflow 的默认采样配置。

## 分支和代码基线

| 项目 | 固定版本 |
| --- | --- |
| Fork | `GACLove/ComfyUI` |
| 分支 | `gp/sensenova-u1-alignment` |
| ComfyUI 推理代码 | `5ba4c4ae907190c08c2704091e8a2becda577b3a` |
| SenseNova 功能提交 | `94c71af2124ad892dd30756a7432be8c3929db8d` |
| SenseNova-U1 对比代码 | `f3e3781160b4afa01a92f0d21ca3ebefc39ec103` |

分支 HEAD 包含本说明的文档提交；**实际验证的运行代码固定为 `5ba4c4ae`**。文档提交不改变模型、节点、采样器或加载逻辑。

冻结后不例行 rebase、同步上游或继续处理轻微审查建议。需要修改时，从此基线另建分支并重新验证，不覆盖这份已验证快照。

## 已验证环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA H100 80 GB |
| Torch / CUDA | `2.8.0+cu128` |
| FlashAttention | `2.8.3` |
| Transformers | `5.15.0` |
| comfy-aimdo / comfy-kitchen | `0.6.0` / `0.2.41` |
| 权重 | U1.6 自修正 step400 EMA，BF16 |
| ComfyUI 模型文件 | `SenseNova-U1.6-8B-MoT-BF16-selfcorr-step400-ema.safetensors` |
| ComfyUI 加载方式 | 原生 DynamicVRAM / `ModelPatcherDynamic` |
| ComfyUI 卸载方式 | 同步卸载，`--disable-async-offload` |
| SenseNova-U1 卸载方式 | 同步卸载，`prefetch_count=1`；各 interleave 用例独立进程 |

本次验证通过独立脚本调用节点执行路径。脚本使用以下 ComfyUI 启动选项：

```text
--use-flash-attention
--disable-async-offload
--disable-cuda-malloc
--disable-comfy-compiler
--disable-cuda-graphs
--reserve-vram 40
```

`--reserve-vram 40` 对应本次共享 GPU 的显存预留配置，不代表通用最低显存要求。本次结果不是浏览器导入 workflow 后的端到端 UI 验证。

共同推理参数：seed `42`，steps `50`，CFG `4`，image CFG `1`，shift `3`，Euler，`cfg_norm=none`，thinking 关闭。自修正使用图像生成 system prompt 和非 thinking interleave；文本上限为 `4096` tokens，实际均正常 EOS 结束。

## 对齐节点的使用约定

节点 ID：`SenseNovaU1InferenceAlignment`；显示名：`SenseNova-U1 Inference Alignment`。

- 将节点输出的 `MODEL` 接到 `CFGGuider`；需要生成文本时，也接到 `TextGenerateSenseNova`。
- 将 `NOISE`、`SAMPLER`、`SIGMAS` 接到 `SamplerCustomAdvanced`。
- 普通非 thinking T2I/edit 不需要 `TextGenerateSenseNova`。
- interleave 每轮保持同一 seed。对齐节点和文本生成节点接收相同的本轮输入 history/conditioning，以及上一轮生成的 `samples`；第一轮省略 `samples`。
- 对齐节点根据已生成图像的尺寸重放随机序列。外部参考图不占用生成图像的噪声序列位置。

这些连接之外，输入 prompt、system prompt、参考图、分辨率、采样参数和运行环境也必须一致。

## 验证结果

两侧均从记录的 Git 版本重新执行推理。**5 个用例、7 张生成图，原始输出张量数值与 PNG 像素全部一致。** 外部输入参考图不计入生成图数量。

| 用例 | 宽 × 高 | SenseNova-U1 / ComfyUI 生成图数 | 原始张量最大绝对误差 | PNG 最大通道误差 |
| --- | --- | --- | --- | --- |
| T2I | 1024 × 1024 | 1 / 1 | 0 | 0 |
| 参考图 Edit | 832 × 1248 | 1 / 1 | 0 | 0 |
| 自修正 row-7 | 2720 × 1568 | 2 / 2 | 0 | 0 |
| 自修正 row-15 | 1856 × 2304 | 2 / 2 | 0 | 0 |
| 带外部参考图的自修正 | 2720 × 1568 | 1 / 1 | 0 | 0 |

完整生成文本、生成图数量和 EOS 状态也一致；输入及 system prompt 的一致性单独核对。

原始输出保留各自实际 dtype：SenseNova-U1 为 BF16，ComfyUI 为 FP32 容器。比较时将二者无损转换到 FP32，要求逐元素相等，未使用 `atol`/`rtol` 容差，也未先将 ComfyUI 结果舍入到 BF16。另行确认 ComfyUI 输出中的值可由 BF16 精确表示。因此这里的“一致”指张量数值和像素，不表示文件字节或存储 dtype 相同。

## 适用范围和限制

**复现以上零误差结果需保留 DynamicVRAM 和 `--disable-async-offload`。** 在本次共享 GPU 配置中，启用异步卸载时的 Edit 输出出现差异；关闭异步卸载后，无诊断探针的全量五例复跑全部一致。该证据尚未定位到具体故障函数，也不表示所有异步卸载配置都会失败。

普通 ModelPatcher 加异步卸载的探测运行也出现过 T2I 差异，未纳入通过配置。本冻结分支没有为此修改 ComfyUI 核心加载、卸载或 ModelPatcher。

结论仅覆盖表中 U1.6、BF16、H100、注意力后端及输入配置。它不证明普通 ComfyUI 默认采样器、其他设备、FP8/INT 量化模型、任意 prompt，或 U1.5 thinking/interleave 均达到像素级一致。

## 外部验证材料

验证脚本、输入图片、原始输出、PNG、日志和诊断报告单独归档，不提交到本分支；`script_examples/` 同样不纳入提交。

- 证据目录标识：`alignment-freeze-20261112`。
- `freeze.json`：冻结版本及验证摘要。
- `environment.json`、`provenance.json`：环境、代码来源和校验信息。
- `comparison-comfy-sync.json`：本说明采用的五例通过报告。
- `results/native/`、`results/comfy-sync/`：对应的输出和每例参数。
- `README.md`、`verify.py`：归档内的复现说明和验证入口。
- `evidence.tar.gz` 的 SHA-256：`b36329bc5e43b02b107d0f96ab1f99111b7925bf81eef3f0537daba5cc3f5a42`。

模型权重及验证材料不随此 Git 分支分发。需要重跑时，使用同一份证据归档中的输入、脚本、环境记录和固定代码版本。
