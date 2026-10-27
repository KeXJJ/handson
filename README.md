# HandsOn π0.5 微调：A100 检查记录与 H100 迁移包

用于 `t09_insert_peg`（源任务 `b03_peg_in_socket`）的独立训练代码包。复用 **HandsOn 官方 OpenPI 训练器**，包含固定版本的必要源码、依赖锁、数据/基础权重下载器、EEF52 动作头转换和检查点核验。

**当前验证状态**：A100 上已完成 batch=1 的 5 个真实优化步，LoRA（包括 action expert）和动作头权重发生更新；完整检查点已保存并独立重载。JAX 0.5.3 在训练完成后的资源清理阶段仍出现段错误（139），所以完整运行状态为未通过。**H100 尚未实测**，请先执行下面的 5 步检查。

## 当前到底微调哪些参数？

| 模块 | 更新方式 |
|---|---|
| PaliGemma / Gemma 2B 语言骨干 | 注意力及 FFN 的 LoRA，rank=16；其余语言骨干参数冻结 |
| **Gemma 300M action expert** | **注意力及 FFN 的 LoRA，rank=32**；其余 expert 骨干参数冻结 |
| 视觉编码器 | 常规参数更新 |
| 动作输入、输出投影 | 常规参数更新 |
| 时间 MLP | 常规参数更新 |

这是官方配置中的 **LoRA + 部分模块全参数更新**，并不是整个 π0.5 全参微调。配置来自 [`make_dual_arm_config`](third_party/handson/openpi/src/openpi/training/handson_configs.py)，冻结规则来自 [`Pi0Config.get_freeze_filter`](third_party/handson/openpi/src/openpi/models/pi0_config.py)，rank 定义在 [`gemma.py`](third_party/handson/openpi/src/openpi/models/gemma.py)。

状态和动作均为 **52 维绝对世界掌部目标**：右侧 26 维在前、左侧 26 维在后，预测 30 步动作块。保留真实指令动作，不用未来实测状态替代。模型输入为三路 RGB，沿用官方 `224×224 resize-with-padding`；与仿真相机的原生/半尺寸渲染设置独立。

## H100 上开始使用

环境目标：Linux x86_64、Python **3.12.15**、单张 H100。`requirements-lock.txt` 保留 A100 上实测版本（JAX 0.5.3、Flax 0.10.2、PyTorch 2.7.1 / CUDA 12.6、Orbax 0.11.13），并包含首次安装时补齐的 `jsonlines`。

```bash
git clone https://github.com/KeXJJ/handson.git
cd handson
# 如果尚未安装 uv，请先按 https://docs.astral.sh/uv/getting-started/installation/ 安装。
bash scripts/install.sh
source scripts/env.sh

# 默认为可见 GPU 0。其他设备可在 source 前设置 CUDA_VISIBLE_DEVICES。
# export CUDA_VISIBLE_DEVICES=1

# 只下载 8 条训练轨迹、2 条验证轨迹；保持官方按采集会话划分的 split。
.venv/bin/python download_assets.py dataset --train-episodes 8 --validation-episodes 2

# 完整基础模型仍须下载；29 个对象，12,441,749,581 字节。
.venv/bin/python download_assets.py base

# 新实验名，避免覆盖旧日志；执行 5 步并核验检查点。
bash scripts/run_smoke.sh h100_smoke_001
```

`HANDSON_WORK_DIR` 默认是仓库内的 `workdir/`，已被 Git 忽略；可在 `source scripts/env.sh` **之前**设置到大容量磁盘。`HANDSON_DATASET_DIR` 可单独指定数据目录。默认关闭 W&B、GPU 显存预占和首次 GPU 自动调优。`HANDSON_DISABLE_AUTOTUNE=0` 可以保留 XLA 默认调优行为，但不代表退出问题已修复。

若 Hugging Face 直连不可用，可显式设置 `HF_ENDPOINT=https://hf-mirror.com`。下载器固定上游 revision，基础权重同时固定 [对象 generation](BASE_CHECKPOINT.json)，数据文件以官方 SHA256 校验，基础权重以 GCS CRC32C 校验（提供 MD5 时同时核验）。基础下载支持分块续传；不需要手动登录 Google Cloud。

检查结果在 `workdir/results/<实验名>.json`，训练日志在 `workdir/logs/<实验名>.log`。**只有训练干净退出且核验通过，`passed` 才会为 true**；若训练末尾退出失败但检查点已完整提交，脚本仍会独立核验，并保留真实非零退出状态。

## 数据集下载链接

- 官方数据集：[Cyan666/HandsOn-Datasets](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets)
- 固定版本：[7ad38776d91e4e13c02dad4e6ad892e8b8a1d187](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets/tree/7ad38776d91e4e13c02dad4e6ad892e8b8a1d187)
- 本任务共有 **500 条轨迹**：412 条训练、88 条验证，共 204,487 帧；任务数据/视频文件约 2.72 GB。

默认只选择 8+2 条，用于迁移检查。可以先查看计划；需要完整任务数据时再明确选择 `--all-task`：

```bash
.venv/bin/python download_assets.py dataset --plan-only
# 明确请求完整插销任务时使用，不会下载其余任务：
.venv/bin/python download_assets.py dataset --all-task
```

原始视频为 front 854×480、两路 wrist 848×480。归一化仅使用选中的训练轨迹；验证集不参与训练或归一化。训练时排除不足 30 帧的末尾动作窗口，避免填充动作进入无掩码损失。发布包自带的 reader 和 prompt sampler 按原字节保留。

## Checkpoint 下载链接

### 官方 π0.5 基础权重（从头开始本任务微调）

- 官方说明：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- GCS 目录：`gs://openpi-assets/checkpoints/pi05_base`
- [HTTPS 目录列表](https://storage.googleapis.com/openpi-assets/?prefix=checkpoints/pi05_base/)
- [对象下载清单 API](https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints%2Fpi05_base%2F)

这是 Orbax 多文件目录，**不是一个单独的 `.safetensors` 文件**。直接运行 `download_assets.py base` 会下载并验证完整目录。`scripts/run_smoke.sh` 首次运行时按官方转换器的方式，将动作输入/输出投影从 32 扩展至 52 维；前 32 维保留基础权重，新维度初始化。

官方 HandsOn 仓库未公开论文任务微调权重，不能把这里的 `pi05_base` 称为 HandsOn 微调 checkpoint。

### 已有 A100 的 5 步 checkpoint

本仓库保留[检查报告](examples/a100_smoke/smoke_result.json)，**没有上传约 8.9 GB 的检查点或生成虚构的 HTTP 下载地址**。已有产物可从 A100 直接迁移到 H100；相对原实验目录的位置为：

```text
pi05/checkpoints/t09_insert_peg_pi05_lora/smoke_5/4/
  params/       # 推理参数，包含已更新的 LoRA
  assets/       # 对应训练集的归一化统计
  train_state/  # 优化器状态
```

例如在 H100 上使用 `rsync`（替换主机和源路径）：

```bash
mkdir -p "$HANDSON_WORK_DIR/checkpoints/t09_insert_peg_pi05_lora/smoke_5"
rsync -a --partial --info=progress2 \
  A100_HOST:/path/to/handson-repro/pi05/checkpoints/t09_insert_peg_pi05_lora/smoke_5/4 \
  "$HANDSON_WORK_DIR/checkpoints/t09_insert_peg_pi05_lora/smoke_5/"
```

保持目录结构，不单独重命名 shard。该 5 步产物只用于运行检查，不代表已学会插销任务。若要继续原实验，还须迁移相应 config、归一化、原数据选择、初始审计张量及日志；新实验建议从官方基础权重开始。

## 分步检查与训练

```bash
source scripts/env.sh
# 首次 tokenizer 下载及 EEF52 转换由 run_smoke.sh 完成。
JAX_PLATFORMS=cpu .venv/bin/python prepare_stats.py
JAX_PLATFORMS=cpu .venv/bin/python preflight.py
.venv/bin/python preflight.py --gpu

# 仅在 5 步检查和资源条件确认后，由使用者自行选择训练长度。
.venv/bin/python train_task.py --steps 1000 --batch-size 1 --exp-name my_run
# 对应实验的完整目录已存在时可显式 --resume；不会自动覆盖实验。
```

`train_task.py` 调用未修改的官方 `scripts/train.py`，包装层负责本任务配置、数据划分和初始化张量审计。短程检查缩短学习率预热；默认学习率上限 5e-5。未迁移完整论文的 66 任务训练/评测，也未运行闭环策略成功率评测。

## A100 实测记录与已知问题

| 实际执行顺序 | 训练 Loss | 梯度范数 |
|---|---:|---:|
| 1 | 0.7543 | 1.6901 |
| 2 | 0.6082 | 6.0742 |
| 3 | 0.6195 | 1.1639 |
| 4 | 0.5287 | 0.8423 |
| 5 | 0.6316 | 1.6037 |

8 条训练轨迹 3,196 帧，2 条验证轨迹 937 帧。可用完整窗口分别为 2,964 和 879。抽查的 8 个 LoRA 张量及动作输出投影都发生更新；action expert 的 `_1` 分支 LoRA 也有约 2.24e-4 的最大绝对变化。检查点经 Orbax 完成提交并独立 CPU 重载。

关闭自动调优并使用单进程数据加载后能够完成上述 5 步；但在检查点写完后的 JAX GPU 资源清理阶段仍以 139 退出。示例报告明确保留 `core_finetuning_checks_passed=true`、`passed=false`，没有将退出故障隐藏为成功。H100 上是否重现此问题需实际检查，不能仅凭型号推断已解决。

## 源码与许可证

- HandsOn 固定提交：`be2b59aad016869e8cede1f485d64e3c2d191685`。
- [`UPSTREAM.json`](UPSTREAM.json) 记录 vendored 文件来源和逐文件 SHA256；`python scripts/check_vendor.py` 可核对原字节。
- OpenPI 与 LeRobot 保留 Apache-2.0、NOTICE 和原文件头；HandsOn 与发布数据 reader 的 MIT 声明保留在各自目录。
- Gemma 模型条款保留在 [`LICENSE_GEMMA.txt`](third_party/handson/openpi/LICENSE_GEMMA.txt)。权重不纳入 Git。
- Python 依赖通过锁定清单安装；Git 中只包含源码、说明和经过路径脱敏的本次检查报告，不包含数据、模型或访问令牌。上游源码中的历史示例路径保留原样，由本仓库配置覆盖。
