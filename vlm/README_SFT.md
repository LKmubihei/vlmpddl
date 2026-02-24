# VLM SFT 训练/评测（`vlm/train/train_sft.py`）

这个仓库里用于 **Qwen2.5-VL LoRA SFT** 的脚本是：

- `vlm/train/train_sft.py`

默认配置已经对齐到本 repo 内的路径（训练/测试 JSON、deepspeed 配置、YOLO 权重等），**在 `open_reasoner` conda 环境中可直接运行**。

## 0. 环境准备

先进入项目根目录：

```bash
cd /root/Plan/vlmpddl
```

激活环境（你当前机器的约定）：

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate open_reasoner
```

（可选）快速检查关键依赖：

```bash
python -c "import torch, transformers, datasets, peft, deepspeed; print('deps_ok')"
```

## 1. 直接训练（默认路径）

> **注意**：启用 DeepSpeed 时请使用 `deepspeed` 或 `torchrun` 启动器，不要用裸 `python`。

### 单卡训练（推荐）

```bash
deepspeed --num_gpus 1 vlm/train/train_sft.py --mode train
```

或用 `torchrun`：

```bash
torchrun --nproc_per_node=1 vlm/train/train_sft.py --mode train
```

### 多卡训练（如 4 卡）

```bash
deepspeed --num_gpus 4 vlm/train/train_sft.py --mode train
```

默认会使用：

- **模型**：`vlm/models/Qwen2.5-VL-3B-Instruct`（当前为符号链接，指向机器上的真实模型目录）
- **训练集**：`pddl_data/data_vl_train_130.json`
- **输出目录**：`vlm/train/output_lora_refactor`
- **deepspeed 配置**：`vlm/train/ds_z2_offload_config.json`（ZeRO-2 + CPU optimizer offload + bf16）

常用的"快速 smoke-train"（只跑少量样本/步数）：

```bash
deepspeed --num_gpus 1 vlm/train/train_sft.py \
  --mode train \
  --max_train_samples 8 \
  --max_steps 5 \
  --save_steps 5
```

## 2. 单方法评测（base / sft / sft_yolo_prompt）

评测只看 `(:init ...)` 原子集合是否与 GT 完全一致（脚本会把结果写到 `output_dir`）。

### 2.1 评测基座（不加载 LoRA）

```bash
python -u vlm/train/train_sft.py --mode eval --eval_method base
```

### 2.2 评测 SFT（加载 LoRA adapter）

先确保你已经训练过，`output_dir` 下有 `checkpoint-*` 或 `final_model`。

```bash
python -u vlm/train/train_sft.py --mode eval --eval_method sft
```

如果你想显式指定 adapter：

```bash
python -u vlm/train/train_sft.py \
  --mode eval \
  --eval_method sft \
  --adapter_path /root/Plan/vlmpddl/vlm/train/output_lora_refactor/final_model
```

### 2.3 评测 SFT + YOLO 提示注入（不改模型，只把 YOLO 检测摘要拼进 prompt）

```bash
python -u vlm/train/train_sft.py --mode eval --eval_method sft_yolo_prompt
```

（可选）调 YOLO 阈值：

```bash
python -u vlm/train/train_sft.py \
  --mode eval \
  --eval_method sft_yolo_prompt \
  --yolo_conf 0.25
```

## 3. 一次性对比三种方法（base / sft / sft_yolo_prompt）

```bash
python -u vlm/train/train_sft.py --mode eval --eval_compare_all
```

会在 `output_dir` 写入：

- `eval_compare_init.json`
- `eval_compare_init.md`

## 4. 输出文件位置

默认输出目录：`/root/Plan/vlmpddl/vlm/train/output_lora_refactor`

- 训练：`checkpoint-*` 与 `final_model/`
- 评测：`eval_*.json`、`eval_compare_init.*`
