# YOLOv8 目标检测项目

基于 ultralytics YOLOv8 的目标检测项目，用于识别 regulator、battery、pump 等部件（含颜色细分）。
info 有 129 个，images 只有 104 个，info 中约 25 个没有对应图片。 (images 和 label 中没有sensor)

## 目录结构

```
yolo_project/
├── config/
│   └── data.yaml       # 数据集配置 (9 类: green/red/blue_regulator, green/red/blue_battery, green/red/blue_pump), 没有 sensor 这个类，我们最后 vlm2pddl 也不使用 sensor
├── data/
│   ├── images/         # 原始图片
│   ├── labels/         # YOLO 格式标注
│   └── info/           # 原始 txt 信息 (非 YOLO 格式)
├── main.py             # 统一运行脚本 (训练和推理)
├── evaluate_json.py    # 针对 JSON 测试集的评估脚本
├── requirements.txt    # 依赖库
└── README.md           # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

使用 YOLOv8s 预训练权重在 `data/labels` 标注数据上训练：

```bash
python main.py --mode train --epochs 100
```

训练完成后，模型权重保存在 `runs/train/custom_exp/weights/best.pt`。

### 3. 运行推理 (对指定图片做目标检测，并保存带框的可视化结果)

使用训练好的模型对图片进行识别：

```bash
python main.py --mode predict --source data/images
```

默认使用 `runs/train/custom_exp2/weights/best.pt`，也可指定其他权重：

```bash
python main.py --mode predict --weights runs/train/custom_exp2/weights/best.pt --source data/images
```

结果保存在 `runs/detect/inference` 文件夹中。

### 4. 对vlm_pddl测试集进行推理 (针对 VL 测试集的评估脚本)

# test 模式：从 test_4_scene 读取
python evaluate_pddl_recognize.py --mode test

# json 模式（默认）
python evaluate_pddl_recognize.py --mode json

# 指定模型
python evaluate_pddl_recognize.py --mode test --weights runs/train/custom_exp2/weights/best.pt