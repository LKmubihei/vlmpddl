import argparse
import os
from pathlib import Path
import sys

# Try importing ultralytics, provide help if missing
try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 未安装 'ultralytics' 库。")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

def train_model(data_yaml, epochs=100, img_size=640):
    """
    训练模型 (需要有带标注的 labels 文件夹)
    使用 YOLOv8 预训练权重，训练完成后权重保存在 runs/train/custom_exp/weights/
    """
    print(f"开始训练: Data={data_yaml}, Epochs={epochs}")
    # 加载 YOLOv8 预训练模型 (yolov8s.pt 会自动下载)
    model = YOLO('yolov8s.pt')

    # 开始训练
    # 注意: 这里的 project 和 name 参数决定了结果保存的位置
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        project='runs/train',
        name='custom_exp'
    )
    print("训练完成。结果保存在 runs/train/custom_exp")

def run_inference(model_path, source, conf=0.25):
    """
    使用训练好的模型进行推理
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"开始推理: Source={source}")
    # save=True 会将结果图片保存到 runs/detect/...
    results = model.predict(
        source=source, 
        conf=conf, 
        save=True,
        project='runs/detect',
        name='inference'
    )
    print(f"推理完成。查看 runs/detect/inference")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Project Manager")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help='运行模式: train 或 predict')
    parser.add_argument('--weights', type=str, default='runs/train/custom_exp2/weights/best.pt', help='模型权重路径 (训练后生成)')
    parser.add_argument('--source', type=str, default='data/images', help='预测图片来源 (文件夹或文件)')
    parser.add_argument('--data', type=str, default='config/data.yaml', help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args.data, args.epochs)
    elif args.mode == 'predict':
        run_inference(args.weights, args.source)

