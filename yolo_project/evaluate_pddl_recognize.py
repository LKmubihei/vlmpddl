import argparse
import json
import re
import os
import shutil
from ultralytics import YOLO
from collections import Counter
from pathlib import Path

# 路径配置
_BASE_DIR = Path(__file__).resolve().parent
TEST_SCENE_DIR = _BASE_DIR.parent / "pddl_data" / "test_4_scene"
IMAGE_FALLBACKS = [
    _BASE_DIR / "data" / "images",
    _BASE_DIR.parent / "pddl_data" / "real_pictures",
]


def parse_pddl_objects(pddl_text):
    """
    Parses the (:objects ...) section from PDDL text and returns a count of parts.
    Mappings:
    - *pump* -> pump
    - *battery* -> battery
    - *regulator* -> regulator
    """
    match = re.search(r'\(:objects\s+(.*?)\)', pddl_text, re.DOTALL)
    if not match:
        return Counter()
    
    content = match.group(1)
    # Remove types like "- location" or "- part"
    content = re.sub(r'-\s+\w+', '', content)
    
    # Split by whitespace
    tokens = content.split()
    
    # Filter known locations (heuristic: usually pump_placement, etc. or just checking if it contains pump/battery/regulator)
    parts = []
    for token in tokens:
        token = token.lower()
        if 'pump' in token and 'placement' not in token:
            parts.append('pump')
        elif 'battery' in token and 'placement' not in token:
            parts.append('battery')
        elif 'regulator' in token and 'placement' not in token:
            parts.append('regulator')
            
    return Counter(parts)


def find_image_for_picture(picture_id, scene_dir=None):
    """查找 picture_id 对应的图片，优先 scene_dir，再尝试备用路径"""
    possible_names = [
        f"picture_{picture_id}.jpg",
        f"picture_{picture_id}.png",
        f"{picture_id}.jpg",
        f"{picture_id}.png",
    ]
    search_dirs = []
    if scene_dir and Path(scene_dir).exists():
        search_dirs.append(Path(scene_dir))
    search_dirs.extend(IMAGE_FALLBACKS)

    for folder in search_dirs:
        if not folder.exists():
            continue
        for name in possible_names:
            img_path = folder / name
            if img_path.exists():
                return str(img_path)
    return None


def load_test_data():
    """
    从 test_4_scene 加载测试数据。
    遍历 normal, stack, redundance, fault_placement 四个场景，
    每个场景下读取 .nl 指令和 true/*.pddl 作为 ground truth。
    返回 [(img_path, instruction, gt_pddl_path), ...]
    """
    if not TEST_SCENE_DIR.exists():
        raise FileNotFoundError(f"Test scene directory not found: {TEST_SCENE_DIR}")

    scene_folders = ["normal", "stack", "redundance", "fault_placement"]
    samples = []

    for scene in scene_folders:
        scene_dir = TEST_SCENE_DIR / scene
        if not scene_dir.is_dir():
            continue

        # 遍历该场景下的 .nl 文件
        for nl_file in sorted(scene_dir.glob("picture_*.nl")):
            match = re.search(r"picture_(\d+)", nl_file.stem)
            if not match:
                continue
            picture_id = match.group(1)

            # 读取指令
            with open(nl_file, "r", encoding="utf-8") as f:
                instruction = f.read().strip()

            # Ground truth PDDL
            gt_pddl_path = scene_dir / "true" / f"picture_{picture_id}.pddl"
            if not gt_pddl_path.exists():
                continue

            # 查找图片：优先场景目录，再备用路径
            img_path = find_image_for_picture(picture_id, scene_dir)
            if not img_path:
                print(f"警告: 找不到图片 picture_{picture_id} (场景: {scene})，跳过")
                continue

            samples.append(
                {
                    "img_path": img_path,
                    "instruction": instruction,
                    "gt_pddl_path": str(gt_pddl_path),
                    "scene": scene,
                    "picture_id": picture_id,
                }
            )

    return samples


def run_evaluation(model, yolo_to_type, samples, output_dir):
    """通用评估逻辑：对 samples 进行推理并比较"""
    total_samples = 0
    perfect_matches = 0

    print(f"{'Image':<35} | {'Scene':<12} | {'Ground Truth':<20} | {'Prediction':<20} | {'Match'}")
    print("-" * 120)

    for item in samples:
        img_path = item["img_path"]
        scene = item.get("scene", "")

        # Ground Truth: 优先用 gt_counts，否则从 gt_pddl_path 读取
        if "gt_counts" in item:
            gt_counts = item["gt_counts"]
        else:
            with open(item["gt_pddl_path"], "r", encoding="utf-8") as f:
                gt_counts = parse_pddl_objects(f.read())

        # 推理：按 scene 分目录保存，避免同名图片（如 picture_1 在 4 个场景中）互相覆盖
        save_name = f"eval_vis/{scene}" if scene else "eval_vis"
        results = model.predict(
            img_path, save=True, project="runs", name=save_name, exist_ok=True, verbose=False
        )

        pred_counts = Counter()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = yolo_to_type.get(cls_id, "unknown")
                pred_counts[cls_name] += 1

        is_match = gt_counts == pred_counts
        total_samples += 1
        if is_match:
            perfect_matches += 1

        gt_str = f"P:{gt_counts['pump']} B:{gt_counts['battery']} R:{gt_counts['regulator']}"
        pred_str = f"P:{pred_counts['pump']} B:{pred_counts['battery']} R:{pred_counts['regulator']}"
        match_str = "✅" if is_match else "❌"
        img_name = os.path.basename(img_path)
        print(f"{img_name:<35} | {scene:<12} | {gt_str:<20} | {pred_str:<20} | {match_str}")

    print("-" * 120)
    print(f"Total Samples: {total_samples}")
    print(f"Perfect Matches (Count Exact): {perfect_matches}")
    if total_samples > 0:
        print(f"Accuracy: {perfect_matches / total_samples:.2%}")
    print(f"\nVisualization images saved to: {output_dir}")
    if any(s.get("scene") for s in samples):
        print("  (test mode: images organized by scene: normal/, stack/, redundance/, fault_placement/)")


def load_json_data(json_path):
    """从 JSON 文件加载评估数据"""
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue

        user_msg = convs[0]["value"]
        img_match = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", user_msg)
        if not img_match:
            continue

        img_path = img_match.group(1)
        if not os.path.exists(img_path):
            filename = os.path.basename(img_path)
            for fallback in IMAGE_FALLBACKS:
                candidate = fallback / filename
                if candidate.exists():
                    img_path = str(candidate)
                    break
            else:
                print(f"Image not found: {img_path}")
                continue

        assistant_msg = convs[1]["value"]
        gt_counts = parse_pddl_objects(assistant_msg)
        samples.append({"img_path": img_path, "gt_counts": gt_counts, "scene": ""})
    return samples


def main():
    parser = argparse.ArgumentParser(description="YOLO PDDL 识别评估")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["json", "test"],
        default="json",
        help="json: 从 JSON 文件读取; test: 从 test_4_scene 读取图片和指令",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=str(_BASE_DIR.parent / "pddl_data" / "data_vl_train.json"),
        help="JSON 模式下的数据文件路径",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/custom_exp2/weights/best.pt",
        help="模型权重路径",
    )
    args = parser.parse_args()

    model_path = args.weights
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 {model_path}，请先运行 python main.py --mode train 进行训练")
        return

    print(f"Mode: {args.mode}")
    print(f"Using model: {model_path}")

    output_dir = "runs/eval_vis"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model = YOLO(model_path)
    yolo_to_type = {
        0: "regulator", 1: "regulator", 8: "regulator",
        2: "battery", 6: "battery", 7: "battery",
        3: "pump", 4: "pump", 5: "pump",
    }

    if args.mode == "test":
        print(f"Loading test data from: {TEST_SCENE_DIR}")
        samples = load_test_data()
        print(f"Loaded {len(samples)} samples from test_4_scene")
    else:
        print(f"Loading JSON data from: {args.json}")
        samples = load_json_data(args.json)
        print(f"Loaded {len(samples)} samples from JSON")

    if not samples:
        print("No samples to evaluate.")
        return

    run_evaluation(model, yolo_to_type, samples, output_dir)


if __name__ == "__main__":
    main()

