#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_image_key(sample: Dict) -> str:
    """从 conversations[0].value 中提取图片路径作为稳定键。"""
    conversations = sample.get("conversations", [])
    if not conversations:
        return ""
    user_value = conversations[0].get("value", "")
    match = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", user_value)
    if not match:
        return ""
    return Path(match.group(1)).name


def split_dataset(
    input_path: Path,
    train_path: Path,
    test_path: Path,
    train_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_required = train_size + test_size
    if len(data) < total_required:
        raise ValueError(
            f"数据量不足: 当前 {len(data)}，但需要至少 {total_required} 条。"
        )

    # 先稳定排序，再按固定随机种子洗牌，保证可复现。
    indexed = list(enumerate(data))
    indexed.sort(key=lambda x: (extract_image_key(x[1]), x[0]))

    rng = random.Random(seed)
    rng.shuffle(indexed)

    selected = indexed[:total_required]
    train_items = [item for _, item in selected[:train_size]]
    test_items = [item for _, item in selected[train_size: train_size + test_size]]

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_items, f, ensure_ascii=False, indent=2)

    return train_items, test_items


def build_log(train_items: List[Dict], test_items: List[Dict], seed: int) -> str:
    lines = [
        f"seed={seed}",
        f"train_count={len(train_items)}",
        f"test_count={len(test_items)}",
        "",
        "[train_examples]",
    ]
    for sample in train_items[:10]:
        lines.append(extract_image_key(sample))
    lines.append("")
    lines.append("[test_examples]")
    for sample in test_items[:10]:
        lines.append(extract_image_key(sample))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="固定拆分 data_vl_train.json 为 130/50")
    parser.add_argument(
        "--input",
        type=str,
        default="data_vl_train.json",
        help="输入 JSON 路径",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data_vl_train_130.json",
        help="训练集输出路径",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="data_vl_test_50.json",
        help="测试集输出路径",
    )
    parser.add_argument("--train_size", type=int, default=130, help="训练集大小")
    parser.add_argument("--test_size", type=int, default=50, help="测试集大小")
    parser.add_argument("--seed", type=int, default=20260224, help="随机种子")
    parser.add_argument(
        "--log_output",
        type=str,
        default="data_split_130_50.log",
        help="拆分日志路径",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    train_path = Path(args.train_output).resolve()
    test_path = Path(args.test_output).resolve()
    log_path = Path(args.log_output).resolve()

    train_items, test_items = split_dataset(
        input_path=input_path,
        train_path=train_path,
        test_path=test_path,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    log_text = build_log(train_items, test_items, args.seed)
    log_path.write_text(log_text, encoding="utf-8")

    print(f"拆分完成: train={len(train_items)}, test={len(test_items)}")
    print(f"train -> {train_path}")
    print(f"test  -> {test_path}")
    print(f"log   -> {log_path}")


if __name__ == "__main__":
    main()
