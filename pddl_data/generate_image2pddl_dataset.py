#!/usr/bin/env python3
"""
数据集生成脚本
从 pddl_y_valid 中提取 goal，转换成自然语言指令，并生成训练数据集
"""

import os
import re
import json
from pathlib import Path


# 路径配置（相对于脚本所在目录）
_BASE_DIR = Path(__file__).resolve().parent
PDDL_FOLDER = _BASE_DIR / "pddl_y_valid"
IMAGE_FOLDER = _BASE_DIR / "real_pictures"
INSTRUCT_FOLDER = _BASE_DIR / "instructions"
OUTPUT_JSON = _BASE_DIR / "data_vl_train.json"

# 创建指令文件夹
os.makedirs(INSTRUCT_FOLDER, exist_ok=True)

# Domain 定义文本（仅 predicates，不含 actions，并解释各谓词含义）
DOMAIN_TEXT = """Based on the given domain and predicate definitions:

(define (domain ariac)
  (:requirements :strips :typing :adl)
  (:types
    part location - object
  )

  (:predicates
    (robot_at ?l - location)      ; robot is at location l
    (part_at ?p - part ?l - location)  ; part p is at location l
    (on ?top - part ?bottom - object)  ; part top is stacked on object bottom
    (clear ?x - object)           ; object x has nothing on top, can be picked or stacked on
    (holding ?p - part)           ; robot is holding part p
    (handempty)                   ; robot is not holding anything
  )
)

, and the image:

Analyze the objects and their corresponding states in the image.

"""

FINAL_INSTRUCTION = "If there are the same parts, add the suffix '_1' to the name of the second part. Please output EXACTLY three PDDL sections in order: (:objects ...), (:init ...), (:goal ...). Do not include any description or extra text."


def extract_goal_from_pddl(pddl_file):
    """从 PDDL 文件中提取 goal 部分，使用括号匹配处理嵌套"""
    with open(pddl_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到 (:goal 的位置
    goal_start = content.find('(:goal')
    if goal_start == -1:
        return []
    
    # 从 (:goal 开始，使用括号计数找到匹配的结束位置
    depth = 0
    goal_end = -1
    for i in range(goal_start, len(content)):
        if content[i] == '(':
            depth += 1
        elif content[i] == ')':
            depth -= 1
            if depth == 0:
                goal_end = i + 1
                break
    
    if goal_end == -1:
        return []
    
    goal_content = content[goal_start:goal_end]
    
    # 提取所有的 part_at 语句
    part_at_pattern = r'\(part_at\s+(\w+)\s+(\w+)\)'
    matches = re.findall(part_at_pattern, goal_content)
    
    return matches  # 返回 [(part, placement), ...]


def convert_goal_to_instruction(goal_parts):
    """
    将 goal 转换成自然语言指令
    注意: 所有零件最后都是 "to the ventilator"
    如果 goal 为空，返回空字符串
    """
    if not goal_parts:
        return ""
    
    instructions = []
    for part, placement in goal_parts:
        # 所有零件都统一使用 ventilator
        instruction = f"assembly {part} to the ventilator"
        instructions.append(instruction)
    
    # 如果有多个目标，用逗号连接
    if len(instructions) == 1:
        return instructions[0]
    elif len(instructions) == 2:
        return f"{instructions[0]} and {instructions[1]}"
    else:
        return ", ".join(instructions[:-1]) + f", and {instructions[-1]}"


def extract_three_pddl_sections(pddl_text):
    """
    从完整的 PDDL 文本中提取 (:objects) (:init) (:goal) 三段
    使用括号匹配来精确提取每一段
    """
    def find_section(name: str, text: str) -> str:
        """使用括号匹配找到完整的 section"""
        pattern = rf"\(:{name}\b"
        m = re.search(pattern, text)
        if not m:
            return ""
        
        start = m.start()
        depth = 0
        end = None
        
        for i in range(start, len(text)):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        return text[start:end].strip() if end else ""
    
    objects_part = find_section("objects", pddl_text)
    init_part = find_section("init", pddl_text)
    goal_part = find_section("goal", pddl_text)
    
    # 组合三段，用两个换行符分隔
    parts = [p for p in [objects_part, init_part, goal_part] if p]
    return "\n\n".join(parts)


def find_image_file(picture_id):
    """查找对应的图片文件，仅在 real_pictures 中查找"""
    possible_names = [
        f"picture_{picture_id}.jpg",
        f"picture_{picture_id}.png",
        f"{picture_id}.jpg",
        f"{picture_id}.png"
    ]
    for name in possible_names:
        image_path = IMAGE_FOLDER / name
        if image_path.exists():
            return str(image_path)
    return None


def generate_dataset():
    """生成完整的数据集"""
    dataset = []
    
    # 遍历所有 pddl 文件
    pddl_files = sorted([f for f in os.listdir(PDDL_FOLDER) if f.endswith('.pddl')])
    
    print(f"找到 {len(pddl_files)} 个 PDDL 文件")
    
    for pddl_file in pddl_files:
        pddl_path = PDDL_FOLDER / pddl_file
        
        # 提取 picture_id
        match = re.search(r'picture_(\d+)', pddl_file)
        if not match:
            print(f"警告: 无法从文件名提取 picture_id: {pddl_file}")
            continue
        
        picture_id = match.group(1)
        
        # 查找对应的图片文件
        image_path = find_image_file(picture_id)
        if not image_path:
            print(f"警告: 找不到图片文件 for picture_{picture_id}")
            continue
        
        # 提取 goal
        try:
            goal_parts = extract_goal_from_pddl(pddl_path)
            
            # 转换成自然语言指令（可以为空）
            instruction = convert_goal_to_instruction(goal_parts)
            
            # 如果 goal 为空，也记录下来
            if not goal_parts:
                print(f"注意: {pddl_file} 的 goal 为空，指令也为空")
            
            # 保存指令到文件
            instruct_file = INSTRUCT_FOLDER / f"picture_{picture_id}.nl"
            with open(instruct_file, 'w', encoding='utf-8') as f:
                f.write(instruction)
            
            # 读取完整的 pddl 内容作为标签
            with open(pddl_path, 'r', encoding='utf-8') as f:
                pddl_content = f.read().strip()
            
            # 提取三段 PDDL 内容作为输出
            output_pddl = extract_three_pddl_sections(pddl_content)
            
            # 构造 conversations 格式
            # 构建输入：图片路径 + domain 定义 + 指令
            if instruction:
                goal_text = f"Goal: {instruction}."
            else:
                goal_text = "Goal: Analyze the current state."
            
            input_value = f"<|vision_start|>{image_path}<|vision_end|>{DOMAIN_TEXT}{goal_text}{FINAL_INSTRUCTION}"
            
            # 构造数据集条目（conversations 格式）
            dataset.append({
                "conversations": [
                    {
                        "value": input_value
                    },
                    {
                        "value": output_pddl
                    }
                ]
            })
            
            if instruction:
                print(f"✓ 处理成功: picture_{picture_id} - 指令: {instruction}")
            else:
                print(f"✓ 处理成功: picture_{picture_id} - 指令为空")
            
        except Exception as e:
            print(f"错误: 处理 {pddl_file} 时出错: {e}")
            continue
    
    # 保存数据集
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*60)
    print(f"数据集生成完成！")
    print(f"="*60)
    print(f"总共处理了 {len(dataset)} 个样本")
    print(f"数据集保存到: {OUTPUT_JSON.resolve()}")
    print(f"指令文件保存到: {INSTRUCT_FOLDER.resolve()}")
    print(f"数据格式: conversations (兼容 train_sft.py)")
    print(f"="*60)
    
    return dataset


if __name__ == "__main__":
    dataset = generate_dataset()
    
    # 显示前3个样本
    print("\n=== 前3个样本示例 ===")
    for i, sample in enumerate(dataset[:3]):
        print(f"\n样本 {i+1}:")
        # 提取图片路径
        input_value = sample['conversations'][0]['value']
        image_path_match = re.search(r'<\|vision_start\|>(.*?)<\|vision_end\|>', input_value)
        if image_path_match:
            image_path = image_path_match.group(1)
            print(f"  图片: {image_path}")
        
        # 提取指令
        goal_match = re.search(r'Goal: (.*?)\. Please', input_value)
        if goal_match:
            instruction = goal_match.group(1)
            print(f"  指令: {instruction}")
        
        # 输出内容预览
        output_value = sample['conversations'][1]['value']
        print(f"  输出预览: {output_value[:100]}...")

