#!/usr/bin/env python3
import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:
    process_vision_info = None


def _fallback_process_vision_info(messages: List[Dict[str, Any]]):
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError(
            "缺少 qwen_vl_utils 且无法导入 Pillow (PIL)，无法处理图片输入。"
        ) from e

    image_inputs = []
    video_inputs = []
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") == "image":
                img = item.get("image")
                if isinstance(img, str):
                    image_inputs.append(Image.open(img).convert("RGB"))
                else:
                    image_inputs.append(img)
            elif item.get("type") == "video":
                video_inputs.append(item.get("video"))
    return image_inputs, video_inputs


def process_vision(messages: List[Dict[str, Any]]):
    if process_vision_info is not None:
        return process_vision_info(messages)
    return _fallback_process_vision_info(messages)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_sft_refactor")

_VISION_RE = re.compile(r"<\|vision_start\|>(.*?)<\|vision_end\|>")
_PDDL_SECTION_START_RE = re.compile(r"\(:([a-zA-Z0-9_-]+)\b", re.IGNORECASE)

PART_TYPES = ("pump", "battery", "regulator")
PART_TOKEN_CANON_RE = re.compile(r"^(pump|battery|regulator)_(\d+)$", re.IGNORECASE)
PLACEMENT_HINT_RE = re.compile(r"placement", re.IGNORECASE)

YOLO_TO_TYPE = {
    0: "regulator",
    1: "regulator",
    8: "regulator",
    2: "battery",
    6: "battery",
    7: "battery",
    3: "pump",
    4: "pump",
    5: "pump",
}


@dataclass
class Config:
    # 运行模式
    mode: str

    # 路径配置
    model_path: str
    train_data_path: str
    test_data_path: str
    output_dir: str
    deepspeed_config: Optional[str]

    # 训练参数
    max_length: int
    num_epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    max_train_samples: int
    max_steps: int

    # 评测参数
    eval_method: str  # base|sft|sft_yolo_prompt
    adapter_path: Optional[str]
    max_eval_samples: int
    yolo_weights: Optional[str]
    yolo_conf: float
    max_new_tokens: int
    with_predicate_metrics: bool
    eval_compare_all: bool
    seed: int
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-3B-VL: SFT and 3-method eval")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--model_path", default="/root/Plan/vlmpddl/vlm/models/Qwen2.5-VL-3B-Instruct", help="基础VLM模型路径")
    parser.add_argument("--train_data_path", default="/root/Plan/vlmpddl/pddl_data/data_vl_train_130.json", help="训练集JSON")
    parser.add_argument("--test_data_path", default="/root/Plan/vlmpddl/pddl_data/data_vl_test_50.json", help="测试集JSON（默认50条）")
    parser.add_argument("--output_dir", default="/root/Plan/vlmpddl/vlm/train/output_lora_refactor", help="训练输出/评测输出目录")
    parser.add_argument("--deepspeed_config", default="/root/Plan/vlmpddl/vlm/train/ds_z2_offload_config.json", help="deepspeed配置（可为空）")

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=-1, help="训练样本数上限；-1 表示全量（用于快速 smoke-train 可设很小）")
    parser.add_argument("--max_steps", type=int, default=-1, help="训练步数上限；-1 表示按 epoch 跑全量")

    parser.add_argument("--eval_method", choices=["base", "sft", "sft_yolo_prompt"], default="base", help="评测方法：base=基座；sft=加载LoRA；sft_yolo_prompt=LoRA+YOLO提示注入")
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter目录；不填则自动找 output_dir 下 checkpoint-*/final_model")
    parser.add_argument("--max_eval_samples", type=int, default=50, help="评测样本数上限（默认50）")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="推理生成token上限")
    parser.add_argument("--with_predicate_metrics", action="store_true", help="额外输出每个谓词的precision/recall（用于定位误差）")
    parser.add_argument("--eval_compare_all", action="store_true", help="一次性评测 base/sft/sft_yolo_prompt，并输出对比文件")

    parser.add_argument("--yolo_weights", default="/root/Plan/vlmpddl/yolo_project/runs/train/custom_exp2/weights/best.pt", help="YOLO权重路径（仅 sft_yolo_prompt 需要）")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260224)
    args, _ = parser.parse_known_args()
    return args


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        mode=args.mode,
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        deepspeed_config=args.deepspeed_config,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_train_samples=args.max_train_samples,
        max_steps=args.max_steps,
        eval_method=args.eval_method,
        adapter_path=args.adapter_path,
        max_eval_samples=args.max_eval_samples,
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        max_new_tokens=args.max_new_tokens,
        with_predicate_metrics=bool(args.with_predicate_metrics),
        eval_compare_all=bool(args.eval_compare_all),
        seed=args.seed,
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_image_and_prompt(input_value: str) -> Tuple[str, str]:
    match = _VISION_RE.search(input_value)
    if not match:
        raise ValueError("input has no vision markers")
    image_path = match.group(1).strip()
    prompt_text = input_value.replace(match.group(0), "").strip()
    return image_path, prompt_text


def build_messages(image_path: str, prompt_text: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def process_train_sample(example: Dict[str, Any], processor, tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    input_value = example["conversations"][0]["value"]
    output_value = example["conversations"][1]["value"]
    image_path, prompt_text = extract_image_and_prompt(input_value)

    messages = build_messages(image_path, prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    prompt_input_ids = inputs["input_ids"][0].tolist()
    prompt_attention_mask = inputs["attention_mask"][0].tolist()

    response = tokenizer(output_value, add_special_tokens=False)
    input_ids = prompt_input_ids + response["input_ids"]
    attention_mask = prompt_attention_mask + response["attention_mask"]
    labels = [-100] * len(prompt_input_ids) + response["input_ids"]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"][0],
    }


def vlm_data_collator(tokenizer):
    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        def to_1d_long(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.to(dtype=torch.long).view(-1)
            return torch.tensor(x, dtype=torch.long).view(-1)

        def to_tensor(x: Any, *, dtype: torch.dtype) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype)
            return torch.tensor(x, dtype=dtype)

        input_ids_list = [to_1d_long(f["input_ids"]) for f in features]
        attn_list = [to_1d_long(f["attention_mask"]) for f in features]
        labels_list = [to_1d_long(f["labels"]) for f in features]

        max_len = max(int(x.numel()) for x in input_ids_list) if input_ids_list else 0
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        def pad_1d(x: torch.Tensor, *, pad_value: int) -> torch.Tensor:
            if x.numel() >= max_len:
                return x[:max_len]
            pad = torch.full((max_len - int(x.numel()),), pad_value, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(x, pad_value=int(pad_id)) for x in input_ids_list], dim=0)
        attention_mask = torch.stack([pad_1d(x, pad_value=0) for x in attn_list], dim=0)
        labels = torch.stack([pad_1d(x, pad_value=-100) for x in labels_list], dim=0)

        # pixel_values: (bs, 3, H, W)；image_grid_thw: (bs, 3)
        # 使用 bfloat16 与模型精度保持一致，避免 DeepSpeed bf16 模式下的 dtype 不匹配
        pv_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pixel_values = torch.cat([to_tensor(f["pixel_values"], dtype=pv_dtype) for f in features], dim=0)
        image_grid_thw = torch.stack([to_tensor(f["image_grid_thw"], dtype=torch.long) for f in features], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    return collate


def latest_checkpoint_or_final(output_dir: str) -> Optional[str]:
    path = Path(output_dir)
    if not path.exists():
        return None
    checkpoints = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda p: int(p.name.split("-")[1]))
        return str(checkpoints[-1])
    final_model = path / "final_model"
    return str(final_model) if final_model.exists() else None


def load_base_model(model_path: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer, processor


def train_sft(cfg: Config) -> None:
    seed_everything(cfg.seed)
    model, tokenizer, processor = load_base_model(cfg.model_path)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    # 不手动 .to(device)，让 HuggingFace Trainer + DeepSpeed 统一管理设备放置

    train_raw = load_json_list(cfg.train_data_path)
    if cfg.max_train_samples is not None and cfg.max_train_samples > 0:
        train_raw = train_raw[: cfg.max_train_samples]
    train_ds = Dataset.from_list(train_raw)
    train_dataset = train_ds.map(
        lambda x: process_train_sample(x, processor, tokenizer, cfg.max_length),
        remove_columns=train_ds.column_names,
    )

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        bf16=use_cuda,
        fp16=False,
        report_to="none",
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        deepspeed=cfg.deepspeed_config if cfg.deepspeed_config else None,
        seed=cfg.seed,
    )
    if cfg.max_steps is not None and cfg.max_steps > 0:
        training_args.max_steps = cfg.max_steps

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vlm_data_collator(tokenizer),
    )
    trainer.train()
    final_dir = Path(cfg.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    logger.info("saved final adapter/model: %s", final_dir)

@torch.inference_mode()
def predict_one(model, processor, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    outputs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs[0] if outputs else ""


def extract_pddl_section(text: str, section: str) -> str:
    """
    从任意文本中提取形如 (:init ... ) 的完整S表达式片段（括号配平）。
    找不到则返回空字符串。
    """
    if not text:
        return ""
    start = None
    for m in _PDDL_SECTION_START_RE.finditer(text):
        if m.group(1).lower() == section.lower():
            start = m.start()
            break
    if start is None:
        return ""

    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return ""


def _tokenize_sexpr(s: str) -> List[str]:
    return re.findall(r"\(|\)|[^\s()]+", s)


def _parse_sexpr(tokens: Sequence[str]) -> Any:
    stack: List[List[Any]] = []
    cur: List[Any] = []
    for tok in tokens:
        if tok == "(":
            stack.append(cur)
            cur = []
        elif tok == ")":
            if not stack:
                continue
            completed = cur
            cur = stack.pop()
            cur.append(completed)
        else:
            cur.append(tok)
    if len(cur) == 1 and isinstance(cur[0], list):
        return cur[0]
    return cur


def _clean_symbol(sym: str) -> str:
    sym = sym.strip().strip(",;")
    if sym.startswith('"') and sym.endswith('"') and len(sym) >= 2:
        sym = sym[1:-1]
    return sym.lower()


def normalize_part_token(token: str) -> str:
    """
    标准化 token：
    - 谓词名：由调用方直接 lower
    - 参数：仅对零件名 token 做后缀归一化：pump_数字/battery_数字/regulator_数字 -> pump/battery/regulator
      同时避免对位置名（如 pump_placement / pump_placement_1）误处理。
    """
    t = _clean_symbol(token)
    if PLACEMENT_HINT_RE.search(t):
        return t
    m = PART_TOKEN_CANON_RE.match(t)
    if m:
        return m.group(1).lower()
    return t


def iter_init_atoms(pddl_text: str) -> Iterable[Tuple[str, ...]]:
    """
    解析 (:init ...) 中的原子谓词（只取 init 的直接子表达式）。
    对每个原子做标准化：
    - 谓词名保留（lower）
    - 参数中属于零件名的 token 做后缀归一化
    """
    init_section = extract_pddl_section(pddl_text, "init")
    if not init_section:
        return []

    parsed = _parse_sexpr(_tokenize_sexpr(init_section))
    if not (isinstance(parsed, list) and parsed):
        return []

    head = parsed[0]
    if not isinstance(head, str) or _clean_symbol(head) != ":init":
        return []

    atoms: List[Tuple[str, ...]] = []
    for item in parsed[1:]:
        if not isinstance(item, list) or not item:
            continue
        # 处理 (not (pred ...))
        if isinstance(item[0], str) and _clean_symbol(item[0]) == "not" and len(item) == 2 and isinstance(item[1], list):
            item = item[1]
        if not item or not isinstance(item[0], str):
            continue
        pred = _clean_symbol(item[0])
        if pred.startswith(":") or pred in {"=", "and", "or"}:
            continue
        args: List[str] = []
        ok = True
        for a in item[1:]:
            if isinstance(a, list):
                ok = False
                break
            if not isinstance(a, str):
                ok = False
                break
            args.append(normalize_part_token(a))
        if not ok:
            continue
        atoms.append(tuple([pred, *args]))
    return atoms


def parse_init_atoms_set(pddl_text: str) -> set:
    return set(iter_init_atoms(pddl_text))


def init_exact_match(pred_text: str, gt_text: str) -> int:
    return int(parse_init_atoms_set(pred_text) == parse_init_atoms_set(gt_text))


def accumulate_predicate_counts(
    pred_atoms: Iterable[Tuple[str, ...]],
    gt_atoms: Iterable[Tuple[str, ...]],
    out_counts: Dict[str, Counter],
) -> None:
    """
    以谓词名为粒度累计 TP/FP/FN（micro）。
    out_counts[pred] 里会写入 {"tp":..,"fp":..,"fn":..,"pred":..,"gt":..}
    """
    pred_set = set(pred_atoms)
    gt_set = set(gt_atoms)

    pred_by = defaultdict(set)
    gt_by = defaultdict(set)
    for a in pred_set:
        pred_by[a[0]].add(a)
    for a in gt_set:
        gt_by[a[0]].add(a)

    for name in set(pred_by.keys()) | set(gt_by.keys()):
        p = pred_by[name]
        g = gt_by[name]
        tp = len(p & g)
        fp = len(p - g)
        fn = len(g - p)
        out_counts[name].update({"tp": tp, "fp": fp, "fn": fn, "pred": len(p), "gt": len(g)})


def build_yolo_hint(yolo_model, image_path: str, conf: float) -> str:
    """
    将 YOLO 检测结果摘要拼接到原始 user prompt 尾部（不替换图片与原始文本）。
    输出包含：零件列表 + 计数（pump/battery/regulator）。
    """
    counts = Counter({k: 0 for k in PART_TYPES})
    parts_list: List[str] = []
    results = yolo_model.predict(image_path, conf=conf, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = YOLO_TO_TYPE.get(cls_id)
            if cls_name in PART_TYPES:
                counts[cls_name] += 1
                parts_list.append(cls_name)
    parts_list_str = ", ".join(parts_list) if parts_list else "(empty)"
    return (
        "\n\n[YOLO辅助提示]\n"
        f"- 检测到的零件计数: pump={counts['pump']}, battery={counts['battery']}, regulator={counts['regulator']}\n"
        f"- 检测到的零件列表: {parts_list_str}\n"
        "请将其作为辅助线索（可能有误），并仍以图像与原始指令为主生成 PDDL。"
    )


def load_eval_model(cfg: Config):
    model, _, processor = load_base_model(cfg.model_path)
    model = model.to(cfg.device)
    if cfg.eval_method == "base":
        model.eval()
        return model, processor
    adapter_path = cfg.adapter_path or latest_checkpoint_or_final(cfg.output_dir)
    if not adapter_path:
        raise FileNotFoundError("SFT adapter not found, use --adapter_path or train first")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor


def evaluate_once(cfg: Config) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    test_data = load_json_list(cfg.test_data_path)[: cfg.max_eval_samples]
    if not test_data:
        raise ValueError("empty test set")

    model, processor = load_eval_model(cfg)

    yolo_model = None
    if cfg.eval_method == "sft_yolo_prompt":
        if YOLO is None:
            raise ImportError("ultralytics not installed (pip install ultralytics)")
        if not cfg.yolo_weights or not Path(cfg.yolo_weights).exists():
            raise FileNotFoundError(f"YOLO weights missing: {cfg.yolo_weights}")
        yolo_model = YOLO(cfg.yolo_weights)

    records: List[Dict[str, Any]] = []
    exact_sum = 0
    predicate_counts: Dict[str, Counter] = defaultdict(Counter)

    for i, sample in enumerate(test_data, start=1):
        input_value = sample["conversations"][0]["value"]
        gt_output = sample["conversations"][1]["value"]
        image_path, prompt_text = extract_image_and_prompt(input_value)

        if cfg.eval_method == "sft_yolo_prompt":
            prompt_text = prompt_text + build_yolo_hint(yolo_model, image_path, cfg.yolo_conf)

        pred_output = predict_one(
            model,
            processor,
            build_messages(image_path, prompt_text),
            max_new_tokens=cfg.max_new_tokens,
        )

        pred_atoms = parse_init_atoms_set(pred_output)
        gt_atoms = parse_init_atoms_set(gt_output)
        exact = int(pred_atoms == gt_atoms)
        exact_sum += exact

        if cfg.with_predicate_metrics:
            accumulate_predicate_counts(pred_atoms, gt_atoms, predicate_counts)

        records.append(
            {
                "index": i,
                "image_path": image_path,
                "init_exact_match": exact,
                "prediction": pred_output,
                "ground_truth": gt_output,
                "pred_init_atoms_count": len(pred_atoms),
                "gt_init_atoms_count": len(gt_atoms),
            }
        )
        if i % 5 == 0 or i == len(test_data):
            logger.info("eval progress (%s): %d/%d", cfg.eval_method, i, len(test_data))

    summary: Dict[str, Any] = {
        "method": cfg.eval_method,
        "samples": len(test_data),
        "init_exact_match_acc": exact_sum / len(test_data),
    }

    predicate_summary: Dict[str, Any] = {}
    if cfg.with_predicate_metrics:
        for name in sorted(predicate_counts.keys()):
            c = predicate_counts[name]
            tp = float(c.get("tp", 0))
            fp = float(c.get("fp", 0))
            fn = float(c.get("fn", 0))
            predicate_summary[name] = {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": (tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                "recall": (tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            }

    payload = {
        "summary": summary,
        "predicate_summary": predicate_summary,
        "records": records,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{cfg.eval_method}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("saved eval result: %s", out_file)
    return payload


def evaluate_compare_all(cfg: Config) -> Dict[str, Any]:
    methods = ["base", "sft", "sft_yolo_prompt"]
    results: Dict[str, Any] = {}
    for m in methods:
        cfg_m = Config(**{**cfg.__dict__, "eval_method": m, "eval_compare_all": False})
        results[m] = evaluate_once(cfg_m)["summary"]

    compare = {
        "test_data_path": cfg.test_data_path,
        "samples": cfg.max_eval_samples,
        "results": results,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eval_compare_init.json"
    out_md = out_dir / "eval_compare_init.md"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("## init 一致性评测对比（同一测试集）\n\n")
        f.write(f"- test_data_path: {cfg.test_data_path}\n")
        f.write(f"- samples: {cfg.max_eval_samples}\n\n")
        f.write("| method | init_exact_match_acc |\n")
        f.write("|---|---:|\n")
        for m in methods:
            acc = results[m].get("init_exact_match_acc", 0.0)
            f.write(f"| {m} | {acc:.4f} |\n")
    logger.info("saved compare results: %s , %s", out_json, out_md)
    return compare


def main() -> None:
    cfg = build_config(parse_args())
    if cfg.mode == "train":
        train_sft(cfg)
    else:
        if cfg.eval_compare_all:
            evaluate_compare_all(cfg)
        else:
            evaluate_once(cfg)


if __name__ == "__main__":
    main()
