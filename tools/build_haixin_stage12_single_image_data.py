#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """输入是渲染/转码后的文档图片，仅用于受控结构化抽取；请只返回字段值，不输出额外信息。
你是严谨的医疗病历图片结构化抽取助手。
只根据输入图片中清晰可见的信息抽取字段值；不要根据医学常识、字段名或上下文猜测。
没有明确证据时 value 必须为 null。
如果图片与某个 tag 无关，value 必须为 null。
返回必须是严格 JSON，不要输出 Markdown。
"""


PRESENCE_PROMPT_TEMPLATE = """请判断这张图片中是否存在下面每个 tag 对应的明确取值。

要求：
1. 每个输入 tag 都必须在返回 JSON 的 items 中出现一次。
2. 这里只判断有没有值，不要抽取具体值。
3. 只有图片中能看到与该 tag 明确对应的信息时，has_value 才能为 true。
4. 如果图片与某个 tag 无关，或没有明确证据，has_value 必须为 false。
5. 不要补充输入列表以外的 tag。
6. 每个 tag 的 table_name_cn 和 description 只用于理解字段含义，不能当作图片证据。

返回 JSON 格式：
{{
  "items": [
    {{
      "tag_id": "原始tag_id",
      "tag_name": "原始tag_name",
      "has_value": false
    }}
  ]
}}

本次需要判断的 tags：
{tags_json}
"""


EXTRACTION_PROMPT_TEMPLATE = """请从这张图片中判断并抽取下面单个 tag 的值。

要求：
1. 返回 JSON 的 items 中必须出现这个 tag。
2. 先再次判断图片中是否有这个 tag 的明确取值，写入 has_value。
3. has_value 为 false 时，value 必须为 null。
4. has_value 为 true 时，value 写图片中的明确取值。
5. 不要根据医学常识、字段名或上下文猜测。
6. 注意同一段原文可能同时对应多个 tag。遇到组合值时要在相关字段里分别填写拆分值，同时在整体字段里保留原文整体值。
   例如图片原文是“pTNM分期：pT2N0Mx”，如果当前 tag 是 t_stage，则 value 应为 "pT2"；
   如果当前 tag 是 n_stage，则 value 应为 "N0"；如果当前 tag 是 overall_stage，则 value 应为 "pT2N0Mx"。
7. 如果当前 tag 对应表格明细中的一行或多行项目，除非 description 明确指定返回字段，否则 value 必须按图片表头原文返回对象或对象数组。
   - 对象 key 使用图片中对应表格的列名原文，不要改写成固定字段名。
   - key 的顺序必须与表头从左到右一致；多行对象必须按表格从上到下顺序排列。
   - 例如图片表头为“检验项目、测定结果、参考区间、单位、测定方法”，则对象 key 也按这个顺序。
   - 只填写图片中明确出现的信息；某列缺失或该行为空就写 null。
8. 如果 value 是数组/列表，必须按照图片中的出现顺序排列：从上到下、从左到右；表格按行顺序；多栏内容按人类正常阅读顺序。
9. 对日期、分期、TNM、ER/PR/HER2/Ki-67、检验指标等结构化内容，优先保留图片中的原始写法，不要自行标准化。
10. 当前 tag 的 table_name_cn 和 description 只用于理解字段含义，不能当作图片证据。

返回 JSON 格式：
{{
  "items": [
    {{
      "tag_id": "原始tag_id",
      "tag_name": "原始tag_name",
      "has_value": false,
      "value": null
    }}
  ]
}}

本次需要抽取的 tag：
{tags_json}
"""


PROMPT_TAG_FIELDS = ("tag_id", "tag_name", "table_name_cn", "description")


def strict_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            records.append(row)
    return records


def load_tag_csv(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    tags: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table_name = str(row.get("table_name_cn") or "").strip()
            tag_name = str(row.get("tag_name") or "").strip()
            if table_name and tag_name:
                tags[(table_name, tag_name)] = row
    return tags


def infer_tag_csv(train_dir: Path) -> Path | None:
    for parent in [train_dir, *train_dir.parents]:
        candidate = parent / "tag-pool_乳腺癌_20260610.csv"
        if candidate.exists():
            return candidate
    return None


def is_not_null(value: Any) -> bool:
    return value is not None


def enriched_record(row: dict[str, Any], tag_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    table_name = str(row.get("table_name_cn") or "").strip()
    tag_name = str(row.get("tag_name") or "").strip()
    csv_row = tag_index.get((table_name, tag_name), {})
    merged = dict(csv_row)
    merged.update(row)
    return merged


def normalized_tag(row: dict[str, Any], tag_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    merged = enriched_record(row, tag_index)
    return {
        field: merged.get(field)
        for field in PROMPT_TAG_FIELDS
        if merged.get(field) not in (None, "")
    }


def stage_answer_item(row: dict[str, Any], has_value: bool, include_value: bool = False) -> dict[str, Any]:
    item = {
        "tag_id": row.get("tag_id", ""),
        "tag_name": row.get("tag_name", ""),
        "has_value": has_value,
    }
    if include_value:
        item["value"] = row.get("v")
    return item


def build_user_text(tags: list[dict[str, Any]], mode: str) -> str:
    tags_json = pretty_json(tags)
    if mode == "presence":
        return SYSTEM_PROMPT.strip() + "\n\n" + PRESENCE_PROMPT_TEMPLATE.format(tags_json=tags_json)
    if mode == "extract":
        return SYSTEM_PROMPT.strip() + "\n\n" + EXTRACTION_PROMPT_TEMPLATE.format(tags_json=tags_json)
    raise ValueError(f"unsupported mode: {mode}")


def build_example(image: str, user_text: str, answer: dict[str, Any], task: str | None) -> dict[str, Any]:
    example = {
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + user_text,
            },
            {
                "from": "gpt",
                "value": strict_json(answer),
            },
        ],
    }
    if task:
        example["task"] = task
    return example


def image_for_output(image_path: Path, train_dir: Path, mode: str) -> str:
    if mode == "absolute":
        return str(image_path.resolve())
    if mode == "relative":
        return str(image_path.resolve().relative_to(train_dir.resolve()))
    raise ValueError(f"unsupported image path mode: {mode}")


def iter_image_label_pairs(train_dir: Path):
    for image_path in sorted(train_dir.glob("*.jpg")):
        jsonl_path = image_path.with_suffix(".jsonl")
        if jsonl_path.exists():
            yield image_path, jsonl_path


def random_chunks(items: list[dict[str, Any]], rng: random.Random, min_size: int, max_size: int):
    shuffled = list(items)
    rng.shuffle(shuffled)
    pos = 0
    while pos < len(shuffled):
        size = rng.randint(min_size, max_size)
        yield shuffled[pos : pos + size]
        pos += size


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train_dir = args.train_dir.expanduser().resolve()
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir does not exist: {train_dir}")

    tag_csv = args.tag_csv
    if tag_csv is None and not args.no_auto_tag_csv:
        tag_csv = infer_tag_csv(train_dir)
    tag_index = load_tag_csv(tag_csv)

    rng = random.Random(args.seed)
    examples: list[dict[str, Any]] = []
    image_count = 0
    jsonl_count = 0
    stage1_count = 0
    stage2_count = 0
    skipped_empty_jsonl = 0

    for image_path, jsonl_path in iter_image_label_pairs(train_dir):
        image_count += 1
        records = [
            enriched_record(row, tag_index)
            for row in read_jsonl(jsonl_path)
        ]
        if not records:
            skipped_empty_jsonl += 1
            continue
        jsonl_count += 1
        image_value = image_for_output(image_path, train_dir, args.image_path_mode)

        if args.include_stage1:
            for chunk in random_chunks(records, rng, args.min_presence_tags, args.max_presence_tags):
                tags = [normalized_tag(row, tag_index) for row in chunk]
                answer = {
                    "items": [
                        stage_answer_item(row, is_not_null(row.get("v")))
                        for row in chunk
                    ]
                }
                examples.append(
                    build_example(
                        image_value,
                        build_user_text(tags, "presence"),
                        answer,
                        "stage1_presence" if args.include_task_field else None,
                    )
                )
                stage1_count += 1

        if args.include_stage2:
            for row in records:
                if not is_not_null(row.get("v")):
                    continue
                tags = [normalized_tag(row, tag_index)]
                answer = {"items": [stage_answer_item(row, True, include_value=True)]}
                examples.append(
                    build_example(
                        image_value,
                        build_user_text(tags, "extract"),
                        answer,
                        "stage2_extract" if args.include_task_field else None,
                    )
                )
                stage2_count += 1

    if args.shuffle_examples:
        rng.shuffle(examples)

    summary = {
        "images_with_same_stem_jsonl": image_count,
        "non_empty_jsonl": jsonl_count,
        "skipped_empty_jsonl": skipped_empty_jsonl,
        "stage1_examples": stage1_count,
        "stage2_examples": stage2_count,
        "total_examples": len(examples),
        "tag_csv_loaded": len(tag_index),
    }
    return examples, summary


def write_output(examples: list[dict[str, Any]], output_path: Path, output_format: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl" or output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for item in examples:
                f.write(strict_json(item) + "\n")
        return
    output_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qwen-VL official Single Image Example data for Haixin stage1/stage2."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/inspire/hdd/global_user/chaimingxu-240108540141/haixin/label/train"),
        help="Directory containing xxx.jpg and the same-stem xxx.jsonl labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/haixin_stage12_single_image.json"),
        help="Output annotation file for official qwen-vl-finetune.",
    )
    parser.add_argument(
        "--tag-csv",
        type=Path,
        default=None,
        help="Optional tag-pool CSV used to enrich tag_id/description by table_name_cn + tag_name.",
    )
    parser.add_argument(
        "--no-auto-tag-csv",
        action="store_true",
        help="Do not search parent directories for tag-pool_乳腺癌_20260610.csv.",
    )
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--min-presence-tags", type=int, default=1)
    parser.add_argument("--max-presence-tags", type=int, default=5)
    parser.add_argument(
        "--image-path-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help="Use absolute image paths, or paths relative to train-dir.",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "jsonl"),
        default="json",
        help="Official code can read both JSON list and JSONL based on file extension.",
    )
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument(
        "--no-shuffle-examples",
        dest="shuffle_examples",
        action="store_false",
        help="Keep examples ordered by image: all stage1 chunks then all stage2 rows.",
    )
    parser.add_argument(
        "--include-task-field",
        action="store_true",
        help="Add a non-official helper field task=stage1_presence/stage2_extract.",
    )
    parser.set_defaults(shuffle_examples=True)
    args = parser.parse_args()
    args.include_stage1 = not args.stage2_only
    args.include_stage2 = not args.stage1_only
    if args.min_presence_tags < 1:
        raise ValueError("--min-presence-tags must be >= 1")
    if args.max_presence_tags < args.min_presence_tags:
        raise ValueError("--max-presence-tags must be >= --min-presence-tags")
    return args


def main():
    args = parse_args()
    examples, summary = build_dataset(args)
    write_output(examples, args.output, args.output_format)
    print(strict_json({"output": str(args.output), **summary}))


if __name__ == "__main__":
    main()
