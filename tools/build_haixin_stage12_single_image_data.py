#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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


TAG_META_FIELDS = (
    "cancer_type",
    "table_key",
    "table_name_cn",
    "tag_id",
    "tag_name",
    "field_key",
    "annotation_scope",
    "description",
    "description_source",
    "remark",
    "nested_keys",
    "schema_merge_components",
    "schema_merge_reason",
)


def clean_str(value: Any) -> str:
    return str(value or "").strip()


def normalize_tag_row(row: dict[str, Any]) -> dict[str, Any]:
    tag = {
        field: row.get(field, "")
        for field in TAG_META_FIELDS
        if row.get(field) not in (None, "")
    }
    if "remark" not in tag:
        remark = row.get("value") or row.get("备注") or row.get("description")
        if remark not in (None, ""):
            tag["remark"] = remark
    return tag


def load_remove_tag_names(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    obj = json.loads(path.read_text(encoding="utf-8"))
    values: list[Any] = []
    if isinstance(obj, dict):
        if "v" in obj:
            values.append(obj["v"])
        values.extend(obj.values())
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict) and "v" in item:
                values.append(item["v"])
    return {clean_str(value) for value in values if clean_str(value)}


def load_tag_csv(path: Path | None, remove_tag_names: set[str] | None = None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    remove_tag_names = remove_tag_names or set()
    tags: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        has_annotation_scope = "annotation_scope" in (reader.fieldnames or [])
        for row in reader:
            if not clean_str(row.get("tag_name")):
                continue
            if has_annotation_scope and clean_str(row.get("annotation_scope")).lower() != "structured":
                continue
            if clean_str(row.get("tag_name")) in remove_tag_names:
                continue
            tags.append(normalize_tag_row(row))
    return tags


def load_add_tags(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    tags: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tag_name = clean_str(row.get("tag_name"))
            table_name_cn = clean_str(row.get("table_name_cn"))
            if not tag_name or not table_name_cn:
                raise ValueError(
                    f"{path}:{line_no} must contain non-empty table_name_cn and tag_name."
                )
            field_key = clean_str(row.get("field_key"))
            if not field_key:
                digest = hashlib.sha1(
                    f"{table_name_cn}:{tag_name}".encode("utf-8")
                ).hexdigest()[:12]
                field_key = f"manual_{digest}"
            tag_id = clean_str(row.get("tag_id"))
            if not tag_id:
                tag_id = f"manual:{field_key}"
            normalized = normalize_tag_row(row)
            normalized.update(
                {
                    "table_name_cn": table_name_cn,
                    "tag_name": tag_name,
                    "field_key": field_key,
                    "tag_id": tag_id,
                    "cancer_type": clean_str(row.get("cancer_type")) or "乳腺癌",
                    "annotation_scope": clean_str(row.get("annotation_scope")) or "structured",
                }
            )
            description = clean_str(row.get("description"))
            if description:
                normalized["description"] = description
                normalized.setdefault("remark", description)
            tags.append(normalized)
    return tags


def infer_tag_csv(train_dir: Path) -> Path | None:
    for parent in [train_dir, *train_dir.parents]:
        candidate = parent / "tag-pool_乳腺癌_20260610.csv"
        if candidate.exists():
            return candidate
    return None


def infer_add_jsonl(train_dir: Path) -> Path | None:
    for parent in [train_dir, *train_dir.parents]:
        candidate = parent / "add.jsonl"
        if candidate.exists():
            return candidate
    return None


def infer_remove_json(train_dir: Path) -> Path | None:
    for parent in [train_dir, *train_dir.parents]:
        candidate = parent / "remove.json"
        if candidate.exists():
            return candidate
    return None


def merge_add_tags(base_tags: list[dict[str, Any]], add_tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {
        (clean_str(tag.get("table_name_cn")), clean_str(tag.get("tag_name")))
        for tag in base_tags
    }
    merged = list(base_tags)
    for tag in add_tags:
        key = (clean_str(tag.get("table_name_cn")), clean_str(tag.get("tag_name")))
        if key in seen:
            continue
        merged.append(tag)
        seen.add(key)
    return merged


def is_not_null(value: Any) -> bool:
    return value is not None


def merge_preserving_metadata(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base) if base else {}
    if not merged:
        merged.update({key: value for key, value in row.items() if value not in (None, "")})
    for key, value in row.items():
        if key == "v":
            merged[key] = value
        elif key not in TAG_META_FIELDS and value not in (None, ""):
            merged[key] = value
    return merged


class TagLookup:
    def __init__(self, tags: list[dict[str, Any]]):
        self.by_exact: dict[tuple[str, str], dict[str, Any]] = {}
        self.by_table_key: dict[tuple[str, str], dict[str, Any]] = {}
        self.by_tag_id: dict[str, dict[str, Any]] = {}
        self.by_field_key: dict[str, dict[str, Any]] = {}
        by_tag_name: dict[str, list[dict[str, Any]]] = {}
        for tag in tags:
            table_name = clean_str(tag.get("table_name_cn"))
            table_key = clean_str(tag.get("table_key"))
            tag_name = clean_str(tag.get("tag_name"))
            tag_id = clean_str(tag.get("tag_id"))
            field_key = clean_str(tag.get("field_key"))
            if table_name and tag_name:
                self.by_exact[(table_name, tag_name)] = tag
            if table_key and tag_name:
                self.by_table_key[(table_key, tag_name)] = tag
            if tag_id:
                self.by_tag_id[tag_id] = tag
            if field_key:
                self.by_field_key[field_key] = tag
            if tag_name:
                by_tag_name.setdefault(tag_name, []).append(tag)
        self.by_unique_tag_name = {
            tag_name: values[0]
            for tag_name, values in by_tag_name.items()
            if len(values) == 1
        }

    def match(self, row: dict[str, Any]) -> dict[str, Any]:
        tag_id = clean_str(row.get("tag_id"))
        if tag_id and tag_id in self.by_tag_id:
            return self.by_tag_id[tag_id]
        field_key = clean_str(row.get("field_key"))
        if field_key and field_key in self.by_field_key:
            return self.by_field_key[field_key]
        table_name = clean_str(row.get("table_name_cn"))
        tag_name = clean_str(row.get("tag_name"))
        if table_name and tag_name and (table_name, tag_name) in self.by_exact:
            return self.by_exact[(table_name, tag_name)]
        table_key = clean_str(row.get("table_key"))
        if table_key and tag_name and (table_key, tag_name) in self.by_table_key:
            return self.by_table_key[(table_key, tag_name)]
        if tag_name and tag_name in self.by_unique_tag_name:
            return self.by_unique_tag_name[tag_name]
        return {}


def enriched_record(row: dict[str, Any], tag_lookup: TagLookup) -> dict[str, Any]:
    return merge_preserving_metadata(tag_lookup.match(row), row)


def normalized_tag(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field)
        for field in PROMPT_TAG_FIELDS
        if row.get(field) not in (None, "")
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
    add_jsonl = args.add_jsonl
    if add_jsonl is None and not args.no_auto_add_jsonl:
        add_jsonl = infer_add_jsonl(train_dir)
    remove_json = args.remove_json
    if remove_json is None and not args.no_auto_remove_json:
        remove_json = infer_remove_json(train_dir)
    remove_tag_names = load_remove_tag_names(remove_json)
    tag_library = merge_add_tags(
        load_tag_csv(tag_csv, remove_tag_names=remove_tag_names),
        load_add_tags(add_jsonl),
    )
    tag_lookup = TagLookup(tag_library)

    rng = random.Random(args.seed)
    examples: list[dict[str, Any]] = []
    image_count = 0
    jsonl_count = 0
    stage1_count = 0
    stage2_count = 0
    stage2_positive_count = 0
    stage2_negative_count = 0
    skipped_empty_jsonl = 0
    missing_tag_id = 0
    missing_description = 0
    missing_examples: list[dict[str, str]] = []
    unmatched_records: list[dict[str, str]] = []
    skipped_unmatched_records = 0

    for image_path, jsonl_path in iter_image_label_pairs(train_dir):
        image_count += 1
        records = []
        for raw_row in read_jsonl(jsonl_path):
            matched_tag = tag_lookup.match(raw_row)
            if not matched_tag:
                unmatched = {
                    "image": image_path.name,
                    "table_name_cn": clean_str(raw_row.get("table_name_cn")),
                    "table_key": clean_str(raw_row.get("table_key")),
                    "field_key": clean_str(raw_row.get("field_key")),
                    "tag_id": clean_str(raw_row.get("tag_id")),
                    "tag_name": clean_str(raw_row.get("tag_name")),
                }
                unmatched_records.append(unmatched)
                if args.allow_unmatched:
                    skipped_unmatched_records += 1
                    continue
                if len(unmatched_records) >= args.max_missing_examples:
                    raise ValueError(
                        "Some jsonl records do not match the final tag pool "
                        "(tag-pool minus remove plus add). Examples: "
                        + strict_json(unmatched_records[: args.max_missing_examples])
                    )
                continue
            records.append(merge_preserving_metadata(matched_tag, raw_row))
        if unmatched_records and not args.allow_unmatched:
            raise ValueError(
                "Some jsonl records do not match the final tag pool "
                "(tag-pool minus remove plus add). Examples: "
                + strict_json(unmatched_records[: args.max_missing_examples])
            )
        if not records:
            skipped_empty_jsonl += 1
            continue
        jsonl_count += 1
        image_value = image_for_output(image_path, train_dir, args.image_path_mode)
        for row in records:
            if not clean_str(row.get("tag_id")):
                missing_tag_id += 1
            if not clean_str(row.get("description")):
                missing_description += 1
                if len(missing_examples) < args.max_missing_examples:
                    missing_examples.append(
                        {
                            "image": image_path.name,
                            "table_name_cn": clean_str(row.get("table_name_cn")),
                            "table_key": clean_str(row.get("table_key")),
                            "tag_name": clean_str(row.get("tag_name")),
                        }
                    )

        if args.include_stage1:
            for chunk in random_chunks(records, rng, args.min_presence_tags, args.max_presence_tags):
                tags = [normalized_tag(row) for row in chunk]
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
            positive_rows = [row for row in records if is_not_null(row.get("v"))]
            negative_rows = [row for row in records if not is_not_null(row.get("v"))]
            for row in positive_rows:
                tags = [normalized_tag(row)]
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
                stage2_positive_count += 1

            if args.stage2_negative_ratio > 0 and negative_rows:
                if positive_rows:
                    negative_count = int(len(positive_rows) * args.stage2_negative_ratio)
                    if args.stage2_negative_ratio > 0 and negative_count == 0:
                        negative_count = 1
                else:
                    negative_count = 1 if args.stage2_include_negative_without_positive else 0
                negative_count = min(negative_count, len(negative_rows))
                for row in rng.sample(negative_rows, k=negative_count):
                    tags = [normalized_tag(row)]
                    answer = {"items": [stage_answer_item(row, False, include_value=True)]}
                    examples.append(
                        build_example(
                            image_value,
                            build_user_text(tags, "extract"),
                            answer,
                            "stage2_extract_negative" if args.include_task_field else None,
                        )
                    )
                    stage2_count += 1
                    stage2_negative_count += 1

    if args.shuffle_examples:
        rng.shuffle(examples)

    summary = {
        "images_with_same_stem_jsonl": image_count,
        "non_empty_jsonl": jsonl_count,
        "skipped_empty_jsonl": skipped_empty_jsonl,
        "stage1_examples": stage1_count,
        "stage2_examples": stage2_count,
        "stage2_positive_examples": stage2_positive_count,
        "stage2_negative_examples": stage2_negative_count,
        "stage2_negative_ratio": args.stage2_negative_ratio,
        "total_examples": len(examples),
        "tag_csv": str(tag_csv) if tag_csv else "",
        "add_jsonl": str(add_jsonl) if add_jsonl else "",
        "remove_json": str(remove_json) if remove_json else "",
        "remove_tag_names_loaded": len(remove_tag_names),
        "tag_library_loaded": len(tag_library),
        "records_missing_tag_id_after_enrich": missing_tag_id,
        "records_missing_description_after_enrich": missing_description,
        "missing_description_examples": missing_examples,
        "skipped_unmatched_records": skipped_unmatched_records,
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
        "--add-jsonl",
        type=Path,
        default=None,
        help="Optional add.jsonl used to enrich manually added/merged tags.",
    )
    parser.add_argument(
        "--remove-json",
        type=Path,
        default=None,
        help="Optional remove.json used exactly like API extraction: remove CSV tags by tag_name before adding add.jsonl.",
    )
    parser.add_argument(
        "--no-auto-tag-csv",
        action="store_true",
        help="Do not search parent directories for tag-pool_乳腺癌_20260610.csv.",
    )
    parser.add_argument(
        "--no-auto-add-jsonl",
        action="store_true",
        help="Do not search parent directories for add.jsonl.",
    )
    parser.add_argument(
        "--no-auto-remove-json",
        action="store_true",
        help="Do not search parent directories for remove.json.",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Skip jsonl records that cannot be matched to tag-pool minus remove plus add. By default this is an error.",
    )
    parser.add_argument(
        "--max-missing-examples",
        type=int,
        default=20,
        help="Maximum missing-description examples to print in the summary.",
    )
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--min-presence-tags", type=int, default=1)
    parser.add_argument("--max-presence-tags", type=int, default=5)
    parser.add_argument(
        "--stage2-negative-ratio",
        type=float,
        default=0.5,
        help="Sample about this many stage2 negative examples per positive example, per image. Use 0 to disable.",
    )
    parser.add_argument(
        "--stage2-include-negative-without-positive",
        action="store_true",
        help="If an image has no stage2 positive rows, sample one negative row from it.",
    )
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
    if args.stage2_negative_ratio < 0:
        raise ValueError("--stage2-negative-ratio must be >= 0")
    return args


def main():
    args = parse_args()
    examples, summary = build_dataset(args)
    write_output(examples, args.output, args.output_format)
    print(strict_json({"output": str(args.output), **summary}))


if __name__ == "__main__":
    main()
