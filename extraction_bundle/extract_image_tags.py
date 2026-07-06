#!/usr/bin/env python3
"""Extract CSV-defined tags from normalized medical record images.

Each image gets one same-stem JSONL result file beside the image itself, such as
images/0001.jsonl. Each line corresponds to one CSV tag row and contains:
Chinese table name, tag name, and extracted value.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_FALLBACK_CONFIG_PATHS = (
    Path(__file__).resolve().parent / "extraction_config_copy_1.yaml",
    Path(__file__).resolve().parent / "extraction_config_copy_2.yaml",
)
CACHE_VERSION = "presence-extract-v10-no-tag-id-output"
RETRY_AFTER_EXHAUSTED_SECONDS = 3600
REFUSAL_RETRY_SLEEP_SECONDS = 300
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:22002/v1"
DEFAULT_VLLM_MODEL = "haixin_stage12"
DEFAULT_VLLM_API_KEY = "EMPTY"
MAX_IMAGE_PIXELS_CAP = 1048576
REFUSAL_ERROR_MARKERS = (
    "sensitive_words_detected",
    "content_filter",
    "content policy",
    "safety",
    "unsafe",
    "refusal",
    "refused",
    "拒绝",
    "抱歉",
    "无法提供",
    "不能提供",
    "不能协助",
)


CACHE_STATS = {
    "local_hits": 0,
    "api_calls": 0,
}


class RequestContextError(RuntimeError):
    """Carry the tag request that produced an API/model error."""

    def __init__(
        self,
        original_error: BaseException,
        *,
        request_mode: str,
        tag_chunk: list[dict[str, Any]],
        chunk_index: int | None = None,
        chunk_total: int | None = None,
    ) -> None:
        super().__init__(str(original_error))
        self.original_error = original_error
        self.request_mode = request_mode
        self.tag_chunk = tag_chunk
        self.chunk_index = chunk_index
        self.chunk_total = chunk_total


class NonRetryableApiError(RuntimeError):
    """API errors that should fail immediately instead of sleeping and retrying."""


class ApiRequestError(RuntimeError):
    """HTTP/network error returned by the model gateway."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str = "",
        response_body: str = "",
        response_headers: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.response_body = response_body
        self.response_headers = response_headers


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
3. 只有图片中能看到与该 tag 对应的信息时，has_value 需要为 true。
4. 如果图片与某个 tag 无关，或没有明确证据，has_value 应该为 false。
5. 不要补充输入列表以外的 tag。
6. 每个 tag 的 table_name_cn 和 description 只用于理解字段含义，不能当作图片证据。

返回 JSON 格式：
{{
  "items": [
    {{
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
11. 对药物、医嘱、治疗方案等明细字段，最小对象粒度应由“名称/药品/方案”和“日期/时间/周期”等共同决定。如果图片写明多个具体日期且同一段同时列出多个药品/方案，如“2024-3-15、2025-4-15、2026-5-16（药品A+药品B+药品C+药品D）”，原则上要按“日期 × 药品/方案”拆成多条对象；这个例子应返回 12 条，而不是 4 条。

返回 JSON 格式：
{{
  "items": [
    {{
      "tag_name": "原始tag_name",
      "has_value": false,
      "value": null
    }}
  ]
}}

本次需要抽取的 tag：
{tags_json}
"""


NESTED_REVIEW_PROMPT_TEMPLATE = """请复核并修正这张图片中下面这个列表字段的抽取结果。

这是第二遍质量检查：第一遍已经给出 current_value，但它可能把同一药品不同日期混在一起，或把多个药品/日期的对应关系拆错。

要求：
1. 先重新查看图片，只根据图片中清晰可见的信息修正，不要根据医学常识猜测。
2. has_value 为 false 时，value 必须为 null。
3. has_value 为 true 时，value 必须是数组。
   - 如果字段定义里的 nested_keys 非空，每个对象只包含 nested_keys 中列出的 key；缺失值写 null。
   - 如果 current_value 是字符串/数字等普通值组成的列表，复核后仍返回普通值列表，不要改成对象数组。
   - 如果 nested_keys 为空，沿用 current_value 中已有对象 key；若 current_value 来自表格，则按图片表头原文和从左到右顺序保留 key。
4. 对药物、医嘱、治疗方案等明细字段，最小对象粒度应由“名称/药品/方案”和“日期/时间/周期”等共同决定。
   - 不能只按药品名、方案名或名称去重。
   - 一个对象应由 object_identity_dimensions 中多个维度共同决定。
   - 同一药品如果对应不同给药日期、剂量、规格、给药方式或频次，应拆成多条对象。
   - 同一药品同一日期的剂量、规格、方式、频次等应放在同一个对象中。
   - 不同药品不能合并到同一个对象。
   - 如果图片写明多个具体日期且同一段同时列出多个药品/方案，如“2024-3-15、2025-4-15、2026-5-16（药品A+药品B+药品C+药品D）”，原则上要按“日期 × 药品/方案”拆成多条对象；这个例子应返回 12 条，而不是 4 条。
   - 同一个事实在现病史、处理、处方等区域重复出现时，只有 object_identity_dimensions 完全一致才合并；任一维度不同都要保留为不同对象。
5. 复核后 value 数组中的元素必须按照图片中的出现顺序排列：基本上是从上到下，从左到右对应人类正常阅读顺序；表格按行顺序；多栏内容按人类正常阅读顺序。不要按名称、日期、数值大小或字段顺序自行重排。
6. 不要输出 Markdown，不要解释，只输出严格 JSON。

返回 JSON 格式：
{{
  "items": [
    {{
      "tag_name": "原始tag_name",
      "has_value": true,
      "value": []
    }}
  ]
}}

字段定义：
{tag_json}

第一遍抽取结果 current_value：
{current_value_json}
"""


USER_PROMPT_TEMPLATE = """请从这张图片中抽取下面 tag 的值。

要求：
1. 每个输入 tag 都必须在返回 JSON 的 items 中出现一次。
2. value 找不到就写 null。
3. 不要补充输入列表以外的 tag。
4. 注意同一段原文可能同时对应多个 tag。遇到组合值时要在相关字段里分别填写拆分值，同时在整体字段里保留原文整体值。
   例如图片原文是“pTNM分期：pT2N0Mx”，如果本次 tags 中包含 t_stage、n_stage、m_stage、overall_stage，
   则应返回 t_stage="pT2"、n_stage="N0"、m_stage="Mx"、overall_stage="pT2N0Mx"。
5. 如果 tag 对应表格明细中的一行或多行项目，除非 description 明确指定返回字段，否则 value 必须按图片表头原文返回对象或对象数组。
   - 对象 key 使用图片中对应表格的列名原文，不要改写成固定字段名。
   - key 的顺序必须与表头从左到右一致；多行对象必须按表格从上到下顺序排列。
   - 例如图片表头为“检验项目、测定结果、参考区间、单位、测定方法”，则对象 key 也按这个顺序。
   - 只填写图片中明确出现的信息；某列缺失或该行为空就写 null。
6. 如果 value 是数组/列表，必须按照图片中的出现顺序排列：从上到下、从左到右；表格按行顺序；多栏内容按人类正常阅读顺序。不要按名称、日期、数值大小或字段顺序自行重排。
7. 对日期、分期、TNM、ER/PR/HER2/Ki-67、检验指标等结构化内容，优先保留图片中的原始写法，不要自行标准化。
8. 每个 tag 的 table_name_cn 和 description 只用于理解字段含义，不能当作图片证据。

返回 JSON 格式：
{{
  "items": [
    {{
      "tag_name": "原始tag_name",
      "value": null
    }}
  ]
}}

本次需要抽取的 tags：
{tags_json}
"""


def first_nonempty(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def load_config(path: Path, require_credentials: bool = True) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    config["api_key"] = str(os.environ.get("VLLM_API_KEY", "") or DEFAULT_VLLM_API_KEY)
    config["base_url"] = str(
        os.environ.get("VLLM_BASE_URL", "")
        or DEFAULT_VLLM_BASE_URL
    )
    config["model"] = str(
        os.environ.get("VLLM_MODEL", "")
        or DEFAULT_VLLM_MODEL
    )

    config.setdefault("api_type", "openai")
    config.setdefault("openai_path", "/chat/completions")
    config.setdefault("api_key_header", "Authorization")
    config.setdefault("api_key_prefix", "Bearer")
    config.setdefault("tags_per_call", 4)
    config.setdefault("image_detail", "high")
    config.setdefault("temperature", 0)
    config.setdefault("top_p", 0.01)
    config.setdefault("timeout_seconds", 180)
    config.setdefault("max_retries", 3)
    config.setdefault("max_tokens", 24000)
    config.setdefault("max_workers", 1)
    config.setdefault("response_format_json", False)
    config.setdefault("prompt_cache_enabled", False)
    config.setdefault("send_prompt_cache_key", False)
    config.setdefault("output_format", "tagged_object")

    required = ["base_url", "model"]
    missing = [k for k in required if not config.get(k)]
    if require_credentials and missing:
        raise SystemExit(
            f"Missing config values in {path}: {', '.join(missing)}. "
            "Set VLLM_BASE_URL and VLLM_MODEL in the environment if the defaults "
            "do not match your local vLLM deployment."
        )
    config["api_type"] = str(config.get("api_type", "openai")).strip().lower()
    if config["api_type"] not in {"openai", "anthropic"}:
        raise SystemExit("api_type must be either 'openai' or 'anthropic'.")
    return config


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
    return {str(v).strip() for v in values if v is not None and str(v).strip()}


def load_tags(path: Path, remove_tag_names: set[str] | None = None) -> list[dict[str, str]]:
    remove_tag_names = remove_tag_names or set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        has_annotation_scope = "annotation_scope" in (reader.fieldnames or [])
        rows = [r for r in reader if r.get("tag_name")]
    if has_annotation_scope:
        rows = [
            r
            for r in rows
            if r.get("annotation_scope", "").strip().lower() == "structured"
        ]
    rows = [r for r in rows if r.get("tag_name", "").strip() not in remove_tag_names]
    return [
        {
            "cancer_type": r.get("cancer_type", ""),
            "table_key": r.get("table_key", ""),
            "table_name_cn": r.get("table_name_cn", ""),
            "tag_id": r.get("tag_id", ""),
            "tag_name": r.get("tag_name", ""),
            "field_key": r.get("field_key", ""),
            "annotation_scope": r.get("annotation_scope", ""),
            "description": r.get("description", ""),
            "remark": r.get("value") or r.get("备注") or r.get("description") or None,
        }
        for r in rows
    ]


def load_add_tags(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []

    tags = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tag_name = str(row.get("tag_name", "")).strip()
            table_name_cn = str(row.get("table_name_cn", "")).strip()
            if not tag_name or not table_name_cn:
                raise ValueError(
                    f"{path}:{line_no} must contain non-empty table_name_cn and tag_name."
                )
            field_key = str(row.get("field_key", "")).strip()
            if not field_key:
                field_key = f"manual_{hashlib.sha1((table_name_cn + ':' + tag_name).encode('utf-8')).hexdigest()[:12]}"
            tag_id = str(row.get("tag_id", "")).strip()
            if not tag_id:
                tag_id = f"manual:{field_key}"
            description = str(row.get("description", "")).strip()
            tag = {
                "cancer_type": str(row.get("cancer_type", "乳腺癌")),
                "table_key": str(row.get("table_key", "")),
                "table_name_cn": table_name_cn,
                "tag_id": tag_id,
                "tag_name": tag_name,
                "field_key": field_key,
                "annotation_scope": str(row.get("annotation_scope", "structured")),
                "description": description,
                "remark": description or None,
            }
            if isinstance(row.get("nested_keys"), list):
                tag["nested_keys"] = [str(x) for x in row["nested_keys"]]
            if isinstance(row.get("schema_merge_components"), list):
                tag["schema_merge_components"] = [
                    str(x) for x in row["schema_merge_components"]
                ]
            if row.get("schema_merge_reason"):
                tag["schema_merge_reason"] = str(row["schema_merge_reason"])
            tags.append(tag)
    return tags


def merge_add_tags(base_tags: list[dict[str, str]], add_tags: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {(tag["table_name_cn"], tag["tag_name"]) for tag in base_tags}
    merged = list(base_tags)
    for tag in add_tags:
        key = (tag["table_name_cn"], tag["tag_name"])
        if key in seen:
            continue
        merged.append(tag)
        seen.add(key)
    return merged


def group_tags_by_table(tags: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep table groups contiguous while preserving first-seen table order."""
    table_order: dict[str, int] = {}
    indexed_tags = []
    for index, tag in enumerate(tags):
        table_name = str(tag.get("table_name_cn", ""))
        if table_name not in table_order:
            table_order[table_name] = len(table_order)
        indexed_tags.append((index, tag))
    return [
        tag
        for index, tag in sorted(
            indexed_tags,
            key=lambda item: (
                table_order[str(item[1].get("table_name_cn", ""))],
                item[0],
            ),
        )
    ]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def chunks(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iter_error_chain(error: BaseException | str):
    current = error
    seen: set[int] = set()
    while current is not None:
        yield current
        if not isinstance(current, BaseException):
            break
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)
        current = getattr(current, "original_error", None) or current.__cause__


def is_refusal_error(error: BaseException | str) -> bool:
    text = " ".join(str(item).lower() for item in iter_error_chain(error))
    return any(marker.lower() in text for marker in REFUSAL_ERROR_MARKERS)


def is_timeout_error(error: BaseException | str) -> bool:
    for item in iter_error_chain(error):
        if isinstance(item, TimeoutError):
            return True
    text = " ".join(str(item).lower() for item in iter_error_chain(error))
    return any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "read operation timed out",
            "operation timed out",
        )
    )


def request_context_from_error(error: BaseException) -> RequestContextError | None:
    for item in iter_error_chain(error):
        if isinstance(item, RequestContextError):
            return item
    return None


def compact_tag_for_error_log(tag: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tag_id",
        "table_key",
        "table_name_cn",
        "tag_name",
        "field_key",
        "annotation_scope",
        "description",
        "remark",
    ]
    return {
        key: tag.get(key)
        for key in keys
        if tag.get(key) not in (None, "")
    }


def request_context_log(error: BaseException) -> dict[str, Any] | None:
    context = request_context_from_error(error)
    if context is None:
        return None
    data: dict[str, Any] = {
        "request_mode": context.request_mode,
        "chunk_index": context.chunk_index,
        "chunk_total": context.chunk_total,
        "tags": [compact_tag_for_error_log(tag) for tag in context.tag_chunk],
    }
    if isinstance(context.original_error, BaseException):
        data["original_error_type"] = type(context.original_error).__name__
        data["original_error"] = str(context.original_error)
    return data


def request_context_summary(error: BaseException) -> str:
    context = request_context_from_error(error)
    if context is None:
        return ""
    chunk = ""
    if context.chunk_index is not None:
        chunk = f" chunk={context.chunk_index}"
        if context.chunk_total is not None:
            chunk += f"/{context.chunk_total}"
    tag_names = [
        str(tag.get("tag_name"))
        for tag in context.tag_chunk
        if tag.get("tag_name")
    ]
    if tag_names:
        return f"{chunk} tags={';'.join(tag_names)}"
    return chunk


def rejected_tag_log_entries(
    image_path: Path,
    image_row: dict[str, Any],
    stage: str,
    error: BaseException,
) -> list[dict[str, Any]]:
    context = request_context_from_error(error)
    tags = context.tag_chunk if context is not None else []
    base = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_name": image_row.get("normalized_name", image_path.name),
        "image_path": str(image_path),
        "image_index": image_row.get("image_index"),
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if context is not None:
        base.update(
            {
                "request_mode": context.request_mode,
                "chunk_index": context.chunk_index,
                "chunk_total": context.chunk_total,
                "chunk_tag_count": len(tags),
            }
        )
        if isinstance(context.original_error, BaseException):
            base["original_error_type"] = type(context.original_error).__name__
            base["original_error"] = str(context.original_error)

    if not tags:
        entry = dict(base)
        entry["tag_index_in_request"] = None
        entry["tag"] = None
        return [entry]

    entries = []
    for tag_index, tag in enumerate(tags, start=1):
        entry = dict(base)
        entry["tag_index_in_request"] = tag_index
        entry["tag"] = compact_tag_for_error_log(tag)
        entries.append(entry)
    return entries


def write_rejected_tags_log(
    log_path: Path,
    image_path: Path,
    image_row: dict[str, Any],
    stage: str,
    error: BaseException,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for entry in rejected_tag_log_entries(image_path, image_row, stage, error):
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def with_request_context(
    error: BaseException,
    *,
    request_mode: str,
    tag_chunk: list[dict[str, Any]],
    chunk_index: int | None = None,
    chunk_total: int | None = None,
) -> RequestContextError:
    if isinstance(error, RequestContextError):
        return error
    return RequestContextError(
        error,
        request_mode=request_mode,
        tag_chunk=tag_chunk,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
    )


def enabled(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_image_dir(image_dir: Path) -> list[dict[str, Any]]:
    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    rows = []
    for offset, image_path in enumerate(images, start=1):
        image_index = int(image_path.stem) if image_path.stem.isdigit() else offset
        rows.append(
            {
                "image_index": image_index,
                "normalized_name": image_path.name,
                "normalized_path": str(image_path.resolve()),
                "source_sha256": sha256_file(image_path),
            }
        )
    return rows


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, **kwargs)


def effective_image_max_pixels(requested: int | None = None) -> int:
    raw_value: Any = requested
    if raw_value is None:
        raw_value = os.environ.get("MAX_PIXELS") or MAX_IMAGE_PIXELS_CAP
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"MAX_PIXELS must be an integer, got {raw_value!r}.") from exc
    if value < 1:
        raise SystemExit("MAX_PIXELS must be >= 1.")
    return min(value, MAX_IMAGE_PIXELS_CAP)


def resized_image_bytes(path: Path, max_pixels: int) -> tuple[str, bytes]:
    if Image is None or ImageOps is None:
        raise SystemExit(
            "Pillow is required to enforce image max_pixels before extraction. "
            "Install pillow or use an environment that already includes PIL."
        )
    with Image.open(path) as image:
        original_format = (image.format or "").upper()
        width, height = image.size
        if width * height <= max_pixels:
            return mimetypes.guess_type(path.name)[0] or "image/jpeg", path.read_bytes()

        image = ImageOps.exif_transpose(image)
        width, height = image.size
        scale = (max_pixels / float(width * height)) ** 0.5
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        while new_width * new_height > max_pixels:
            if new_width >= new_height and new_width > 1:
                new_width -= 1
            elif new_height > 1:
                new_height -= 1
            else:
                break

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize((new_width, new_height), resampling)
        buffer = io.BytesIO()
        if original_format == "PNG":
            resized.save(buffer, format="PNG", optimize=True)
            return "image/png", buffer.getvalue()

        if resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")
        resized.save(buffer, format="JPEG", quality=95, optimize=True)
        return "image/jpeg", buffer.getvalue()


def image_to_base64(path: Path, max_pixels: int = MAX_IMAGE_PIXELS_CAP) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if max_pixels < MAX_IMAGE_PIXELS_CAP:
        mime, image_bytes = resized_image_bytes(path, max_pixels)
    else:
        maybe_mime, image_bytes = resized_image_bytes(path, max_pixels)
        mime = maybe_mime or mime
    data = base64.b64encode(image_bytes).decode("ascii")
    return mime, data


def prompt_tag_fields(config: dict[str, Any], mode: str) -> list[str]:
    raw = config.get(f"{mode}_prompt_tag_fields", config.get("prompt_tag_fields"))
    if raw is None:
        return ["tag_name", "table_name_cn", "description"]
    if isinstance(raw, str):
        fields = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, list):
        fields = [str(item).strip() for item in raw]
    else:
        fields = []
    return [field for field in fields if field]


def tag_for_prompt(tag: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {
        field: tag.get(field)
        for field in fields
        if tag.get(field) not in (None, "")
    }


def template_for_mode(mode: str) -> str:
    if mode == "presence":
        return PRESENCE_PROMPT_TEMPLATE
    if mode == "extract":
        return EXTRACTION_PROMPT_TEMPLATE
    return USER_PROMPT_TEMPLATE


def build_user_text(
    tag_chunk: list[dict[str, str]],
    mode: str,
    config: dict[str, Any],
) -> str:
    fields = prompt_tag_fields(config, mode)
    tags_for_prompt = [tag_for_prompt(t, fields) for t in tag_chunk]
    template = template_for_mode(mode)
    return template.format(
        tags_json=json.dumps(tags_for_prompt, ensure_ascii=False, indent=2)
    )


def build_user_text_parts(
    tag_chunk: list[dict[str, str]],
    mode: str,
    config: dict[str, Any],
) -> list[str]:
    fields = prompt_tag_fields(config, mode)
    tags_for_prompt = [tag_for_prompt(t, fields) for t in tag_chunk]
    tags_json = json.dumps(tags_for_prompt, ensure_ascii=False, indent=2)
    template = template_for_mode(mode)
    marker = "{tags_json}"
    if marker not in template:
        return [template.format(tags_json=tags_json)]
    prefix, suffix = template.split(marker, 1)
    fixed_text = prefix.rstrip()
    variable_text = tags_json + suffix
    return [part for part in (fixed_text, variable_text.strip()) if part]


def build_openai_payload(
    config: dict[str, Any],
    image_data_url: str,
    image_sha256: str,
    tag_chunk: list[dict[str, str]],
    mode: str,
) -> dict[str, Any]:
    user_text_parts = build_user_text_parts(tag_chunk, mode, config)
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url,
                "detail": config.get("image_detail", "high"),
            },
        },
    ]
    content.extend({"type": "text", "text": part} for part in user_text_parts)
    payload = {
        "model": config["model"],
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_tokens", 4096),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": content,
            },
        ],
    }
    if config.get("send_prompt_cache_key"):
        payload["prompt_cache_key"] = f"img:{image_sha256}"
    if enabled(config, "prompt_cache_enabled", False):
        payload["prompt_cache_key"] = str(
            config.get("prompt_cache_key")
            or f"image:{image_sha256[:24]}:{mode}"
        )
        if config.get("prompt_cache_retention"):
            payload["prompt_cache_retention"] = str(config["prompt_cache_retention"])
    if config.get("response_format_json"):
        payload["response_format"] = {"type": "json_object"}
    return payload


def build_anthropic_payload(
    config: dict[str, Any],
    image_media_type: str,
    image_base64: str,
    tag_chunk: list[dict[str, str]],
    mode: str,
) -> dict[str, Any]:
    user_text_parts = build_user_text_parts(tag_chunk, mode, config)
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_base64,
            },
        },
    ]
    content.extend({"type": "text", "text": part} for part in user_text_parts)
    return {
        "model": config["model"],
        "system": SYSTEM_PROMPT,
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_tokens", 4096),
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def nested_keys_from_tag(tag: dict[str, Any]) -> list[str]:
    keys = tag.get("nested_keys")
    if isinstance(keys, list):
        return [str(x).strip() for x in keys if str(x).strip()]
    return []


def identity_dimensions_for_tag(tag: dict[str, Any]) -> list[str]:
    keys = nested_keys_from_tag(tag)
    if not keys:
        return []

    dimensions: list[str] = []

    def add_matches(patterns: tuple[str, ...]) -> None:
        for key in keys:
            if key in dimensions:
                continue
            if any(pattern in key for pattern in patterns):
                dimensions.append(key)

    add_matches(("药品通用名", "医嘱/药品名称", "药品信息", "药物名称", "方案名称", "名称"))
    add_matches(("给药日期", "治疗日期", "检查日期", "日期", "时间", "当前周期", "周期内天数", "周期"))
    add_matches(("实际使用剂量", "药物剂量", "单次剂量", "剂量"))
    add_matches(("剂量单位", "单位", "药物规格", "规格", "剂型"))
    add_matches(("给药方式", "给药频次", "给药方式频次", "频次", "方式"))
    add_matches(("治疗类别", "药物类别", "药物细分类别", "治疗线级", "治疗目的"))

    if not dimensions:
        dimensions = keys[: min(3, len(keys))]
    return dimensions


def review_tag_json(tag: dict[str, Any]) -> dict[str, Any]:
    nested_keys = nested_keys_from_tag(tag)
    return {
        "tag_name": tag.get("tag_name", ""),
        "nested_keys": nested_keys,
        "object_identity_dimensions": identity_dimensions_for_tag(tag),
        "schema_merge_components": tag.get("schema_merge_components") or [],
        "schema_merge_reason": tag.get("schema_merge_reason", ""),
    }


def build_nested_review_user_text(tag: dict[str, Any], current_value: Any) -> str:
    return NESTED_REVIEW_PROMPT_TEMPLATE.format(
        tag_json=json.dumps(review_tag_json(tag), ensure_ascii=False, indent=2),
        current_value_json=json.dumps(current_value, ensure_ascii=False, indent=2),
    )


def build_openai_nested_review_payload(
    config: dict[str, Any],
    image_data_url: str,
    image_sha256: str,
    tag: dict[str, Any],
    current_value: Any,
) -> dict[str, Any]:
    payload = {
        "model": config["model"],
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get(
            "nested_review_max_tokens",
            config.get("max_tokens", 4096),
        ),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": config.get("image_detail", "high"),
                        },
                    },
                    {
                        "type": "text",
                        "text": build_nested_review_user_text(tag, current_value),
                    },
                ],
            },
        ],
    }
    if config.get("send_prompt_cache_key"):
        payload["prompt_cache_key"] = f"img:{image_sha256}"
    if enabled(config, "prompt_cache_enabled", False):
        payload["prompt_cache_key"] = str(
            config.get("prompt_cache_key")
            or f"image:{image_sha256[:24]}:nested_review"
        )
        if config.get("prompt_cache_retention"):
            payload["prompt_cache_retention"] = str(config["prompt_cache_retention"])
    if config.get("response_format_json"):
        payload["response_format"] = {"type": "json_object"}
    return payload


def build_anthropic_nested_review_payload(
    config: dict[str, Any],
    image_media_type: str,
    image_base64: str,
    tag: dict[str, Any],
    current_value: Any,
) -> dict[str, Any]:
    return {
        "model": config["model"],
        "system": SYSTEM_PROMPT,
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get(
            "nested_review_max_tokens",
            config.get("max_tokens", 4096),
        ),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": build_nested_review_user_text(tag, current_value),
                    },
                ],
            }
        ],
    }


def resolve_api_url(base_url: str, path: str) -> str:
    base = str(base_url).rstrip("/")
    path = str(path or "").strip()
    if not path:
        return base
    normalized_path = "/" + path.strip("/")
    if base.endswith(normalized_path):
        return base
    if normalized_path == "/chat/completions" and base.endswith("/chat/completions"):
        return base
    return base + normalized_path


def build_api_auth_headers(
    config: dict[str, Any],
    *,
    default_header_name: str,
    default_prefix: str = "",
) -> dict[str, str]:
    header_name = str(config.get("api_key_header", default_header_name)).strip()
    prefix = config.get("api_key_prefix", default_prefix)
    prefix = str(prefix).strip() if prefix is not None else ""
    api_key = str(config["api_key"])
    return {header_name: f"{prefix} {api_key}" if prefix else api_key}


def compact_text(text: str, limit: int = 2000) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def tag_chunk_summary(tag_chunk: list[dict[str, Any]]) -> str:
    parts = []
    for tag in tag_chunk:
        table_name = str(tag.get("table_name_cn", "")).strip()
        tag_name = str(tag.get("tag_name", "")).strip()
        if table_name and tag_name:
            parts.append(f"{table_name}/{tag_name}")
        elif tag_name:
            parts.append(tag_name)
    return "; ".join(parts) if parts else "(no tags)"


def request_label(
    mode: str,
    tag_chunk: list[dict[str, Any]],
    chunk_index: int | None,
    chunk_total: int | None,
) -> str:
    chunk = ""
    if chunk_index is not None:
        chunk = f" chunk={chunk_index}"
        if chunk_total is not None:
            chunk += f"/{chunk_total}"
    return f"stage={mode}{chunk} tags={tag_chunk_summary(tag_chunk)}"


def log_timeout_skip(
    mode: str,
    tag_chunk: list[dict[str, Any]],
    chunk_index: int | None,
    chunk_total: int | None,
    error: BaseException,
) -> None:
    print(
        "REQUEST_TIMEOUT_SKIP "
        f"{request_label(mode, tag_chunk, chunk_index, chunk_total)} "
        f"error={compact_text(str(error), 1200)}",
        flush=True,
    )


def compact_usage_value(value: Any) -> str:
    if isinstance(value, dict):
        return stable_json(value)
    return str(value)


def usage_summary(response: dict[str, Any]) -> str:
    usage = response.get("usage")
    if not isinstance(usage, dict) or not usage:
        return "usage=unavailable"
    parts = []
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "prompt_tokens_details",
        "input_tokens_details",
    ):
        if key in usage:
            parts.append(f"{key}={compact_usage_value(usage[key])}")
    if not parts:
        parts = [f"{key}={compact_usage_value(value)}" for key, value in usage.items()]
    return "usage " + " ".join(parts)


def log_local_cache_hit(
    mode: str,
    tag_chunk: list[dict[str, Any]],
    chunk_index: int | None,
    chunk_total: int | None,
    cache_path: Path,
) -> None:
    CACHE_STATS["local_hits"] += 1
    print(
        f"LOCAL_CACHE_HIT {request_label(mode, tag_chunk, chunk_index, chunk_total)} "
        f"cache={cache_path.name} total_local_hits={CACHE_STATS['local_hits']}",
        flush=True,
    )


def log_api_usage(
    mode: str,
    tag_chunk: list[dict[str, Any]],
    chunk_index: int | None,
    chunk_total: int | None,
    response: dict[str, Any],
) -> None:
    CACHE_STATS["api_calls"] += 1
    print(
        f"API_USAGE {request_label(mode, tag_chunk, chunk_index, chunk_total)} "
        f"{usage_summary(response)} total_api_calls={CACHE_STATS['api_calls']}",
        flush=True,
    )


def add_request_context_to_error(
    error: BaseException,
    *,
    request_mode: str,
    tag_chunk: list[dict[str, Any]],
    chunk_index: int | None = None,
    chunk_total: int | None = None,
) -> RequestContextError:
    context_bits = [f"mode={request_mode}"]
    if chunk_index is not None:
        chunk = f"chunk={chunk_index}"
        if chunk_total is not None:
            chunk += f"/{chunk_total}"
        context_bits.append(chunk)
    context_bits.append(f"tags={tag_chunk_summary(tag_chunk)}")
    message = f"{error}; {'; '.join(context_bits)}"
    if isinstance(error, ApiRequestError):
        if error.response_headers:
            message += f"; response_headers={compact_text(error.response_headers, 1000)}"
        if error.response_body:
            message += f"; response_body={compact_text(error.response_body)}"
    wrapped = RequestContextError(
        RuntimeError(message),
        request_mode=request_mode,
        tag_chunk=tag_chunk,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
    )
    wrapped.original_error = error
    return wrapped


def sleep_before_retrying_same_request(message: str) -> None:
    print(
        f"{message} Sleeping {RETRY_AFTER_EXHAUSTED_SECONDS} seconds before "
        "retrying the same request.",
        flush=True,
    )
    time.sleep(RETRY_AFTER_EXHAUSTED_SECONDS)


def post_json(
    base_url: str,
    api_key: str,
    path: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    url = resolve_api_url(base_url, path)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", **headers}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            response_headers = "\n".join(
                f"{key}: {value}" for key, value in e.headers.items()
            )
            message = (
                f"HTTP {e.code} from {url} on attempt {attempt}/{max_retries}: "
                f"{compact_text(detail)}"
            )
            if e.code in {401, 403}:
                raise NonRetryableApiError(message) from e
            last_error = ApiRequestError(
                message,
                status_code=e.code,
                url=url,
                response_body=detail,
                response_headers=response_headers,
            )
            if is_refusal_error(last_error):
                raise last_error
        except Exception as e:  # noqa: BLE001
            last_error = e
        if attempt < max_retries:
            time.sleep(min(2**attempt, 20))
    if isinstance(last_error, ApiRequestError):
        raise last_error
    raise ApiRequestError(
        f"Request failed after {max_retries} attempts: {last_error}",
        url=url,
        response_body=str(last_error or ""),
    )


def extract_text_from_openai_response(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def extract_text_from_anthropic_response(response: dict[str, Any]) -> str:
    content = response.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def extract_text_from_response(response: dict[str, Any], api_type: str) -> str:
    if api_type == "anthropic":
        content = extract_text_from_anthropic_response(response)
        return content or extract_text_from_openai_response(response)
    return extract_text_from_openai_response(response)


def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Model returned empty content.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        snippet = text[:500].replace("\n", "\\n")
        raise ValueError(f"Model returned non-JSON content: {snippet}") from None


def default_records(tags: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "table_name_cn": t["table_name_cn"],
            "tag_name": t["tag_name"],
            "v": None,
        }
        for t in tags
    ]


def default_presence_records(tags: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "table_name_cn": t["table_name_cn"],
            "tag_name": t["tag_name"],
            "has_value": False,
        }
        for t in tags
    ]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp.replace(path)


def write_json_file(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def image_sidecar_paths(image_path: Path) -> list[Path]:
    # Move every same-stem file in the image directory, but leave .api_cache alone.
    return sorted(
        [
            path
            for path in image_path.parent.iterdir()
            if path.is_file()
            and (path.name == image_path.name or path.name.startswith(f"{image_path.stem}."))
        ],
        key=lambda path: path.name,
    )


def current_image_record_paths(image_path: Path) -> list[Path]:
    paths = []
    for path in image_path.parent.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(f"{image_path.stem}."):
            continue
        if path.name == image_path.name:
            continue
        if (
            path.suffix.lower() in {".jsonl", ".json"}
            or path.name.endswith(".jsonl.done")
        ):
            paths.append(path)
    return sorted(paths, key=lambda path: path.name)


def cleanup_current_image_records_and_cache(
    image_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    deleted_records = []
    for path in current_image_record_paths(image_path):
        if not path.exists():
            continue
        path.unlink()
        deleted_records.append(str(path))

    cache_cleared = False
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_cleared = True

    return {
        "deleted_records": deleted_records,
        "cache_dir": str(cache_dir),
        "cache_cleared": cache_cleared,
    }


def cache_dir_for_image(image_path: Path, image_sha256: str, cache_scope: str) -> Path:
    base_cache_dir = image_path.parent / ".api_cache"
    if cache_scope == "shared":
        return base_cache_dir
    image_cache_key = image_sha256[:16] if image_sha256 else image_path.stem
    return base_cache_dir / f"{image_path.stem}-{image_cache_key}"


def clear_image_cache(cache_dir: Path, image_name: str) -> None:
    if not cache_dir.exists():
        return
    shutil.rmtree(cache_dir)
    print(
        f"IMAGE_CACHE_CLEARED image={image_name} cache_dir={cache_dir}",
        flush=True,
    )


def unique_error_stem(error_dir: Path, image_path: Path) -> str:
    base = f"{image_path.parent.name}__{image_path.stem}"
    stem = base
    counter = 2
    while (error_dir / f"{stem}{image_path.suffix}").exists():
        stem = f"{base}__{counter}"
        counter += 1
    return stem


def move_error_image_bundle(
    image_path: Path,
    image_row: dict[str, Any],
    error_dir: Path,
    stage: str,
    error: BaseException,
) -> None:
    error_dir.mkdir(parents=True, exist_ok=True)
    target_stem = unique_error_stem(error_dir, image_path)
    moved_files = []
    for source in image_sidecar_paths(image_path):
        if not source.exists():
            continue
        suffix = source.name[len(image_path.stem) :] if source.name.startswith(image_path.stem) else source.suffix
        target = error_dir / f"{target_stem}{suffix}"
        shutil.move(str(source), str(target))
        moved_files.append({"from": str(source), "to": str(target)})

    log_entry = {
        "image_name": image_row.get("normalized_name", image_path.name),
        "image_path": str(image_path),
        "image_index": image_row.get("image_index"),
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "moved_files": moved_files,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    failed_request = request_context_log(error)
    if failed_request is not None:
        log_entry["failed_request"] = failed_request
    with (error_dir / "error_images.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def handle_refusal_image(
    image_path: Path,
    image_row: dict[str, Any],
    error_dir: Path | None,
    stage: str,
    error: BaseException,
    rejected_tags_log_path: Path | None = None,
) -> bool:
    if not is_refusal_error(error):
        return False
    if rejected_tags_log_path is not None:
        write_rejected_tags_log(
            rejected_tags_log_path,
            image_path,
            image_row,
            stage,
            error,
        )
    if error_dir is None:
        return False
    move_error_image_bundle(image_path, image_row, error_dir, stage, error)
    context_summary = request_context_summary(error)
    print(
        f"REFUSAL {image_row.get('normalized_name', image_path.name)} "
        f"stage={stage}{context_summary} moved_to={error_dir}"
    )
    return True


def cache_key_for_request(
    config: dict[str, Any],
    image_sha256: str,
    mode: str,
    tag_chunk: list[dict[str, str]],
) -> str:
    fields = prompt_tag_fields(config, mode)
    payload = {
        "cache_version": CACHE_VERSION,
        "api_type": config.get("api_type", "openai"),
        "base_url": config.get("base_url", ""),
        "model": config.get("model", ""),
        "mode": mode,
        "image_sha256": image_sha256,
        "image_max_pixels": config.get("image_max_pixels", MAX_IMAGE_PIXELS_CAP),
        "image_detail": config.get("image_detail", "high"),
        "temperature": config.get("temperature", 0),
        "prompt_tag_fields": fields,
        "tags": [tag_for_prompt(t, fields) for t in tag_chunk],
    }
    if mode != "presence":
        payload["list_order_policy"] = "image_reading_order_v1"
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def cache_key_for_nested_review(
    config: dict[str, Any],
    image_sha256: str,
    tag: dict[str, Any],
    current_value: Any,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "api_type": config.get("api_type", "openai"),
        "base_url": config.get("base_url", ""),
        "model": config.get("model", ""),
        "mode": "nested_review",
        "image_sha256": image_sha256,
        "image_max_pixels": config.get("image_max_pixels", MAX_IMAGE_PIXELS_CAP),
        "image_detail": config.get("image_detail", "high"),
        "temperature": config.get("temperature", 0),
        "list_order_policy": "image_reading_order_v1",
        "tag": review_tag_json(tag),
        "current_value": current_value,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def read_cached_items(cache_path: Path) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    try:
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = obj.get("items") if isinstance(obj, dict) else None
    return items if isinstance(items, list) else None


def write_cached_items(cache_path: Path, items: list[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(cache_path)


def item_matches_tag(item: dict[str, Any], tag: dict[str, Any]) -> bool:
    item_tag_id = str(item.get("tag_id", "")).strip()
    if item_tag_id and item_tag_id == str(tag.get("tag_id", "")).strip():
        return True
    item_field_key = str(item.get("field_key", "")).strip()
    if item_field_key and item_field_key == str(tag.get("field_key", "")).strip():
        return True
    item_tag_name = str(item.get("tag_name", "")).strip()
    return bool(item_tag_name and item_tag_name == str(tag.get("tag_name", "")).strip())


def tag_record_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("table_name_cn", "")), str(row.get("tag_name", "")))


def tag_by_item(
    item: dict[str, Any], tags: list[dict[str, str]]
) -> dict[str, str] | None:
    for tag in tags:
        if item_matches_tag(item, tag):
            return tag
    return None


def merge_items(
    records: list[dict[str, Any]],
    items: list[dict[str, Any]],
    tags: list[dict[str, str]],
) -> None:
    by_key = {tag_record_key(r): r for r in records}
    for item in items:
        tag = tag_by_item(item, tags)
        if tag is None:
            continue
        record = by_key.get(tag_record_key(tag))
        if record is None:
            continue
        if item.get("has_value") is False:
            record["v"] = None
        else:
            record["v"] = item.get("value")


def merge_presence_items(
    records: list[dict[str, Any]],
    items: list[dict[str, Any]],
    tags: list[dict[str, str]],
) -> None:
    by_key = {tag_record_key(r): r for r in records}
    for item in items:
        tag = tag_by_item(item, tags)
        if tag is None:
            continue
        record = by_key.get(tag_record_key(tag))
        if record is None:
            continue
        record["has_value"] = bool(item.get("has_value"))


def tags_from_presence(
    tags: list[dict[str, str]], presence_records: list[dict[str, Any]]
) -> list[dict[str, str]]:
    positive_names = {
        r["tag_name"] for r in presence_records if bool(r.get("has_value"))
    }
    return [t for t in tags if t["tag_name"] in positive_names]


def presence_record_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("table_name_cn", "")), str(row.get("tag_name", "")))


def merge_presence_union(
    tags: list[dict[str, str]],
    primary_records: list[dict[str, Any]],
    extra_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_by_key = {presence_record_key(r): bool(r.get("has_value")) for r in primary_records}
    extra_by_key = {presence_record_key(r): bool(r.get("has_value")) for r in extra_records}
    union_records = []
    for tag in tags:
        key = (tag["table_name_cn"], tag["tag_name"])
        union_records.append(
            {
                "table_name_cn": tag["table_name_cn"],
                "tag_name": tag["tag_name"],
                "has_value": bool(primary_by_key.get(key)) or bool(extra_by_key.get(key)),
            }
        )
    return union_records


def compute_presence_kappa(
    tags: list[dict[str, str]],
    primary_records: list[dict[str, Any]],
    extra_records: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_by_key = {presence_record_key(r): bool(r.get("has_value")) for r in primary_records}
    extra_by_key = {presence_record_key(r): bool(r.get("has_value")) for r in extra_records}
    tt = tf = ft = ff = 0
    disagreements: list[dict[str, Any]] = []
    for tag in tags:
        key = (tag["table_name_cn"], tag["tag_name"])
        a = bool(primary_by_key.get(key))
        b = bool(extra_by_key.get(key))
        if a and b:
            tt += 1
        elif a and not b:
            tf += 1
        elif not a and b:
            ft += 1
        else:
            ff += 1
        if a != b:
            disagreements.append(
                {
                    "table_name_cn": tag["table_name_cn"],
                    "tag_name": tag["tag_name"],
                    "primary_has_value": a,
                    "extra_has_value": b,
                }
            )
    n = tt + tf + ft + ff
    observed = (tt + ff) / n if n else None
    p_primary_true = (tt + tf) / n if n else 0
    p_primary_false = (ft + ff) / n if n else 0
    p_extra_true = (tt + ft) / n if n else 0
    p_extra_false = (tf + ff) / n if n else 0
    expected = (
        p_primary_true * p_extra_true + p_primary_false * p_extra_false
        if n
        else None
    )
    if observed is None or expected is None:
        kappa = None
    elif expected == 1:
        kappa = 1.0 if observed == 1 else None
    else:
        kappa = (observed - expected) / (1 - expected)
    return {
        "n": n,
        "primary_positive": tt + tf,
        "extra_positive": tt + ft,
        "union_positive": tt + tf + ft,
        "both_positive": tt,
        "primary_only_positive": tf,
        "extra_only_positive": ft,
        "both_negative": ff,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def is_nested_review_tag(tag: dict[str, Any], record: dict[str, Any]) -> bool:
    return isinstance(record.get("v"), list)


def value_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def identity_key_for_item(item: dict[str, Any], identity_dimensions: list[str]) -> tuple[str, ...]:
    if not identity_dimensions:
        return tuple()
    return tuple(value_key(item.get(key)) for key in identity_dimensions)


def normalize_nested_value(
    value: Any,
    nested_keys: list[str],
    identity_dimensions: list[str] | None = None,
) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return value
    if not nested_keys and not any(isinstance(item, dict) for item in value):
        return value
    identity_dimensions = identity_dimensions or []
    normalized = []
    seen_identity: set[tuple[str, ...]] = set()
    seen_full: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        if nested_keys:
            obj = {key: item.get(key) for key in nested_keys}
        else:
            obj = item
        ident = identity_key_for_item(obj, identity_dimensions)
        if identity_dimensions and any(ident):
            if ident in seen_identity:
                continue
            seen_identity.add(ident)
        else:
            full = stable_json(obj)
            if full in seen_full:
                continue
            seen_full.add(full)
        normalized.append(obj)
    # Keep model order: prompts require image/reading order for list values.
    return normalized


def nested_review_tags(
    tags: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key = {(t["table_name_cn"], t["tag_name"]): t for t in tags}
    pairs = []
    for record in records:
        tag = by_key.get((record.get("table_name_cn"), record.get("tag_name")))
        if tag is None:
            continue
        if is_nested_review_tag(tag, record):
            pairs.append((tag, record))
    return pairs


def result_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".jsonl")


def presence_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".presence.jsonl")


def primary_presence_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".presence.primary.jsonl")


def extra_presence_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".presence.extra.jsonl")


def presence_kappa_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".presence.kappa.json")


def done_path_for_result(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".done")


def is_complete_result(path: Path, expected_lines: int) -> bool:
    if not done_path_for_result(path).exists():
        return False
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            count = 0
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != {"table_name_cn", "tag_name", "v"}:
                    return False
                count += 1
            return count == expected_lines
    except (OSError, json.JSONDecodeError):
        return False


def extract_chunk(
    config: dict[str, Any],
    image_media_type: str,
    image_base64: str,
    image_sha256: str,
    tag_chunk: list[dict[str, str]],
    mode: str,
    cache_dir: Path | None = None,
    chunk_index: int | None = None,
    chunk_total: int | None = None,
) -> list[dict[str, Any]]:
    api_type = config.get("api_type", "openai")
    cache_path = None
    if cache_dir is not None and config.get("use_api_cache", True):
        cache_path = cache_dir / f"{cache_key_for_request(config, image_sha256, mode, tag_chunk)}.json"
        cached_items = read_cached_items(cache_path)
        if cached_items is not None:
            log_local_cache_hit(mode, tag_chunk, chunk_index, chunk_total, cache_path)
            return cached_items

    if api_type == "anthropic":
        payload = build_anthropic_payload(
            config=config,
            image_media_type=image_media_type,
            image_base64=image_base64,
            tag_chunk=tag_chunk,
            mode=mode,
        )
        path = str(config.get("anthropic_path", "/messages"))
        headers = build_api_auth_headers(config, default_header_name="x-api-key")
        anthropic_version = str(config.get("anthropic_version", "2023-06-01")).strip()
        if anthropic_version:
            headers["anthropic-version"] = anthropic_version
    else:
        payload = build_openai_payload(
            config=config,
            image_data_url=f"data:{image_media_type};base64,{image_base64}",
            image_sha256=image_sha256,
            tag_chunk=tag_chunk,
            mode=mode,
        )
        path = str(config.get("openai_path", "/chat/completions"))
        headers = build_api_auth_headers(
            config,
            default_header_name="Authorization",
            default_prefix="Bearer",
        )

    max_retries = int(config.get("max_retries", 3))
    retry_round = 1
    while True:
        last_error: Exception | None = None
        last_content = ""
        for attempt in range(1, max_retries + 1):
            try:
                response = post_json(
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    path=path,
                    payload=payload,
                    timeout=int(config.get("timeout_seconds", 120)),
                    max_retries=max_retries,
                    headers=headers,
                )
                log_api_usage(mode, tag_chunk, chunk_index, chunk_total, response)
                content = extract_text_from_response(response, api_type)
                last_content = content
                parsed = parse_model_json(content)
                items = parsed.get("items", [])
                if not isinstance(items, list):
                    raise ValueError("Model JSON has no items list.")
                if cache_path is not None:
                    write_cached_items(cache_path, items)
                return items
            except Exception as e:  # noqa: BLE001
                if is_refusal_error(e):
                    raise with_request_context(
                        e,
                        request_mode=mode,
                        tag_chunk=tag_chunk,
                        chunk_index=chunk_index,
                        chunk_total=chunk_total,
                    ) from e
                if is_timeout_error(e):
                    log_timeout_skip(mode, tag_chunk, chunk_index, chunk_total, e)
                    return []
                if not isinstance(e, (json.JSONDecodeError, ValueError)):
                    raise add_request_context_to_error(
                        e,
                        request_mode=mode,
                        tag_chunk=tag_chunk,
                        chunk_index=chunk_index,
                        chunk_total=chunk_total,
                    ) from e
                last_error = e
                time.sleep(min(2**attempt, 20))

        snippet = last_content[:500].replace("\n", "\\n")
        final_error = RuntimeError(
            f"Model did not return valid JSON after {max_retries} attempts: "
            f"{last_error}; last_content={snippet}"
        )
        if is_refusal_error(final_error):
            raise with_request_context(
                final_error,
                request_mode=mode,
                tag_chunk=tag_chunk,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
            ) from final_error
        sleep_before_retrying_same_request(
            f"{final_error} (retry round {retry_round})."
        )
        retry_round += 1


def review_nested_tag(
    config: dict[str, Any],
    image_media_type: str,
    image_base64: str,
    image_sha256: str,
    tag: dict[str, Any],
    current_value: Any,
    cache_dir: Path | None = None,
    chunk_index: int | None = None,
    chunk_total: int | None = None,
) -> list[dict[str, Any]]:
    api_type = config.get("api_type", "openai")
    cache_path = None
    if cache_dir is not None and config.get("use_api_cache", True):
        cache_path = cache_dir / f"{cache_key_for_nested_review(config, image_sha256, tag, current_value)}.json"
        cached_items = read_cached_items(cache_path)
        if cached_items is not None:
            log_local_cache_hit("nested-review", [tag], chunk_index, chunk_total, cache_path)
            return cached_items

    if api_type == "anthropic":
        payload = build_anthropic_nested_review_payload(
            config=config,
            image_media_type=image_media_type,
            image_base64=image_base64,
            tag=tag,
            current_value=current_value,
        )
        path = str(config.get("anthropic_path", "/messages"))
        headers = build_api_auth_headers(config, default_header_name="x-api-key")
        anthropic_version = str(config.get("anthropic_version", "2023-06-01")).strip()
        if anthropic_version:
            headers["anthropic-version"] = anthropic_version
    else:
        payload = build_openai_nested_review_payload(
            config=config,
            image_data_url=f"data:{image_media_type};base64,{image_base64}",
            image_sha256=image_sha256,
            tag=tag,
            current_value=current_value,
        )
        path = str(config.get("openai_path", "/chat/completions"))
        headers = build_api_auth_headers(
            config,
            default_header_name="Authorization",
            default_prefix="Bearer",
        )

    max_retries = int(config.get("max_retries", 3))
    retry_round = 1
    while True:
        last_error: Exception | None = None
        last_content = ""
        for attempt in range(1, max_retries + 1):
            try:
                response = post_json(
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    path=path,
                    payload=payload,
                    timeout=int(config.get("timeout_seconds", 120)),
                    max_retries=max_retries,
                    headers=headers,
                )
                log_api_usage("nested-review", [tag], chunk_index, chunk_total, response)
                content = extract_text_from_response(response, api_type)
                last_content = content
                parsed = parse_model_json(content)
                items = parsed.get("items", [])
                if not isinstance(items, list):
                    raise ValueError("Model JSON has no items list.")
                if cache_path is not None:
                    write_cached_items(cache_path, items)
                return items
            except Exception as e:  # noqa: BLE001
                if is_refusal_error(e):
                    raise with_request_context(
                        e,
                        request_mode="nested-review",
                        tag_chunk=[tag],
                        chunk_index=chunk_index,
                        chunk_total=chunk_total,
                    ) from e
                if is_timeout_error(e):
                    log_timeout_skip("nested-review", [tag], chunk_index, chunk_total, e)
                    return []
                if not isinstance(e, (json.JSONDecodeError, ValueError)):
                    raise add_request_context_to_error(
                        e,
                        request_mode="nested-review",
                        tag_chunk=[tag],
                        chunk_index=chunk_index,
                        chunk_total=chunk_total,
                    ) from e
                last_error = e
                time.sleep(min(2**attempt, 20))

        snippet = last_content[:500].replace("\n", "\\n")
        final_error = RuntimeError(
            f"Model did not return valid nested-review JSON after {max_retries} attempts: "
            f"{last_error}; last_content={snippet}"
        )
        if is_refusal_error(final_error):
            raise with_request_context(
                final_error,
                request_mode="nested-review",
                tag_chunk=[tag],
                chunk_index=chunk_index,
                chunk_total=chunk_total,
            ) from final_error
        sleep_before_retrying_same_request(
            f"{final_error} (retry round {retry_round})."
        )
        retry_round += 1


def run_presence_phase(
    config: dict[str, Any],
    image_media_type: str,
    image_base64: str,
    image_sha256: str,
    tag_chunks: list[list[dict[str, str]]],
    tags: list[dict[str, str]],
    presence_records: list[dict[str, Any]],
    output_path: Path,
    cache_dir: Path,
    max_workers: int,
    desc: str,
    fallback_config_chain: list[tuple[Path, dict[str, Any]]] | None = None,
    config_path: Path | None = None,
) -> None:
    def extract_presence_chunk(
        tag_chunk: list[dict[str, str]],
        chunk_index: int,
        chunk_total: int,
    ) -> list[dict[str, Any]]:
        try:
            return extract_chunk(
                config,
                image_media_type,
                image_base64,
                image_sha256,
                tag_chunk,
                "presence",
                cache_dir,
                chunk_index,
                chunk_total,
            )
        except Exception as primary_error:  # noqa: BLE001
            if not fallback_config_chain:
                raise
            fallback_errors: list[str] = []
            previous_error: BaseException = primary_error
            fallback_total = len(fallback_config_chain)
            for fallback_index, (fallback_config_path, fallback_config) in enumerate(
                fallback_config_chain,
                start=1,
            ):
                print(
                    "PRESENCE_FALLBACK "
                    f"{request_label('presence', tag_chunk, chunk_index, chunk_total)} "
                    f"fallback={fallback_index}/{fallback_total} "
                    f"primary_config={config_path} "
                    f"fallback_config={fallback_config_path} "
                    f"previous_error={compact_text(str(previous_error), 1200)}",
                    flush=True,
                )
                try:
                    return extract_chunk(
                        fallback_config,
                        image_media_type,
                        image_base64,
                        image_sha256,
                        tag_chunk,
                        "presence",
                        cache_dir,
                        chunk_index,
                        chunk_total,
                    )
                except Exception as fallback_error:  # noqa: BLE001
                    fallback_errors.append(
                        f"{fallback_config_path}: {compact_text(str(fallback_error), 1200)}"
                    )
                    previous_error = fallback_error
            raise RuntimeError(
                "Stage 1 all fallback presence configs failed; "
                f"primary_error={primary_error}; "
                f"fallback_errors={' | '.join(fallback_errors)}"
            ) from previous_error

    if max_workers == 1:
        presence_iter = progress(
            tag_chunks,
            desc=desc,
            unit="chunk",
            leave=True,
            position=1,
        )
        chunk_total = len(tag_chunks)
        for chunk_index, tag_chunk in enumerate(presence_iter, start=1):
            items = extract_presence_chunk(tag_chunk, chunk_index, chunk_total)
            merge_presence_items(presence_records, items, tag_chunk)
            write_records(output_path, presence_records)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    extract_presence_chunk,
                    tag_chunk,
                    chunk_index,
                    len(tag_chunks),
                ): tag_chunk
                for chunk_index, tag_chunk in enumerate(tag_chunks, start=1)
            }
            future_iter = progress(
                concurrent.futures.as_completed(future_to_chunk),
                total=len(future_to_chunk),
                desc=desc,
                unit="chunk",
                leave=True,
                position=1,
            )
            for future in future_iter:
                merge_presence_items(presence_records, future.result(), future_to_chunk[future])
                write_records(output_path, presence_records)


def apply_nested_review_item(record: dict[str, Any], tag: dict[str, Any], item: dict[str, Any]) -> None:
    if item.get("has_value") is False:
        record["v"] = None
        return
    value = item.get("value")
    record["v"] = normalize_nested_value(
        value,
        nested_keys_from_tag(tag),
        identity_dimensions_for_tag(tag),
    )


def path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value if item is not None]


def resolve_config_path(path: Path, primary_config_path: Path) -> Path | None:
    if path.exists():
        return path
    if path.is_absolute():
        return None
    next_to_primary = primary_config_path.parent / path
    if next_to_primary.exists():
        return next_to_primary
    next_to_script = Path(__file__).resolve().parent / path
    if next_to_script.exists():
        return next_to_script
    return None


def resolve_bundle_path(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    next_to_script = Path(__file__).resolve().parent / path
    if next_to_script.exists():
        return next_to_script
    return path


def load_fallback_config_chain(
    paths: Any,
    primary_config_path: Path,
    *,
    require_credentials: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    configs: list[tuple[Path, dict[str, Any]]] = []
    for path in path_list(paths):
        resolved_path = resolve_config_path(path, primary_config_path)
        if resolved_path is None:
            continue
        configs.append(
            (
                resolved_path,
                load_config(resolved_path, require_credentials=require_credentials),
            )
        )
    return configs


def format_config_chain(configs: list[tuple[Path, dict[str, Any]]]) -> str:
    if not configs:
        return "disabled"
    return " -> ".join(str(path) for path, _config in configs)


def validate_openai_chat_endpoint(config: dict[str, Any], path: Path) -> None:
    if (
        config.get("api_type", "openai") == "openai"
        and config.get("endpoint", "chat_completions") != "chat_completions"
    ):
        raise SystemExit(
            "Only endpoint=chat_completions is implemented for api_type=openai "
            f"in {path}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--presence-config",
        default="extraction_config_1.yaml",
        type=Path,
        help="Config used for the first has_value filtering phase.",
    )
    parser.add_argument(
        "--presence-config-fallback",
        nargs="*",
        default=list(DEFAULT_FALLBACK_CONFIG_PATHS),
        type=Path,
        help=(
            "Optional fallback configs for Stage 1 presence detection, tried in order. "
            "When a primary presence chunk fails after its retries, the same chunk is retried with these configs."
        ),
    )
    parser.add_argument(
        "--presence-config-extra",
        default=DEFAULT_FALLBACK_CONFIG_PATHS[0],
        type=Path,
        help="Optional second config for independent has_value filtering.",
    )
    parser.add_argument(
        "--extract-config",
        default="extraction_config_2.yaml",
        type=Path,
        help="Config used for the second single-tag extraction phase.",
    )
    parser.add_argument(
        "--extract-config-fallback",
        nargs="*",
        default=list(DEFAULT_FALLBACK_CONFIG_PATHS),
        type=Path,
        help=(
            "Optional fallback configs for Stage 2/3 extraction, tried in order. "
            "When the primary extract config fails after its retries, the same tag/field is retried with these configs."
        ),
    )
    parser.add_argument("--tag-csv", default="tag-pool_乳腺癌_20260610.csv", type=Path)
    parser.add_argument("--remove-json", default="remove.json", type=Path)
    parser.add_argument(
        "--add-jsonl",
        default="add.jsonl",
        type=Path,
        help="Optional JSONL file with extra tags to append dynamically.",
    )
    parser.add_argument(
        "--manifest",
        default="normalized_images_乳腺癌_20260608_preserve/image_manifest.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing numbered images. Overrides --manifest when set.",
    )
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--start", type=int, help="First image_index to process, 1-3000.")
    parser.add_argument("--end", type=int, help="Last image_index to process, 1-3000.")
    parser.add_argument("--start-image-index", type=int)
    parser.add_argument("--end-image-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess images even when same-stem JSONL already exists.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Parallel model requests per image. Overrides max_workers in YAML.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help=(
            "Maximum pixels sent to the extraction model. Defaults to "
            "min(MAX_PIXELS, 1048576), with 1048576 as a hard cap."
        ),
    )
    parser.add_argument(
        "--cache-scope",
        choices=("per-image", "shared"),
        default="per-image",
        help=(
            "Local API cache scope. per-image uses an isolated cache directory for each image "
            "and clears it after the image completes; shared keeps the old image-dir-wide cache."
        ),
    )
    parser.add_argument(
        "--keep-image-cache",
        action="store_true",
        help="Keep per-image local API cache after each image completes for debugging.",
    )
    parser.add_argument(
        "--disable-nested-review",
        action="store_true",
        help="Compatibility flag. Nested review is always disabled in this vLLM LoRA bundle.",
    )
    parser.add_argument(
        "--disable-extra-presence",
        action="store_true",
        help="Use only --presence-config and skip the second has_value pass.",
    )
    parser.add_argument(
        "--enable-extra-presence",
        action="store_true",
        help="Run the optional second has_value pass from --presence-config-extra.",
    )
    parser.add_argument(
        "--error-image-dir",
        type=Path,
        help="Move images that trigger provider refusal/sensitive-word errors here and continue.",
    )
    parser.add_argument(
        "--rejected-tags-log",
        type=Path,
        help=(
            "JSONL path for per-tag refusal records. Defaults to "
            "--error-image-dir/rejected_tags.jsonl when --error-image-dir is set."
        ),
    )
    args = parser.parse_args()
    args.disable_nested_review = True
    image_max_pixels = effective_image_max_pixels(args.max_pixels)
    args.tag_csv = resolve_bundle_path(args.tag_csv)
    args.remove_json = resolve_bundle_path(args.remove_json)
    args.add_jsonl = resolve_bundle_path(args.add_jsonl)
    args.manifest = resolve_bundle_path(args.manifest)
    error_image_dir = args.error_image_dir.expanduser().resolve() if args.error_image_dir else None
    rejected_tags_log_path = (
        args.rejected_tags_log.expanduser().resolve()
        if args.rejected_tags_log
        else (error_image_dir / "rejected_tags.jsonl" if error_image_dir else None)
    )

    if args.config is not None:
        presence_config_path = args.config
        presence_fallback_config_paths = args.presence_config_fallback
        extract_config_path = args.config
        extract_fallback_config_paths = args.extract_config_fallback
        extra_presence_config_path = None
    else:
        presence_config_path = args.presence_config
        presence_fallback_config_paths = args.presence_config_fallback
        extract_config_path = args.extract_config
        extract_fallback_config_paths = args.extract_config_fallback
        extra_presence_config_path = (
            args.presence_config_extra
            if args.enable_extra_presence and not args.disable_extra_presence
            else None
        )

    presence_config = load_config(presence_config_path, require_credentials=not args.dry_run)
    presence_fallback_config_chain = load_fallback_config_chain(
        presence_fallback_config_paths,
        presence_config_path,
        require_credentials=not args.dry_run,
    )
    presence_config["image_max_pixels"] = image_max_pixels
    for _, fallback_config in presence_fallback_config_chain:
        fallback_config["image_max_pixels"] = image_max_pixels
    extra_presence_config = None
    if extra_presence_config_path is not None and extra_presence_config_path.exists():
        extra_presence_config = load_config(
            extra_presence_config_path,
            require_credentials=not args.dry_run,
        )
        extra_presence_config["image_max_pixels"] = image_max_pixels
    elif extra_presence_config_path is not None and not args.disable_extra_presence:
        extra_presence_config_path = None
    extract_config = load_config(extract_config_path, require_credentials=not args.dry_run)
    extract_fallback_config_chain = load_fallback_config_chain(
        extract_fallback_config_paths,
        extract_config_path,
        require_credentials=not args.dry_run,
    )
    extract_config["image_max_pixels"] = image_max_pixels
    for _, fallback_config in extract_fallback_config_chain:
        fallback_config["image_max_pixels"] = image_max_pixels
    remove_tag_names = load_remove_tag_names(args.remove_json)
    add_tags = load_add_tags(args.add_jsonl)
    all_tags = merge_add_tags(load_tags(args.tag_csv), add_tags)
    tags = merge_add_tags(load_tags(args.tag_csv, remove_tag_names=remove_tag_names), add_tags)
    manifest = (
        iter_image_dir(args.image_dir.resolve())
        if args.image_dir is not None
        else load_manifest(args.manifest)
    )

    start = args.start if args.start is not None else args.start_image_index
    end = args.end if args.end is not None else args.end_image_index
    if start is not None and not 1 <= start <= 3000:
        raise SystemExit("--start must be between 1 and 3000.")
    if end is not None and not 1 <= end <= 3000:
        raise SystemExit("--end must be between 1 and 3000.")
    if start is not None and end is not None and start > end:
        raise SystemExit("--start must be less than or equal to --end.")
    if start is not None:
        manifest = [r for r in manifest if int(r["image_index"]) >= start]
    if end is not None:
        manifest = [r for r in manifest if int(r["image_index"]) <= end]
    if args.limit_images is not None:
        manifest = manifest[: args.limit_images]

    tag_chunks = chunks(tags, int(presence_config.get("tags_per_call", 30)))
    extra_tag_chunks = (
        chunks(tags, int(extra_presence_config.get("tags_per_call", 30)))
        if extra_presence_config is not None
        else []
    )
    max_workers = args.max_workers or int(presence_config.get("max_workers", 1))
    if max_workers < 1:
        raise SystemExit("max_workers must be >= 1.")
    print(
        f"images={len(manifest)} tags={len(tags)} "
        f"removed_tags={len(all_tags) - len(tags)} add_tags={len(add_tags)} "
        f"chunks_per_image={len(tag_chunks)}"
    )
    if extra_presence_config is not None:
        print(f"extra_presence_chunks_per_image={len(extra_tag_chunks)}")
    print(f"max_workers={max_workers}")
    print(f"image_max_pixels={image_max_pixels}")
    print(
        f"cache_scope={args.cache_scope}"
        f"{' keep_image_cache=true' if args.keep_image_cache else ''}"
    )
    print(f"presence_config={presence_config_path}")
    print(
        "presence_config_fallback="
        f"{format_config_chain(presence_fallback_config_chain)}"
    )
    print(
        "presence_config_extra="
        f"{extra_presence_config_path if extra_presence_config is not None else 'disabled'}"
    )
    print(f"extract_config={extract_config_path}")
    print(
        "extract_config_fallback="
        f"{format_config_chain(extract_fallback_config_chain)}"
    )
    print(f"nested_review={'disabled' if args.disable_nested_review else 'enabled'}")
    print(f"error_image_dir={error_image_dir if error_image_dir else 'disabled'}")
    print(
        "rejected_tags_log="
        f"{rejected_tags_log_path if rejected_tags_log_path else 'disabled'}"
    )
    print("results=same directory as each image, same stem .jsonl")

    if args.dry_run:
        return

    validate_openai_chat_endpoint(presence_config, presence_config_path)
    if extra_presence_config is not None:
        validate_openai_chat_endpoint(extra_presence_config, extra_presence_config_path)
    for fallback_config_path, fallback_config in presence_fallback_config_chain:
        validate_openai_chat_endpoint(fallback_config, fallback_config_path)
    validate_openai_chat_endpoint(extract_config, extract_config_path)
    for fallback_config_path, fallback_config in extract_fallback_config_chain:
        validate_openai_chat_endpoint(fallback_config, fallback_config_path)

    def prepare_image_retry_after_refusal(
        image_path: Path,
        image_row: dict[str, Any],
        stage: str,
        error: BaseException,
        cache_dir: Path,
    ) -> bool:
        nonlocal tags, tag_chunks, extra_tag_chunks
        if not is_refusal_error(error):
            return False
        if rejected_tags_log_path is not None:
            write_rejected_tags_log(
                rejected_tags_log_path,
                image_path,
                image_row,
                stage,
                error,
            )
        cleanup_summary = cleanup_current_image_records_and_cache(
            image_path,
            cache_dir,
        )
        random.shuffle(tags)
        tag_chunks = chunks(tags, int(presence_config.get("tags_per_call", 30)))
        extra_tag_chunks = (
            chunks(tags, int(extra_presence_config.get("tags_per_call", 30)))
            if extra_presence_config is not None
            else []
        )
        print(
            "REFUSAL_RETRY "
            f"{image_row.get('normalized_name', image_path.name)} "
            f"stage={stage}{request_context_summary(error)} "
            f"deleted_records={len(cleanup_summary['deleted_records'])} "
            f"cache_cleared={cleanup_summary['cache_cleared']} "
            f"sleep={REFUSAL_RETRY_SLEEP_SECONDS}s",
            flush=True,
        )
        time.sleep(REFUSAL_RETRY_SLEEP_SECONDS)
        return True

    image_iter = progress(manifest, desc="images", unit="image", position=0)
    for image_row in image_iter:
        while True:
            image_path = Path(image_row["normalized_path"])
            result_path = result_path_for_image(image_path)
            presence_path = presence_path_for_image(image_path)
            primary_presence_path = primary_presence_path_for_image(image_path)
            extra_presence_path = extra_presence_path_for_image(image_path)
            kappa_path = presence_kappa_path_for_image(image_path)
            done_path = done_path_for_result(result_path)

            if not args.overwrite and is_complete_result(result_path, len(tags)):
                if tqdm is None:
                    print(f"skip complete image {image_row['normalized_name']} -> {result_path}")
                image_sha256 = image_row.get("source_sha256", "")
                cache_dir = cache_dir_for_image(
                    image_path,
                    image_sha256,
                    args.cache_scope,
                )
                if args.cache_scope == "per-image" and not args.keep_image_cache:
                    clear_image_cache(cache_dir, image_row["normalized_name"])
                break

            if args.overwrite and done_path.exists():
                done_path.unlink()

            records = default_records(tags)
            write_records(result_path, records)
            primary_presence_records = default_presence_records(tags)
            write_records(primary_presence_path, primary_presence_records)
            extra_presence_records = default_presence_records(tags)
            if extra_presence_config is not None:
                write_records(extra_presence_path, extra_presence_records)
            presence_records = default_presence_records(tags)
            write_records(presence_path, presence_records)
            image_media_type, image_base64 = image_to_base64(
                image_path,
                max_pixels=image_max_pixels,
            )
            image_sha256 = image_row.get("source_sha256", "")
            cache_dir = cache_dir_for_image(
                image_path,
                image_sha256,
                args.cache_scope,
            )

            # Phase 1: chunk-level presence detection only. This writes an
            # intermediate JSONL so partial progress is visible if later calls fail.
            image_stem = Path(image_row["normalized_name"]).stem
            try:
                run_presence_phase(
                    presence_config,
                    image_media_type,
                    image_base64,
                    image_sha256,
                    tag_chunks,
                    tags,
                    primary_presence_records,
                    primary_presence_path,
                    cache_dir,
                    max_workers,
                    f"image {image_stem} presence-primary",
                    fallback_config_chain=presence_fallback_config_chain,
                    config_path=presence_config_path,
                )
            except Exception as e:  # noqa: BLE001
                if error_image_dir is not None and handle_refusal_image(
                    image_path,
                    image_row,
                    error_image_dir,
                    "presence-primary",
                    e,
                    rejected_tags_log_path,
                ):
                    break
                if prepare_image_retry_after_refusal(
                    image_path,
                    image_row,
                    "presence-primary",
                    e,
                    cache_dir,
                ):
                    continue
                raise
            if extra_presence_config is not None:
                try:
                    run_presence_phase(
                        extra_presence_config,
                        image_media_type,
                        image_base64,
                        image_sha256,
                        extra_tag_chunks,
                        tags,
                        extra_presence_records,
                        extra_presence_path,
                        cache_dir,
                        max_workers,
                        f"image {image_stem} presence-extra",
                    )
                except Exception as e:  # noqa: BLE001
                    if error_image_dir is not None and handle_refusal_image(
                        image_path,
                        image_row,
                        error_image_dir,
                        "presence-extra",
                        e,
                        rejected_tags_log_path,
                    ):
                        break
                    if prepare_image_retry_after_refusal(
                        image_path,
                        image_row,
                        "presence-extra",
                        e,
                        cache_dir,
                    ):
                        continue
                    raise
                presence_records = merge_presence_union(
                    tags,
                    primary_presence_records,
                    extra_presence_records,
                )
                kappa_summary = compute_presence_kappa(
                    tags,
                    primary_presence_records,
                    extra_presence_records,
                )
                kappa_summary.update(
                    {
                        "image_name": image_row["normalized_name"],
                        "image_path": str(image_path),
                        "primary_config": str(presence_config_path),
                        "extra_config": str(extra_presence_config_path),
                        "union_presence_path": str(presence_path),
                        "primary_presence_path": str(primary_presence_path),
                        "extra_presence_path": str(extra_presence_path),
                    }
                )
                write_json_file(kappa_path, kappa_summary)
            else:
                presence_records = primary_presence_records
            write_records(presence_path, presence_records)

            candidate_tags = tags_from_presence(tags, presence_records)

            # Phase 2: single-tag extraction for tags that passed the coarse pass.
            # The model must judge has_value again; false keeps v as null.
            single_tag_chunks = [[tag] for tag in candidate_tags]

            def extract_stage2_tag(
                tag_chunk: list[dict[str, str]],
                chunk_index: int,
                chunk_total: int,
            ) -> list[dict[str, Any]]:
                try:
                    return extract_chunk(
                        extract_config,
                        image_media_type,
                        image_base64,
                        image_sha256,
                        tag_chunk,
                        "extract",
                        cache_dir,
                        chunk_index,
                        chunk_total,
                    )
                except Exception as primary_error:  # noqa: BLE001
                    if not extract_fallback_config_chain:
                        raise
                    fallback_errors: list[str] = []
                    previous_error: BaseException = primary_error
                    fallback_total = len(extract_fallback_config_chain)
                    for fallback_index, (fallback_config_path, fallback_config) in enumerate(
                        extract_fallback_config_chain,
                        start=1,
                    ):
                        print(
                            "EXTRACT_FALLBACK "
                            f"{request_label('extract', tag_chunk, chunk_index, chunk_total)} "
                            f"fallback={fallback_index}/{fallback_total} "
                            f"primary_config={extract_config_path} "
                            f"fallback_config={fallback_config_path} "
                            f"previous_error={compact_text(str(previous_error), 1200)}",
                            flush=True,
                        )
                        try:
                            return extract_chunk(
                                fallback_config,
                                image_media_type,
                                image_base64,
                                image_sha256,
                                tag_chunk,
                                "extract",
                                cache_dir,
                                chunk_index,
                                chunk_total,
                            )
                        except Exception as fallback_error:  # noqa: BLE001
                            fallback_errors.append(
                                f"{fallback_config_path}: {compact_text(str(fallback_error), 1200)}"
                            )
                            previous_error = fallback_error
                    raise RuntimeError(
                        "Stage 2 all fallback extraction configs failed; "
                        f"primary_error={primary_error}; "
                        f"fallback_errors={' | '.join(fallback_errors)}"
                    ) from previous_error

            def review_stage3_tag(
                tag: dict[str, Any],
                current_value: Any,
                chunk_index: int,
                chunk_total: int,
            ) -> list[dict[str, Any]]:
                try:
                    return review_nested_tag(
                        extract_config,
                        image_media_type,
                        image_base64,
                        image_sha256,
                        tag,
                        current_value,
                        cache_dir,
                        chunk_index,
                        chunk_total,
                    )
                except Exception as primary_error:  # noqa: BLE001
                    if not extract_fallback_config_chain:
                        raise
                    fallback_errors: list[str] = []
                    previous_error: BaseException = primary_error
                    fallback_total = len(extract_fallback_config_chain)
                    for fallback_index, (fallback_config_path, fallback_config) in enumerate(
                        extract_fallback_config_chain,
                        start=1,
                    ):
                        print(
                            "NESTED_REVIEW_FALLBACK "
                            f"{request_label('nested-review', [tag], chunk_index, chunk_total)} "
                            f"fallback={fallback_index}/{fallback_total} "
                            f"primary_config={extract_config_path} "
                            f"fallback_config={fallback_config_path} "
                            f"previous_error={compact_text(str(previous_error), 1200)}",
                            flush=True,
                        )
                        try:
                            return review_nested_tag(
                                fallback_config,
                                image_media_type,
                                image_base64,
                                image_sha256,
                                tag,
                                current_value,
                                cache_dir,
                                chunk_index,
                                chunk_total,
                            )
                        except Exception as fallback_error:  # noqa: BLE001
                            fallback_errors.append(
                                f"{fallback_config_path}: {compact_text(str(fallback_error), 1200)}"
                            )
                            previous_error = fallback_error
                    raise RuntimeError(
                        "Stage 3 all fallback nested-review configs failed; "
                        f"primary_error={primary_error}; "
                        f"fallback_errors={' | '.join(fallback_errors)}"
                    ) from previous_error

            try:
                if max_workers == 1:
                    extract_iter = progress(
                        single_tag_chunks,
                        desc=f"image {Path(image_row['normalized_name']).stem} extract",
                        unit="tag",
                        leave=True,
                        position=2,
                    )
                    chunk_total = len(single_tag_chunks)
                    for chunk_index, tag_chunk in enumerate(extract_iter, start=1):
                        items = extract_stage2_tag(tag_chunk, chunk_index, chunk_total)
                        merge_items(records, items, tag_chunk)
                        write_records(result_path, records)
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_chunk = {
                            executor.submit(
                                extract_stage2_tag,
                                tag_chunk,
                                chunk_index,
                                len(single_tag_chunks),
                            ): tag_chunk
                            for chunk_index, tag_chunk in enumerate(single_tag_chunks, start=1)
                        }
                        future_iter = progress(
                            concurrent.futures.as_completed(future_to_chunk),
                            total=len(future_to_chunk),
                            desc=f"image {Path(image_row['normalized_name']).stem} extract",
                            unit="tag",
                            leave=True,
                            position=2,
                        )
                        for future in future_iter:
                            merge_items(records, future.result(), future_to_chunk[future])
                            write_records(result_path, records)
            except Exception as e:  # noqa: BLE001
                if error_image_dir is not None and handle_refusal_image(
                    image_path,
                    image_row,
                    error_image_dir,
                    "extract",
                    e,
                    rejected_tags_log_path,
                ):
                    break
                if prepare_image_retry_after_refusal(
                    image_path,
                    image_row,
                    "extract",
                    e,
                    cache_dir,
                ):
                    continue
                raise

            if not args.disable_nested_review:
                review_pairs = nested_review_tags(tags, records)
                if review_pairs:
                    try:
                        if max_workers == 1:
                            review_iter = progress(
                                review_pairs,
                                desc=f"image {Path(image_row['normalized_name']).stem} nested-review",
                                unit="field",
                                leave=True,
                                position=3,
                            )
                            chunk_total = len(review_pairs)
                            for chunk_index, (tag, record) in enumerate(review_iter, start=1):
                                items = review_stage3_tag(
                                    tag,
                                    record.get("v"),
                                    chunk_index,
                                    chunk_total,
                                )
                                for item in items:
                                    if item_matches_tag(item, tag):
                                        apply_nested_review_item(record, tag, item)
                                        break
                                write_records(result_path, records)
                        else:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                future_to_pair = {
                                    executor.submit(
                                        review_stage3_tag,
                                        tag,
                                        record.get("v"),
                                        chunk_index,
                                        len(review_pairs),
                                    ): (tag, record)
                                    for chunk_index, (tag, record) in enumerate(review_pairs, start=1)
                                }
                                review_iter = progress(
                                    concurrent.futures.as_completed(future_to_pair),
                                    total=len(future_to_pair),
                                    desc=f"image {Path(image_row['normalized_name']).stem} nested-review",
                                    unit="field",
                                    leave=True,
                                    position=3,
                                )
                                for future in review_iter:
                                    tag, record = future_to_pair[future]
                                    items = future.result()
                                    for item in items:
                                        if item_matches_tag(item, tag):
                                            apply_nested_review_item(record, tag, item)
                                            break
                                    write_records(result_path, records)
                    except Exception as e:  # noqa: BLE001
                        if error_image_dir is not None and handle_refusal_image(
                            image_path,
                            image_row,
                            error_image_dir,
                            "nested-review",
                            e,
                            rejected_tags_log_path,
                        ):
                            break
                        if prepare_image_retry_after_refusal(
                            image_path,
                            image_row,
                            "nested-review",
                            e,
                            cache_dir,
                        ):
                            continue
                        raise

            done_path.write_text("done\n", encoding="utf-8")
            if args.cache_scope == "per-image" and not args.keep_image_cache:
                clear_image_cache(cache_dir, image_row["normalized_name"])
            if tqdm is None:
                print(f"done image {image_row['normalized_name']} -> {result_path}")
            break

    print(
        "CACHE_SUMMARY "
        f"local_hits={CACHE_STATS['local_hits']} "
        f"api_calls={CACHE_STATS['api_calls']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
