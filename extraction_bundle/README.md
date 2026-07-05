# Haixin Qwen3-VL LoRA Extraction Bundle

This bundle runs the same Stage 1 and Stage 2 extraction prompts as the API
version, but defaults to the local vLLM OpenAI-compatible server.

Defaults:

- `VLLM_BASE_URL`: `http://127.0.0.1:22002/v1`
- `VLLM_MODEL`: `haixin_stage12`
- `VLLM_API_KEY`: `EMPTY`
- `MAX_PIXELS`: `1048576`, with `1048576` as a hard cap. Smaller values are allowed.

Example:

```bash
python extract_image_tags.py \
  --image-dir /path/to/images \
  --max-workers 1
```

Override the deployed model if needed:

```bash
VLLM_BASE_URL=http://127.0.0.1:22002/v1 \
VLLM_MODEL=haixin_stage12 \
MAX_PIXELS=1048576 \
python extract_image_tags.py \
  --image-dir /path/to/images \
  --max-workers 1
```

Notes:

- Stage 3 nested review is disabled in this bundle.
- API endpoint and model settings are taken from `VLLM_BASE_URL`,
  `VLLM_MODEL`, and `VLLM_API_KEY`; old YAML API credentials are ignored.
- Images sent to the model are resized in memory when needed so that
  `width * height <= min(MAX_PIXELS, 1048576)`. Original image files are not
  modified.
- Result files are still written beside each image as same-stem `.jsonl`, plus
  the existing presence intermediate files.
- `tag-pool_乳腺癌_20260610.csv`, `add.jsonl`, and `remove.json` are loaded from
  this bundle when relative paths are used.
