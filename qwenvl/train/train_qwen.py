# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import time
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None
PROFILE_STEPS = os.environ.get("HAIXIN_PROFILE_STEPS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROFILE_EVERY = max(1, int(os.environ.get("HAIXIN_PROFILE_EVERY", "1")))


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _profile_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _profile_to_floats(profile):
    if not isinstance(profile, dict):
        return {}
    return {key: _profile_value(value) for key, value in profile.items()}


def _module_by_path(root, path):
    module = root
    for part in path.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module


def _find_profile_module(root, target):
    candidate_paths = {
        "visual": [
            "module.base_model.model.model.visual",
            "module.base_model.model.visual",
            "module.model.visual",
            "base_model.model.model.visual",
            "base_model.model.visual",
            "model.visual",
            "visual",
        ],
        "language_model": [
            "module.base_model.model.model.language_model",
            "module.base_model.model.language_model",
            "module.model.language_model",
            "base_model.model.model.language_model",
            "base_model.model.language_model",
            "model.language_model",
            "language_model",
        ],
    }[target]
    for path in candidate_paths:
        module = _module_by_path(root, path)
        if module is not None and hasattr(module, "forward"):
            return module, path
    if hasattr(root, "named_modules"):
        matches = []
        for name, module in root.named_modules():
            if name.endswith(target) and hasattr(module, "forward"):
                matches.append((name.count("."), name, module))
        if matches:
            _, name, module = sorted(matches)[0]
            return module, name
    return None, ""


class HaixinProfilingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._haixin_last_train_step_end = time.perf_counter()

    def training_step(self, model, inputs, *args, **kwargs):
        profile = inputs.pop("_haixin_profile", None)
        before_train_step = time.perf_counter()
        between_steps = before_train_step - self._haixin_last_train_step_end
        step_profile = {
            "prepare_inputs": 0.0,
            "compute_loss_total": 0.0,
            "model_forward": 0.0,
            "backward": 0.0,
        }

        original_prepare_inputs = self._prepare_inputs
        original_compute_loss = self.compute_loss
        original_backward = self.accelerator.backward
        original_forward = model.forward
        visual_module, visual_module_name = _find_profile_module(model, "visual")
        language_module, language_module_name = _find_profile_module(
            model, "language_model"
        )
        original_visual_forward = (
            visual_module.forward if visual_module is not None else None
        )
        original_language_forward = (
            language_module.forward if language_module is not None else None
        )

        def sync_cuda():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def wrapped_prepare_inputs(step_inputs):
            sync_cuda()
            start = time.perf_counter()
            result = original_prepare_inputs(step_inputs)
            sync_cuda()
            step_profile["prepare_inputs"] += time.perf_counter() - start
            return result

        def wrapped_forward(*forward_args, **forward_kwargs):
            sync_cuda()
            start = time.perf_counter()
            result = original_forward(*forward_args, **forward_kwargs)
            sync_cuda()
            step_profile["model_forward"] += time.perf_counter() - start
            return result

        def wrapped_compute_loss(*compute_args, **compute_kwargs):
            sync_cuda()
            start = time.perf_counter()
            result = original_compute_loss(*compute_args, **compute_kwargs)
            sync_cuda()
            step_profile["compute_loss_total"] += time.perf_counter() - start
            return result

        def wrapped_backward(*backward_args, **backward_kwargs):
            sync_cuda()
            start = time.perf_counter()
            result = original_backward(*backward_args, **backward_kwargs)
            sync_cuda()
            step_profile["backward"] += time.perf_counter() - start
            return result

        def wrapped_visual_forward(*visual_args, **visual_kwargs):
            sync_cuda()
            start = time.perf_counter()
            result = original_visual_forward(*visual_args, **visual_kwargs)
            sync_cuda()
            step_profile["visual_forward"] = (
                step_profile.get("visual_forward", 0.0)
                + time.perf_counter()
                - start
            )
            return result

        def wrapped_language_forward(*language_args, **language_kwargs):
            sync_cuda()
            start = time.perf_counter()
            result = original_language_forward(*language_args, **language_kwargs)
            sync_cuda()
            step_profile["language_forward"] = (
                step_profile.get("language_forward", 0.0)
                + time.perf_counter()
                - start
            )
            return result

        self._prepare_inputs = wrapped_prepare_inputs
        self.compute_loss = wrapped_compute_loss
        self.accelerator.backward = wrapped_backward
        model.forward = wrapped_forward
        if visual_module is not None:
            visual_module.forward = wrapped_visual_forward
        if language_module is not None:
            language_module.forward = wrapped_language_forward
        sync_cuda()
        train_step_start = time.perf_counter()
        try:
            loss = super().training_step(model, inputs, *args, **kwargs)
            sync_cuda()
            train_step_time = time.perf_counter() - train_step_start
        finally:
            self._prepare_inputs = original_prepare_inputs
            self.compute_loss = original_compute_loss
            self.accelerator.backward = original_backward
            model.forward = original_forward
            if visual_module is not None:
                visual_module.forward = original_visual_forward
            if language_module is not None:
                language_module.forward = original_language_forward
        self._haixin_last_train_step_end = time.perf_counter()

        step = self.state.global_step + 1
        if self.is_world_process_zero() and step % PROFILE_EVERY == 0:
            values = _profile_to_floats(profile)
            compute_loss_overhead = max(
                0.0,
                step_profile["compute_loss_total"] - step_profile["model_forward"],
            )
            model_forward_other = max(
                0.0,
                step_profile["model_forward"]
                - step_profile.get("visual_forward", 0.0)
                - step_profile.get("language_forward", 0.0),
            )
            train_step_other = max(
                0.0,
                train_step_time
                - step_profile["prepare_inputs"]
                - step_profile["compute_loss_total"]
                - step_profile["backward"],
            )
            print(
                "[haixin-profile] "
                f"step={step} "
                f"between_steps={between_steps:.3f}s "
                f"train_forward_backward={train_step_time:.3f}s "
                f"prepare_inputs={step_profile['prepare_inputs']:.3f}s "
                f"compute_loss_total={step_profile['compute_loss_total']:.3f}s "
                f"model_forward={step_profile['model_forward']:.3f}s "
                f"visual_forward={step_profile.get('visual_forward', 0.0):.3f}s "
                f"language_forward={step_profile.get('language_forward', 0.0):.3f}s "
                f"model_forward_other={model_forward_other:.3f}s "
                f"compute_loss_overhead={compute_loss_overhead:.3f}s "
                f"backward={step_profile['backward']:.3f}s "
                f"train_step_other={train_step_other:.3f}s "
                f"getitem_total={values.get('getitem_total', 0.0):.3f}s "
                f"apply_chat_template={values.get('apply_chat_template', 0.0):.3f}s "
                f"build_messages={values.get('build_messages', 0.0):.3f}s "
                f"build_labels={values.get('build_labels', 0.0):.3f}s "
                f"rope_index={values.get('rope_index', 0.0):.3f}s "
                f"debug_decode={values.get('debug_decode', 0.0):.3f}s "
                f"collator_total={values.get('collator_total', 0.0):.3f}s "
                f"batch_items={values.get('batch_items', 0.0):.0f} "
                f"batch_seq_len={values.get('batch_seq_len', 0.0):.0f} "
                f"image_tokens={values.get('image_tokens_sum', 0.0):.0f} "
                f"images={values.get('images_sum', 0.0):.0f} "
                f"batch_pixel_rows={values.get('batch_pixel_rows', 0.0):.0f} "
                f"visual_module={visual_module_name or 'not_found'} "
                f"language_module={language_module_name or 'not_found'}",
                flush=True,
            )
        return loss


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer_cls = HaixinProfilingTrainer if PROFILE_STEPS else Trainer
    if PROFILE_STEPS and training_args.local_rank in (-1, 0):
        print(f"HAIXIN_PROFILE_STEPS enabled; printing every {PROFILE_EVERY} step(s).")
    trainer = trainer_cls(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
