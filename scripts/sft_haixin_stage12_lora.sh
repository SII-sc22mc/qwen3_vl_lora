#!/bin/bash
set -euo pipefail

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# Paths
llm=${MODEL_PATH:-/inspire/hdd/global_user/chaimingxu-240108540141/models/Qwen3-VL-8B-Instruct}
datasets=${DATASET_USE:-haixin_stage12}
output_dir=${OUTPUT_DIR:-/inspire/hdd/global_user/chaimingxu-240108540141/haixin/qwen3_vl_lora/outputs/haixin_stage12_lora}

# Training hyperparameters
lr=${LR:-1e-5}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
epochs=${EPOCHS:-2}
save_steps=${SAVE_STEPS:-500}
save_total_limit=${SAVE_TOTAL_LIMIT:-3}
logging_steps=${LOGGING_STEPS:-2}
model_max_length=${MODEL_MAX_LENGTH:-8192}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-4}
min_pixels=${MIN_PIXELS:-784}
max_pixels=${MAX_PIXELS:-50176}
report_to=${REPORT_TO:-none}
run_name=${RUN_NAME:-haixin_stage12_lora}

# Official training entry and DeepSpeed config
entry_file=qwenvl/train/train_qwen.py
deepspeed_config=${DEEPSPEED_CONFIG:-./scripts/zero2.json}

echo "model=${llm}"
echo "dataset_use=${datasets}"
echo "annotation=/inspire/hdd/global_user/chaimingxu-240108540141/haixin/label/haixin_stage12_single_image.json"
echo "output_dir=${output_dir}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "batch_size=${batch_size}"
echo "grad_accum_steps=${grad_accum_steps}"
echo "epochs=${epochs}"
echo "model_max_length=${model_max_length}"
echo "min_pixels=${min_pixels}"
echo "max_pixels=${max_pixels}"

args="
    --deepspeed ${deepspeed_config} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --lora_enable True \
    --output_dir ${output_dir} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps ${logging_steps} \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${dataloader_num_workers} \
    --run_name ${run_name} \
    --report_to ${report_to}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${entry_file} ${args}
