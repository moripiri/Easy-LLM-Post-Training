#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

model_path=Qwen/Qwen3-0.6B

dataset_path=$SCRIPT_DIR/../data/sft_en_demo.json

output_dir=$SCRIPT_DIR/output_dir

python_bin=python
if [[ -x "$SCRIPT_DIR/../.venv/bin/python" ]]; then
  python_bin="$SCRIPT_DIR/../.venv/bin/python"
fi

# single gpu training
"$python_bin" "$SCRIPT_DIR/sft_train.py" \
  --model_path $model_path \
  --train_json $dataset_path \
  --output_dir $output_dir \
  --device mps \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --max_length 256

# # multi-gpu training
# torchrun --nproc_per_node=2 "$SCRIPT_DIR/sft_train_ngpu.py" \
#   --model_path $model_path \
#   --train_json $dataset_path \
#   --output_dir $output_dir \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 10 \
#   --max_length 4096
