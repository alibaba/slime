#!/bin/bash
# Convert an HF checkpoint into a Megatron torch_dist checkpoint for --ref-load.
# Override the paths for your model:
#   MODEL_PRESET  scripts/models/<preset>.sh (defines MODEL_ARGS)
#   HF_CKPT       source HF checkpoint (config + tokenizer + weights)
#   SAVE          output torch_dist directory (pass as --ref-load to run_swebench.sh)
set -euo pipefail

MODEL_PRESET="${MODEL_PRESET:-qwen2.5-0.5B}"
HF_CKPT="${HF_CKPT:-/path/to/Qwen2.5-0.5B-Instruct}"
SAVE="${SAVE:-/path/to/Qwen2.5-0.5B_torch_dist}"

source "$(pwd)/scripts/models/${MODEL_PRESET}.sh"
PYTHONPATH="${MEGATRON_PATH:-/root/Megatron-LM}" python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CKPT" \
    --save "$SAVE"
