#!/bin/bash

source $(pwd)/scripts/models/qwen2.5-7B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
    --hf-checkpoint /var/model/Qwen2.5-7B-Instruct \
    --save /root/Qwen2.5-7B-Instruct
