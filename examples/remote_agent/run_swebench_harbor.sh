#!/usr/bin/env bash
# Run Slime training with SWE-bench verified dataset using Harbor remote agent.
#
# Usage:
#   1. First convert the dataset:
#      python convert_swebench_to_prompts.py \
#        --input swe-bench-verified.jsonl \
#        --output prompts.jsonl \
#        --task-root /data/tasks \
#        --create-task-dirs
#
#   2. Set environment variables and run:
#      export LOCAL_IP=10.0.30.11  # IP reachable from Harbor containers
#      export HARBOR_SERVER_URL=http://harbor-server:8080
#      export HF_CHECKPOINT=/path/to/Qwen2.5-7B-Instruct
#      bash run_swebench_harbor.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# External-reachable IP for the Harbor agent to reach the proxy
export LOCAL_IP="${LOCAL_IP:-10.0.30.11}"

# Harbor server
HARBOR_SERVER_URL="${HARBOR_SERVER_URL:-http://harbor-server:8080}"

# Agent configuration
HARBOR_AGENT_NAME="${HARBOR_AGENT_NAME:-swe-agent}"
HARBOR_MODEL_NAME="${HARBOR_MODEL_NAME:-openai/qwen-max}"

# Task data directory template (must match convert_swebench_to_prompts.py --task-root)
HARBOR_TASK_PATH_TEMPLATE="${HARBOR_TASK_PATH_TEMPLATE:-/data/tasks/{instance_id}}"

# Checkpoint
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/Qwen2.5-7B-Instruct}"

# Prompt data (converted from swe-bench-verified.jsonl)
PROMPT_DATA="${PROMPT_DATA:-./prompts.jsonl}"

# ---------------------------------------------------------------------------
# Training command
# ---------------------------------------------------------------------------

python ../../train_remote_agent.py \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-server-url "$HARBOR_SERVER_URL" \
  --harbor-agent-name "$HARBOR_AGENT_NAME" \
  --harbor-model-name "$HARBOR_MODEL_NAME" \
  --harbor-task-path-template "$HARBOR_TASK_PATH_TEMPLATE" \
  --harbor-adapter-public-host "$LOCAL_IP" \
  --harbor-max-retries 3 \
  --harbor-agent-kwargs '{"total_cost_limit": 0, "per_instance_cost_limit": 0}' \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --rollout-max-response-len 4096 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --num-rollout 10 \
  --prompt-data "$PROMPT_DATA" \
  --rollout-global-dataset \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --rollout-temperature 1.0 \
  --rollout-top-p 1.0 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
  --rollout-num-gpus-per-engine 1 \
  --global-batch-size 1 \
  --micro-batch-size 1 \
  --advantage-estimator grpo \
  --normalize-advantages \
  --seed 42 \
  --use-wandb \
  --wandb-project slime-swebench \
  "$@"
