#!/usr/bin/env bash
# Example: Run Slime training with Harbor local trial mode.
#
# In local mode, the Harbor Trial runs directly in the Python process —
# no remote Harbor server or Docker is required.  The TokenProxy still
# captures LLM tokens for training.
#
# Usage:
#   bash examples/remote_agent/harbor_local_trial.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

# Agent configuration (must be importable)
HARBOR_AGENT_NAME="${HARBOR_AGENT_NAME:-swe-agent}"
HARBOR_MODEL_NAME="${HARBOR_MODEL_NAME:-openai/qwen-max}"

# Task data directory template
HARBOR_TASK_PATH_TEMPLATE="${HARBOR_TASK_PATH_TEMPLATE:-/var/model-dataset/swe-bench-verified/{instance_id}}"

# Checkpoint — must match the architecture of the SGLang engines
HF_CHECKPOINT="${HF_CHECKPOINT:-/var/model/Qwen2.5-7B-Instruct}"

# Prompt data
PROMPT_DATA="${PROMPT_DATA:-/root/slime/examples/remote_agent/swe-bench-verified.jsonl}"

# Environment kwargs (passed to the Harbor environment constructor)
# For K8s-based environments, also set --harbor-env-import-path accordingly.
REMOTE_AGENT_ENVIRONMENT_KWARGS="${REMOTE_AGENT_ENVIRONMENT_KWARGS:-{\"namespace\":\"default\",\"image_pull_secret\":\"acr-registry\",\"use_buildkit\":\"true\",\"use_sandbox_claim\":\"true\",\"sandboxset_replicas\":\"1\",\"sandbox_labels\":\"{\\\"alibabacloud.com/acs\\\":\"true\"}\",\"buildkit_address\":\"tcp://buildkitd:1234\",\"build_job_namespace\":\"default\",\"registry\":\"registry.cn-hongkong.aliyuncs.com/swe-bench\",\"kubeconfig\":\"/root/slime/examples/remote_agent/kubeconfig\"}}"

# ---------------------------------------------------------------------------
# Training command
# ---------------------------------------------------------------------------
LOCAL_IP="10.0.30.17" \
python train_remote_agent.py \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-use-local-trial \
  --harbor-agent-name "$HARBOR_AGENT_NAME" \
  --harbor-model-name "$HARBOR_MODEL_NAME" \
  --harbor-task-path-template "$HARBOR_TASK_PATH_TEMPLATE" \
  --harbor-agent-kwargs '{"total_cost_limit": 0, "per_instance_cost_limit": 0}' \
  --harbor-env-kwargs "$REMOTE_AGENT_ENVIRONMENT_KWARGS" \
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
  "$@"
