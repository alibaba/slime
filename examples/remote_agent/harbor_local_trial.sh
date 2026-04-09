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

# Megatron-format checkpoint directory (for training)
# This is where converted weights are stored/loaded
MEGATRON_CHECKPOINT="${MEGATRON_CHECKPOINT:-/root/Qwen2.5-7B-Instruct-megatron}"

# Prompt data
PROMPT_DATA="${PROMPT_DATA:-/root/slime/examples/remote_agent/swe-bench-verified.jsonl}"

# Environment class import path (default: ACK for Alibaba Cloud Kubernetes)
HARBOR_ENV_IMPORT_PATH="${HARBOR_ENV_IMPORT_PATH:-harbor.environments.ack:ACKEnvironment}"

# Proxy host IP - the IP address that Harbor agents can reach the LLM proxy
# This should be set to the node's reachable IP address
HARBOR_PROXY_HOST="${HARBOR_PROXY_HOST:-10.0.30.19}"

# Environment kwargs (passed to the Harbor environment constructor)
# Option 1: Use BuildKit (requires buildctl installed and buildkitd running)
# Option 2: Use DinD (Docker-in-Docker, creates K8s Job for building)
# Currently using DinD mode as it doesn't require local buildctl/buildkitd setup
REMOTE_AGENT_ENVIRONMENT_KWARGS=${REMOTE_AGENT_ENVIRONMENT_KWARGS:-'{"namespace":"default","image_pull_secret":"acr-registry","use_buildkit":"true","use_sandbox_claim":"true","sandboxset_replicas":"1","sandbox_labels":"{\"alibabacloud.com/acs\":\"true\"}","buildkit_address":"tcp://buildkitd:1234","build_job_namespace":"default","registry":"registry.cn-hongkong.aliyuncs.com/swe-bench","kubeconfig":"/root/slime/examples/remote_agent/kubeconfig"}'}

source /root/slime/scripts/models/qwen2.5-7B.sh
# ---------------------------------------------------------------------------
# Training command
# ---------------------------------------------------------------------------
# Note: Megatron-LM source must be on PYTHONPATH before megatron-core package
# to ensure megatron.training module is accessible.
PYTHONPATH="/root/Megatron-LM:/root/slime:${PYTHONPATH:-}" \
python train_remote_agent.py ${MODEL_ARGS[@]} \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-use-local-trial \
  --harbor-agent-name "$HARBOR_AGENT_NAME" \
  --harbor-model-name "$HARBOR_MODEL_NAME" \
  --harbor-task-path-template "$HARBOR_TASK_PATH_TEMPLATE" \
  --harbor-agent-kwargs '{"total_cost_limit": 0, "per_instance_cost_limit": 0}' \
  --harbor-env-import-path "$HARBOR_ENV_IMPORT_PATH" \
  --harbor-env-kwargs "$REMOTE_AGENT_ENVIRONMENT_KWARGS" \
  --harbor-proxy-host "$HARBOR_PROXY_HOST" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --load "$MEGATRON_CHECKPOINT" \
  --save "$MEGATRON_CHECKPOINT" \
  --save-interval 10 \
  --rollout-max-response-len 4096 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --num-rollout 10 \
  --prompt-data "$PROMPT_DATA" \
  --input-key prompt \
  --rollout-global-dataset \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --rollout-temperature 1.0 \
  --rollout-top-p 1.0 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --rollout-num-gpus-per-engine 1 \
  --global-batch-size 1 \
  --micro-batch-size 1 \
  --advantage-estimator grpo \
  --normalize-advantages \
  --seed 42 \
  "$@"
