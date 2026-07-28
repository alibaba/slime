#!/usr/bin/env bash
# Run slime RL with remote agents on ACK sandboxes over the E2B protocol.
#
# The agent runs inside an ACK sandbox; its LLM calls go through slime's TokenProxy
# to the SGLang engines, and token-level data is captured for GRPO training.
#
# Prerequisites (see docs/en/platform_support/ack_sandbox_e2b.md):
#   - ACK sandbox stack installed (sandbox-manager + sandbox-gateway).
#   - One SandboxSet per task image (spec.runtimes:[{name: agent-runtime}] + image pull secret),
#     routed per-sample via prompt metadata "sandbox_set_name".
#   - Megatron ref-load checkpoint converted from HF (tools/convert_hf_to_torch_dist.py).
#
# Usage:
#   export E2B_API_KEY=<ACK sandbox admin key>
#   bash examples/remote_agent/run_swebench_e2b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (/root/slime in the workspace image)

# --- model preset (defines MODEL_ARGS: --num-layers, --hidden-size, ...) ---
source "scripts/models/${MODEL_PRESET:-qwen2.5-0.5B}.sh"

# --- E2B connection to the in-cluster ACK sandbox stack ---
export E2B_API_KEY="${E2B_API_KEY:?set E2B_API_KEY to the ACK sandbox admin key}"
export E2B_API_URL="${E2B_API_URL:-http://sandbox-manager.sandbox-system:8080}"      # control plane
export E2B_SANDBOX_URL="${E2B_SANDBOX_URL:-http://sandbox-gateway.sandbox-system:7788}"  # data plane (router)
export E2B_VALIDATE_API_KEY="${E2B_VALIDATE_API_KEY:-false}"

# --- Megatron + TP env ---
export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

HF_CKPT="${HF_CKPT:-/root/Qwen2.5-0.5B-Instruct}"
REF_LOAD="${REF_LOAD:-/var/model/Qwen2.5-0.5B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-$(pwd)/examples/remote_agent/small_prompts.jsonl}"
MODEL_NAME="${MODEL_NAME:-openai/Qwen2.5-0.5B-Instruct}"
TASK_PATH_TEMPLATE="${TASK_PATH_TEMPLATE:-/var/model-dataset/swe-bench-verified/{instance_id}}"

# IP of THIS (slime head) pod, reachable from the sandboxes so the in-sandbox agent
# can reach the TokenProxy. Must NOT be 0.0.0.0.
PROXY_HOST="${PROXY_HOST:-$(hostname -i | awk '{print $1}')}"

python train_remote_agent.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CKPT" \
  --ref-load "$REF_LOAD" \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --harbor-use-local-trial \
  --harbor-proxy-host "$PROXY_HOST" \
  --harbor-agent-name swe-agent \
  --harbor-model-name "$MODEL_NAME" \
  --harbor-task-path-template "$TASK_PATH_TEMPLATE" \
  --harbor-env-import-path harbor.environments.e2b:E2BEnvironment \
  --harbor-env-kwargs '{"override_claim_image": false}' \
  --prompt-data "$PROMPT_DATA" \
  --input-key prompt \
  --rollout-global-dataset \
  --num-rollout "${NUM_ROLLOUT:-1}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-2}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-2048}" \
  --rollout-temperature 1.0 \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-4}" \
  --colocate \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${GPUS:-8}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE:-2}" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.5}" \
  --sglang-disable-cuda-graph \
  --tensor-model-parallel-size "${TP:-2}" --pipeline-model-parallel-size 1 --sequence-parallel \
  --use-dynamic-batch-size --max-tokens-per-gpu 9216 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl \
  --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 --attention-backend flash \
  --seed 42 \
  "$@"
