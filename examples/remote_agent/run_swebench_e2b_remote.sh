#!/usr/bin/env bash
# Run slime RL with remote agents, dispatching each trial to a remote kube-rl server
# (which runs the agent inside an ACK sandbox over the E2B protocol).
#
# Difference vs run_swebench_e2b.sh (in-process / local-trial):
#   - NO --harbor-use-local-trial; instead --harbor-server-url points at the kube-rl server.
#   - NO E2B_* env on the slime side (kube-rl holds the sandbox credentials).
#   - slime only runs the SGLang engines + TokenProxy and POSTs trials over HTTP.
#
# The kube-rl server must itself be on the E2B environment:
#   curl http://kube-rl.<ns>:8080/api/v1/capabilities/environments   # importPath == harbor.environments.e2b:E2BEnvironment
#
# See docs/en/platform_support/ack_sandbox_e2b.md (section 5B).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root

source "scripts/models/${MODEL_PRESET:-qwen2.5-0.5B}.sh"

export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

HF_CKPT="${HF_CKPT:-/root/Qwen2.5-0.5B-Instruct}"
REF_LOAD="${REF_LOAD:-/var/model/Qwen2.5-0.5B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-$(pwd)/examples/remote_agent/small_prompts.jsonl}"
MODEL_NAME="${MODEL_NAME:-openai/Qwen2.5-0.5B-Instruct}"
TASK_PATH_TEMPLATE="${TASK_PATH_TEMPLATE:-/var/model-dataset/swe-bench-verified/{instance_id}}"
KUBE_RL_URL="${KUBE_RL_URL:-http://kube-rl.kube-rl.svc.cluster.local:8080}"

# IP of THIS (slime head) pod, reachable from the kube-rl-managed sandboxes so the in-sandbox
# agent can reach the TokenProxy. Must NOT be 0.0.0.0.
PROXY_HOST="${PROXY_HOST:-$(hostname -i | awk '{print $1}')}"

python train_remote_agent.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CKPT" \
  --ref-load "$REF_LOAD" \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --harbor-server-url "$KUBE_RL_URL" \
  --harbor-proxy-host "$PROXY_HOST" \
  --harbor-max-retries 3 \
  --harbor-agent-name swe-agent \
  --harbor-model-name "$MODEL_NAME" \
  --harbor-task-path-template "$TASK_PATH_TEMPLATE" \
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
