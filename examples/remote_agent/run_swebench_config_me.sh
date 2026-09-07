#!/usr/bin/env bash
# SWE-bench Verified remote-agent RL on config-me.
#
# This launcher intentionally uses ACK SandboxClaim + a persistent pods/exec
# websocket, not SSH: the SWE-bench Verified images do not contain sshd (unlike
# the TerminalBench :20260901 images). Persistent exec keeps one API-server
# websocket per Trial and avoids a handshake for every agent command.
#
# Defaults: Qwen3.5-4B, one GPU per SGLang engine (8 rollout replicas on the
# 8-GPU head), 64 distinct simple/medium tasks, 64 concurrent trials, 3 steps.
# Extra CLI flags are forwarded to train_remote_agent.py.
set -euo pipefail
cd "$(dirname "$0")/../.."

rm -rf ./trials/* 2>/dev/null || true

DEPLOY="${DEPLOY:-colocate}"
MODEL_PRESET="${MODEL_PRESET:-qwen3.5-4B}"
HF_CKPT="${HF_CKPT:-/root/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-/root/Qwen3.5-4B_torch_dist}"
MODEL_NAME="${MODEL_NAME:-openai/Qwen3.5-4B}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
SAVE_DIR="${SAVE_DIR:-/root/save-swe-qwen3.5-4b}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
RESUME="${RESUME:-0}"

TP="${TP:-2}"
PP="${PP:-1}"
GPUS="${GPUS:-8}"
# Maximum rollout replicas: one 4B engine per H20. Actor TP remains independent.
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
SGLANG_MEM="${SGLANG_MEM:-$([ "$DEPLOY" = disagg ] && echo 0.8 || echo 0.5)}"

NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
N_SAMPLES="${N_SAMPLES:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_RESP="${MAX_RESP:-2048}"
MAX_CTX="${MAX_CTX:-40960}"

# Use Harbor's SWE-agent integration. The workspace image prepares it once at
# build time; an init container in each sandbox Pod copies that tree to a
# pod-local emptyDir shared with the task container. The custom subclass only
# activates compatibility symlinks—it never clones or pip-installs in a trial.
HARBOR_AGENT_NAME="${HARBOR_AGENT_NAME:-}"
HARBOR_AGENT_IMPORT_PATH="${HARBOR_AGENT_IMPORT_PATH:-examples.remote_agent.shared_swe_agent:SharedSweAgent}"
HARBOR_AGENT_KWARGS="${HARBOR_AGENT_KWARGS:-}"
[ -n "$HARBOR_AGENT_KWARGS" ] || HARBOR_AGENT_KWARGS='{
  "shared_install_dir": "/opt/sweagent-shared",
  "version": "v1.1.0",
  "per_instance_cost_limit": 0,
  "total_cost_limit": 0,
  "max_input_tokens": 32768
}'

# ACK / ACS sandbox backend.
NAMESPACE="${NAMESPACE:-default}"
IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-acr-pro-registry-me}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-rayclustertest}"
KUBECONFIG_IN_POD="${KUBECONFIG_IN_POD:-}"
SANDBOX_LABELS="${SANDBOX_LABELS:-}"
[ -n "$SANDBOX_LABELS" ] || SANDBOX_LABELS='{"alibabacloud.com/acs": "true"}'
# The init container must use the exact running workspace image because that is
# where /opt/sweagent-shared was baked.
WORKSPACE_IMAGE="${WORKSPACE_IMAGE:-}"
if [ -z "$WORKSPACE_IMAGE" ]; then
  WORKSPACE_IMAGE=$(python - <<'PY'
import socket
from kubernetes import client, config
config.load_incluster_config()
pod = client.CoreV1Api().read_namespaced_pod(socket.gethostname(), "default")
print(pod.spec.containers[0].image)
PY
)
fi

# One SandboxSet exists per task image. A warm replica in every set doubles the
# footprint for a diverse batch; zero plus createOnNoStock creates exactly one
# sandbox per claim.
SANDBOXSET_PREFIX="${SANDBOXSET_PREFIX:-slime-swe-$(date +%Y%m%d-%H%M%S)}"
SANDBOXSET_REPLICAS="${SANDBOXSET_REPLICAS:-0}"

HARBOR_ENV_KWARGS="${HARBOR_ENV_KWARGS:-}"
[ -n "$HARBOR_ENV_KWARGS" ] || HARBOR_ENV_KWARGS=$(cat <<JSON
{
  "namespace": "$NAMESPACE",
  "image_pull_secret": "$IMAGE_PULL_SECRET",
  "service_account": "$SERVICE_ACCOUNT",
  $([ -n "$KUBECONFIG_IN_POD" ] && echo "\"kubeconfig\": \"$KUBECONFIG_IN_POD\",")
  "use_sandbox_claim": true,
  "override_claim_image": true,
  "sandbox_labels": $SANDBOX_LABELS,
  "sandboxset_prefix": "$SANDBOXSET_PREFIX",
  "sandboxset_replicas": $SANDBOXSET_REPLICAS,
  "sandbox_runtimes": [],
  "claim_timeout": 1800,
  "sandbox_ready_timeout_sec": 900,
  "build_timeout_sec": 1800,
  "pod_overrides": {
    "spec": {
      "volumes": [
        {"name": "sweagent-shared", "emptyDir": {}}
      ],
      "initContainers": [
        {
          "name": "copy-sweagent",
          "image": "$WORKSPACE_IMAGE",
          "command": ["/bin/sh", "-c", "cp -a /opt/sweagent-shared/. /shared/"],
          "volumeMounts": [{"name": "sweagent-shared", "mountPath": "/shared"}]
        }
      ],
      "containers": [
        {"volumeMounts": [
          {"name": "sweagent-shared", "mountPath": "/opt/sweagent-shared", "readOnly": true}
        ]}
      ]
    }
  },
  "exec_api_pool_size": 256,
  "use_persistent_exec_connection": true,
  "persistent_exec_connect_max_attempts": 10,
  "persistent_exec_connect_initial_backoff_sec": 0.5,
  "persistent_exec_connect_max_backoff_sec": 8.0
}
JSON
)

TASK_PATH_TEMPLATE="${TASK_PATH_TEMPLATE:-}"
[ -n "$TASK_PATH_TEMPLATE" ] || TASK_PATH_TEMPLATE='/root/swe-bench-verified/{instance_id}'
PROMPT_DATA="${PROMPT_DATA:-$(pwd)/examples/remote_agent/prompts_swe_64.jsonl}"
# retarget_tasks.py sets agent=3600 and verifier=1800. Keep room for setup and
# teardown around those two phase budgets.
HARBOR_TIMEOUT="${HARBOR_TIMEOUT:-7200}"

# Empty leaves the router's default cache_aware policy. consistent_hashing uses
# the X-SMG-Routing-Key emitted by the adapter to keep every turn of one agent on
# one engine. The A/B experiment runs this script once each way.
ROUTER_POLICY="${ROUTER_POLICY:-}"

source "scripts/models/${MODEL_PRESET}.sh"
export PYTHONPATH="/root/Megatron-LM:/root/harbor/src:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

[ -f "$HF_CKPT/config.json" ] || { echo "ERROR: HF ckpt missing: $HF_CKPT"; exit 1; }
[ -f "$REF_LOAD/latest_checkpointed_iteration.txt" ] || { echo "ERROR: dist ckpt missing: $REF_LOAD"; exit 1; }
[ -f "$PROMPT_DATA" ] || { echo "ERROR: prompt data missing: $PROMPT_DATA"; exit 1; }
_TASK_ROOT="$(dirname "${TASK_PATH_TEMPLATE//\{instance_id\}/}")"
[ -d "$_TASK_ROOT" ] || { echo "ERROR: task root missing: $_TASK_ROOT"; exit 1; }
python -c "from harbor.environments.ack import ACKEnvironment" 2>/dev/null || {
  echo "ERROR: harbor ACKEnvironment unavailable"; exit 1;
}

_MP=$((TP * PP))
_DP=$((GPUS / (_MP == 0 ? 1 : _MP)))
[ "$((GPUS % _MP))" -eq 0 ] || { echo "ERROR: GPUS($GPUS) not divisible by TP*PP($_MP)"; exit 1; }
[ "$_DP" -gt 0 ] && [ "$((GLOBAL_BATCH_SIZE % _DP))" -eq 0 ] || {
  echo "ERROR: GLOBAL_BATCH_SIZE($GLOBAL_BATCH_SIZE) not divisible by DP($_DP)"; exit 1;
}
HEAD_IP="$(hostname -i | awk '{print $1}')"

ARGS=(
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "$HF_CKPT" --ref-load "$REF_LOAD"
  --save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL"
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
  --harbor-use-local-trial --harbor-env-import-path harbor.environments.ack:ACKEnvironment
  --harbor-adapter-public-host "$HEAD_IP" --harbor-adapter-port "${HARBOR_ADAPTER_PORT:-18001}"
  --harbor-model-name "$MODEL_NAME"
  --harbor-task-path-template "$TASK_PATH_TEMPLATE"
  --harbor-env-kwargs "$HARBOR_ENV_KWARGS"
  --harbor-agent-kwargs "$HARBOR_AGENT_KWARGS"
  --harbor-timeout "$HARBOR_TIMEOUT"
  --prompt-data "$PROMPT_DATA" --input-key prompt --rollout-global-dataset
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "$MAX_RESP" --rollout-max-context-len "$MAX_CTX" --rollout-temperature 1.0
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --actor-num-nodes 1 --actor-num-gpus-per-node "$GPUS" --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static "$SGLANG_MEM" --sglang-disable-cuda-graph
  --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP" --sequence-parallel
  --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_CTX"
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
  --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --seed 42
)

[ "$RESUME" = 1 ] && ARGS+=( --load "$SAVE_DIR" )
[ "$APPLY_CHAT_TEMPLATE" = 1 ] && ARGS+=( --apply-chat-template )
[ -n "$ROUTER_POLICY" ] && ARGS+=( --router-policy "$ROUTER_POLICY" )
if [ -n "$HARBOR_AGENT_IMPORT_PATH" ]; then
  ARGS+=( --harbor-agent-import-path "$HARBOR_AGENT_IMPORT_PATH" )
elif [ -n "$HARBOR_AGENT_NAME" ]; then
  ARGS+=( --harbor-agent-name "$HARBOR_AGENT_NAME" )
else
  echo "ERROR: set HARBOR_AGENT_IMPORT_PATH or HARBOR_AGENT_NAME"; exit 1
fi

if [ "$DEPLOY" = colocate ]; then
  ARGS+=( --colocate )
elif [ "$DEPLOY" = disagg ]; then
  ROLLOUT_GPUS="${ROLLOUT_GPUS:?DEPLOY=disagg needs ROLLOUT_GPUS}"
  ARGS+=( --rollout-num-gpus "$ROLLOUT_GPUS" )
else
  echo "ERROR: DEPLOY must be colocate or disagg (got $DEPLOY)"; exit 1
fi

echo "[run] SWE-bench Verified DEPLOY=$DEPLOY GPUS=$GPUS actor_TP=$TP DP=$_DP rollout_replicas=$((GPUS / ROLLOUT_GPUS_PER_ENGINE))"
echo "[run] concurrency=$((ROLLOUT_BATCH_SIZE * N_SAMPLES)) distinct_prompts=$ROLLOUT_BATCH_SIZE steps=$NUM_ROLLOUT router=${ROUTER_POLICY:-cache_aware} transport=persistent_exec agent=${HARBOR_AGENT_IMPORT_PATH:-$HARBOR_AGENT_NAME}"
echo "[run] data=$PROMPT_DATA tasks=$TASK_PATH_TEMPLATE sandboxset_prefix=$SANDBOXSET_PREFIX"
exec python train_remote_agent.py "${ARGS[@]}" "$@"
