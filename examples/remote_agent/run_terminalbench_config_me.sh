#!/usr/bin/env bash
# TerminalBench remote-agent RL on the config-me cluster, with the ACK sandbox
# reached over SSH (harbor use_ssh_exec) and served from a SandboxClaim warm pool.
#
# Differences from run_terminalbench_test*.sh, which target the cpfs01 setup:
#   * paths are this cluster's: model on the head's local disk, tasks copied off
#     ossfs to /root/terminal-bench (ossfs random reads are an order slower)
#   * no `ray stop`: here slime runs inside the KubeRay head pod, where ray is
#     started by KubeRay — stopping it would tear the cluster down
#   * no kubeconfig by default: harbor talks to the cluster it runs in through
#     the pod's ServiceAccount. Set KUBECONFIG_IN_POD to override.
#   * SSH transport + warm pool, which is what this script exists to exercise
#
# Prerequisites, none of which this script creates:
#   1. The task image must already run an sshd (only adaptive-rejection-sampler
#      is built that way for now, hence the single-task PROMPT_DATA below).
#   2. The two SSH Secrets must exist in NAMESPACE — harbor never creates them:
#        kubectl -n default create secret generic harbor-ssh-private-keys
#        kubectl -n default create secret generic harbor-ssh-authorized-keys
#   3. The identity harbor uses needs `secrets get/patch` (publishing the key
#      pair) and agents.kruise.io `sandboxsets/sandboxclaims/sandboxes`
#      create/get/list/delete in NAMESPACE. The head's ServiceAccount
#      (rayclustertest, bound to ClusterRole kuberay-operator) has neither, so
#      either bind a Role like this one or point KUBECONFIG_IN_POD at a
#      kubeconfig with the rights:
#        apiVersion: rbac.authorization.k8s.io/v1
#        kind: Role
#        metadata: {name: slime-ssh-claim, namespace: default}
#        rules:
#        - {apiGroups: [""], resources: [secrets], verbs: [get, list, patch]}
#        - {apiGroups: ["agents.kruise.io"], resources: [sandboxsets, sandboxclaims, sandboxes],
#           verbs: [create, get, list, watch, patch, delete]}
#        ---
#        apiVersion: rbac.authorization.k8s.io/v1
#        kind: RoleBinding
#        metadata: {name: slime-ssh-claim, namespace: default}
#        roleRef: {apiGroup: rbac.authorization.k8s.io, kind: Role, name: slime-ssh-claim}
#        subjects: [{kind: ServiceAccount, name: rayclustertest, namespace: default}]
#   4. SSH dials the sandbox Pod IP directly, so this must run in-cluster.
#
# Quick start (inside the head pod):
#   cd /root/slime && bash examples/remote_agent/run_terminalbench_config_me.sh
#
# Any extra flags are passed through: ... run_terminalbench_config_me.sh --num-rollout 3
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

rm -rf ./trials/* 2>/dev/null || true

# ---------------------------------------------------------------------------
# Switches + config (all overridable via env)
# ---------------------------------------------------------------------------
DEPLOY="${DEPLOY:-colocate}"             # colocate | disagg

MODEL_PRESET="${MODEL_PRESET:-qwen3.5-4B}"                  # scripts/models/<preset>.sh -> MODEL_ARGS
HF_CKPT="${HF_CKPT:-/root/Qwen3.5-4B}"                      # tokenizer/config (LOCAL disk)
REF_LOAD="${REF_LOAD:-/root/Qwen3.5-4B_torch_dist}"         # Megatron dist ckpt (--ref-load)
MODEL_NAME="${MODEL_NAME:-openai/Qwen3.5-4B}"               # name advertised to the agent
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"             # qwen3.5 loads a processor -> prompt must be a message list

SAVE_DIR="${SAVE_DIR:-/root/save-3.5-4b}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
# Set RESUME=1 to load from SAVE_DIR on startup (--load). Default off — a
# previous crashed run can leave an iter_0000000 placeholder that fails the
# "iteration > 0" assertion on reload.
RESUME="${RESUME:-0}"

TP="${TP:-2}"; PP="${PP:-1}"; GPUS="${GPUS:-8}"                       # actor: DP = GPUS/(TP*PP)
# One GPU per sglang engine, which is the maximum number of rollout replicas this
# node can host (8). The engine's tensor parallelism is independent of the actor's
# TP, and a 4B model needs nowhere near a whole H20, so more replicas is strictly
# better for a 64-way concurrent agent workload: more independent decode streams
# and more aggregate prefix cache.
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
SGLANG_MEM="${SGLANG_MEM:-$([ "$DEPLOY" = disagg ] && echo 0.8 || echo 0.5)}"  # colocate: leave room for training

# 64-way concurrency over 64 *distinct* tasks: one sample per prompt, so every
# concurrent trial is a different case. Note this leaves GRPO with a single sample
# per group, hence zero advantage variance — fine for a throughput measurement,
# not for learning (raise N_SAMPLES for that, at the cost of distinct cases).
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"; N_SAMPLES="${N_SAMPLES:-1}"; ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"; MAX_RESP="${MAX_RESP:-2048}"
MAX_CTX="${MAX_CTX:-40960}"

# terminus-2 is a real LLM terminal agent. Do NOT use 'oracle' for RL — it runs
# the reference solution without any LLM call, so the adapter captures no tokens
# and every trajectory is dropped ("No turns recorded").
HARBOR_AGENT_NAME="${HARBOR_AGENT_NAME:-terminus-2}"
# model_info prices the self-hosted model at zero so litellm's cost logic cannot
# truncate the run; llm_kwargs.api_key only satisfies litellm's client-side check
# (the adapter routes by the session id in the body).
HARBOR_AGENT_KWARGS="${HARBOR_AGENT_KWARGS:-}"
[ -n "$HARBOR_AGENT_KWARGS" ] || HARBOR_AGENT_KWARGS='{
  "model_info": {
    "max_input_tokens": 32768,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0
  },
  "proactive_summarization_threshold": 8000,
  "llm_kwargs": {"api_key": "sk-slime-adapter"},
  "max_turns": 40
}'

# --- ACK sandbox backend (this cluster) ------------------------------------
NAMESPACE="${NAMESPACE:-default}"
# The task image is pulled from the me-east-1 ACR, in-region with this cluster.
# Pulling the Hong Kong copy from here does not survive concurrency: a few hundred
# simultaneous pulls all fail with TLS handshake timeouts against the HK endpoint.
IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-acr-pro-registry-me}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-rayclustertest}"
KUBECONFIG_IN_POD="${KUBECONFIG_IN_POD:-}"                   # empty -> in-cluster ServiceAccount
# Schedules the sandboxes onto ACS (serverless) capacity instead of the cluster's
# own nodes, which is what makes a few hundred concurrent sandboxes practical.
# harbor stamps sandbox_labels on the SandboxSet, its pod template and the claim.
# Assigned in two steps, not as a ${VAR:-default}: a `}` in the default word ends
# the expansion early and would leave a stray backslash in the JSON.
SANDBOX_LABELS="${SANDBOX_LABELS:-}"
[ -n "$SANDBOX_LABELS" ] || SANDBOX_LABELS='{"alibabacloud.com/acs": "true"}'

# One ed25519 key pair is minted per scope, and the SandboxSet name must be
# unique per scope: a pool created for an earlier scope mounts that scope's
# authorized_keys, which this run's private key would not match. Deriving both
# from one value keeps them in step; a fresh value per run keeps them unique.
SSH_KEY_SCOPE="${SSH_KEY_SCOPE:-slime-tb-$(date +%Y%m%d-%H%M%S)}"
SANDBOXSET_PREFIX="${SANDBOXSET_PREFIX:-$SSH_KEY_SCOPE}"

# Warm-pool size, deliberately independent of concurrency. This run has 64
# different task images, hence 64 different SandboxSets: even one warm replica
# per set would allocate 64 idle pods, and sizing each pool to the concurrency
# creates 512 warm pods on top of 64 claims. Zero is valid; every claim uses
# createOnNoStock to cold-create exactly its one sandbox.
SANDBOXSET_REPLICAS="${SANDBOXSET_REPLICAS:-0}"

# harbor starts sshd in the SandboxSet itself (`ssh_server_command`), so the claimed
# pod no longer needs a pod_overrides command. It is wrapped in a shell that runs
# `ssh-keygen -A` first: an image with sshd installed but no host keys would
# otherwise fail to start it, and `|| true` keeps an image that already has keys
# (or lacks ssh-keygen) from failing the container.
SSH_SERVER_COMMAND="${SSH_SERVER_COMMAND:-}"
[ -n "$SSH_SERVER_COMMAND" ] || SSH_SERVER_COMMAND='["/bin/sh", "-c", "mkdir -p /var/run/sshd; ssh-keygen -A || true; exec /usr/sbin/sshd -D -e"]'

# NOTE: build this JSON with a heredoc, not ${VAR:-default} — a `}` inside the
# default word closes the expansion early and mangles the JSON.
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
  "claim_timeout": 1200,
  "sandbox_ready_timeout_sec": 600,
  "build_timeout_sec": 1200,
  "use_ssh_exec": true,
  "ssh_key_scope": "$SSH_KEY_SCOPE",
  "ssh_server_command": $SSH_SERVER_COMMAND,
  "ssh_connect_timeout_sec": 180,
  "ssh_key_revoke_delay_sec": 600
}
JSON
)

# TerminalBench task directories (each contains task.toml + instruction.md),
# copied off ossfs onto the head's local disk and retargeted at the me-east-1
# registry by examples/remote_agent/retarget_tasks.py.
TASK_PATH_TEMPLATE="${TASK_PATH_TEMPLATE:-}"; [ -n "$TASK_PATH_TEMPLATE" ] || TASK_PATH_TEMPLATE='/root/terminal-bench/{instance_id}'
# 64 distinct tasks, all of whose images carry an sshd (tag 20260901) and whose
# manifest digests match between the cn-hongkong and me-east-1 registries.
PROMPT_DATA="${PROMPT_DATA:-$(pwd)/examples/remote_agent/prompts_tb_64.jsonl}"

# Whole-trial wall clock (--harbor-timeout). It has to exceed the sum of the
# per-phase budgets, which harbor takes from the task's own task.toml
# ([agent].timeout_sec + [verifier].timeout_sec, plus environment setup), or a
# trial is cut off mid-verify and its reward is discarded. retarget_tasks.py sets
# agent 3600 + verifier 1800, so 7200 leaves room for setup and teardown.
HARBOR_TIMEOUT="${HARBOR_TIMEOUT:-7200}"

# sglang router load-balancing policy across the rollout replicas. slime passes no
# policy by default, which leaves the router on its own default `cache_aware`.
# The adapter always sends `X-SMG-Routing-Key: <session id>`, but only
# `consistent_hashing` honours it — that is the session-affinity (sticky) mode,
# which keeps every turn of one agent on the worker holding its prefix cache.
# Other useful values: `random`, `round_robin`, `prefix_hash`.
ROUTER_POLICY="${ROUTER_POLICY:-}"

# ---------------------------------------------------------------------------
# Env + prereq checks
# ---------------------------------------------------------------------------
source "scripts/models/${MODEL_PRESET}.sh"                           # defines MODEL_ARGS
export PYTHONPATH="/root/Megatron-LM:/root/harbor/src:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

[ -f "$HF_CKPT/config.json" ] || { echo "ERROR: HF ckpt not found at $HF_CKPT (tokenizer/config)."; exit 1; }
[ -f "$REF_LOAD/latest_checkpointed_iteration.txt" ] || { echo "ERROR: dist ckpt not found at $REF_LOAD. Convert with tools/convert_hf_to_torch_dist.py first."; exit 1; }
[ -f "$PROMPT_DATA" ] || { echo "ERROR: prompt data not found at $PROMPT_DATA."; exit 1; }
_TASK_DIR="$(dirname "${TASK_PATH_TEMPLATE//\{instance_id\}/}")"
[ -d "$_TASK_DIR" ] || echo "[warn] task root $_TASK_DIR not found — copy the dataset to local disk first."
python -c "import asyncssh" 2>/dev/null || { echo "ERROR: asyncssh missing (use_ssh_exec needs it): pip install 'harbor[ack-ssh]'"; exit 1; }
python -c "import harbor.environments.ack_ssh_transport" 2>/dev/null || { echo "ERROR: this harbor has no SSH transport — check out feat/ack-persistent-exec."; exit 1; }
_MP=$((TP * PP)); _DP=$(( GPUS / (_MP == 0 ? 1 : _MP) ))   # data-parallel = actor GPUs / (TP*PP)
[ "$((GPUS % _MP))" -eq 0 ] || echo "[warn] GPUS($GPUS) not divisible by TP*PP($_MP) — invalid model-parallel layout."
[ "$_DP" -gt 0 ] && [ "$((GLOBAL_BATCH_SIZE % _DP))" -eq 0 ] || echo "[warn] GLOBAL_BATCH_SIZE($GLOBAL_BATCH_SIZE) not a multiple of DP($_DP) — Megatron will assert."

HEAD_IP="$(hostname -i | awk '{print $1}')"   # endpoint the sandbox agent dials back to

# ---------------------------------------------------------------------------
# Assemble args
# ---------------------------------------------------------------------------
ARGS=(
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "$HF_CKPT" --ref-load "$REF_LOAD"
  --save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL"
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
  --harbor-use-local-trial --harbor-env-import-path harbor.environments.ack:ACKEnvironment
  --harbor-adapter-public-host "$HEAD_IP" --harbor-adapter-port "${HARBOR_ADAPTER_PORT:-18001}"
  --harbor-agent-name "$HARBOR_AGENT_NAME" --harbor-model-name "$MODEL_NAME"
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

# --- deployment ---
if [ "$DEPLOY" = colocate ]; then
  ARGS+=( --colocate )
elif [ "$DEPLOY" = disagg ]; then
  ROLLOUT_GPUS="${ROLLOUT_GPUS:?DEPLOY=disagg needs ROLLOUT_GPUS (dedicated rollout GPUs)}"
  echo "[check] disaggregated: total GPUs = actor($GPUS) + rollout($ROLLOUT_GPUS) = $((GPUS + ROLLOUT_GPUS))"
  ARGS+=( --rollout-num-gpus "$ROLLOUT_GPUS" )
else
  echo "ERROR: DEPLOY must be 'colocate' or 'disagg' (got '$DEPLOY')"; exit 1
fi

echo "[run] DEPLOY=$DEPLOY preset=$MODEL_PRESET TP=$TP PP=$PP GPUS=$GPUS DP=$_DP rollout_replicas=$((GPUS / ROLLOUT_GPUS_PER_ENGINE)) adapter=$HEAD_IP:${HARBOR_ADAPTER_PORT:-18001} sglang_mem=$SGLANG_MEM router_policy=${ROUTER_POLICY:-<router default: cache_aware>}"
echo "[run] concurrency=$((ROLLOUT_BATCH_SIZE * N_SAMPLES)) prompts=$ROLLOUT_BATCH_SIZE samples_each=$N_SAMPLES steps=$NUM_ROLLOUT gbs=$GLOBAL_BATCH_SIZE harbor_timeout=$HARBOR_TIMEOUT"
echo "[run] ACK sandbox: ns=$NAMESPACE claim=on ssh=on scope=$SSH_KEY_SCOPE sandboxset_prefix=$SANDBOXSET_PREFIX replicas=$SANDBOXSET_REPLICAS pull_secret=$IMAGE_PULL_SECRET"
echo "[run] data=$PROMPT_DATA task_path=$TASK_PATH_TEMPLATE"
exec python train_remote_agent.py "${ARGS[@]}" "$@"
