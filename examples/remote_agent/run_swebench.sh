#!/usr/bin/env bash
# Unified SWE-bench remote-agent RL launcher (in-process OpenAIAdapter).
# One script, three orthogonal switches — see examples/remote_agent/README.md
#
#   MODE   = local | kuberl     trial submission: in-process local-trial (needs E2B_*) | via kube-rl server
#   DEPLOY = colocate | disagg  GPU layout: train+rollout share GPUs | dedicated rollout GPUs (needs ROLLOUT_GPUS)
#   model  = MODEL_PRESET/HF_CKPT/REF_LOAD/MODEL_NAME/TP/PP + APPLY_CHAT_TEMPLATE (multimodal)
#
# Quick starts (from /root/slime):
#   # 0.5B smoke, local-trial, colocate, 2 GPUs
#   MODE=local DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 E2B_API_KEY=<key> bash examples/remote_agent/run_swebench.sh
#   # 27B via kube-rl, colocate, 8 GPUs (TP4xPP2)
#   MODE=kuberl DEPLOY=colocate MODEL_PRESET=qwen3.5-27B HF_CKPT=/root/Qwen3.6-27B REF_LOAD=/root/Qwen3.6-27B_torch_dist \
#     MODEL_NAME=openai/Qwen3.6-27B GPUS=8 TP=4 PP=2 SGLANG_MEM=0.2 APPLY_CHAT_TEMPLATE=1 bash examples/remote_agent/run_swebench.sh
# Any extra flags are passed through: ... bash examples/remote_agent/run_swebench.sh --num-rollout 3
set -euo pipefail
cd "$(dirname "$0")/../.." 2>/dev/null || cd /root/slime   # -> repo root (/root/slime in the image)

# ---------------------------------------------------------------------------
# Switches + config (all overridable via env)
# ---------------------------------------------------------------------------
MODE="${MODE:-local}"                    # local | kuberl
DEPLOY="${DEPLOY:-colocate}"             # colocate | disagg

MODEL_PRESET="${MODEL_PRESET:-qwen2.5-0.5B}"          # scripts/models/<preset>.sh -> MODEL_ARGS
HF_CKPT="${HF_CKPT:-/root/Qwen2.5-0.5B-Instruct}"     # tokenizer/config (LOCAL disk; ossfs is slow)
REF_LOAD="${REF_LOAD:-/var/model/Qwen2.5-0.5B_torch_dist}"   # Megatron dist ckpt (--ref-load)
MODEL_NAME="${MODEL_NAME:-openai/Qwen2.5-0.5B-Instruct}"     # name advertised to the swe-agent
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"       # 1 for multimodal models (prompt must be message-list)

TP="${TP:-2}"; PP="${PP:-1}"; GPUS="${GPUS:-2}"                       # actor: DP = GPUS/TP (colocate) or actor GPUs (disagg)
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-$TP}"            # sglang engine TP = model TP
SGLANG_MEM="${SGLANG_MEM:-$([ "$DEPLOY" = disagg ] && echo 0.8 || echo 0.5)}"  # colocate: leave room for training

NUM_ROLLOUT="${NUM_ROLLOUT:-1}"; N_SAMPLES="${N_SAMPLES:-2}"; ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"; MAX_RESP="${MAX_RESP:-2048}"

HARBOR_AGENT_NAME="${HARBOR_AGENT_NAME:-swe-agent}"
HARBOR_AGENT_KWARGS="${HARBOR_AGENT_KWARGS:-}"        # optional JSON, e.g. '{"per_instance_cost_limit": 0}'
TASK_NAME="${TASK_NAME:-astropy__astropy-14309}"
SANDBOX_SET="${SANDBOX_SET:-slime-sbx-astropy-14309}"
TASK_PATH_TEMPLATE="${TASK_PATH_TEMPLATE:-/var/model-dataset/swe-bench-verified/{instance_id}}"
PROMPT_DATA="${PROMPT_DATA:-$(pwd)/examples/remote_agent/prompts.jsonl}"

KUBE_RL="${KUBE_RL:-http://kube-rl.kube-rl.svc.cluster.local:8080}"   # MODE=kuberl only

# ---------------------------------------------------------------------------
# Env + prereq checks
# ---------------------------------------------------------------------------
source "scripts/models/${MODEL_PRESET}.sh"                           # defines MODEL_ARGS
export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[ -f "$HF_CKPT/config.json" ] || { echo "ERROR: HF ckpt not found at $HF_CKPT (tokenizer/config). cp the model to LOCAL disk."; exit 1; }
[ -f "$REF_LOAD/latest_checkpointed_iteration.txt" ] || { echo "ERROR: dist ckpt not found at $REF_LOAD. Convert with tools/convert_hf_to_torch_dist.py first."; exit 1; }
python -c "import mbridge" 2>/dev/null || echo "[warn] mbridge not importable — required for weight bridging of some models (e.g. qwen3_5/27B). pip install the pinned mbridge if the run fails at update_weights."
_MP=$((TP * PP)); _DP=$(( GPUS / (_MP == 0 ? 1 : _MP) ))   # data-parallel = actor GPUs / (TP*PP)
[ "$((GPUS % _MP))" -eq 0 ] || echo "[warn] GPUS($GPUS) not divisible by TP*PP($_MP) — invalid model-parallel layout."
[ "$_DP" -gt 0 ] && [ "$((GLOBAL_BATCH_SIZE % _DP))" -eq 0 ] || echo "[warn] GLOBAL_BATCH_SIZE($GLOBAL_BATCH_SIZE) not a multiple of DP(GPUS/(TP*PP)=$_DP) — Megatron will assert."

HEAD_IP="$(hostname -i | awk '{print $1}')"   # endpoint A the (possibly external) sandbox agent dials back to

# prompt file: message-list when APPLY_CHAT_TEMPLATE=1 (multimodal), else plain string.
# Generate proper multi-task files with convert_swebench_tasks_to_prompts.py [--prompt-as-messages].
if [ ! -f "$PROMPT_DATA" ]; then
  echo "[info] writing default single-task prompt file: $PROMPT_DATA"
  _instr="Fix the bug described in the task instruction in the mounted repository. Read the instruction, locate the root cause, and edit the source so the failing tests pass."
  if [ "$APPLY_CHAT_TEMPLATE" = 1 ]; then
    echo "{\"prompt\": [{\"role\": \"user\", \"content\": \"$_instr\"}], \"task_name\": \"$TASK_NAME\", \"metadata\": {\"sandbox_set_name\": \"$SANDBOX_SET\"}}" > "$PROMPT_DATA"
  else
    echo "{\"prompt\": \"$_instr\", \"task_name\": \"$TASK_NAME\", \"metadata\": {\"sandbox_set_name\": \"$SANDBOX_SET\"}}" > "$PROMPT_DATA"
  fi
fi

# ---------------------------------------------------------------------------
# Assemble args
# ---------------------------------------------------------------------------
ARGS=(
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "$HF_CKPT" --ref-load "$REF_LOAD"
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
  --harbor-adapter-public-host "$HEAD_IP" --harbor-adapter-port "${HARBOR_ADAPTER_PORT:-18001}"
  --harbor-agent-name "$HARBOR_AGENT_NAME" --harbor-model-name "$MODEL_NAME"
  --harbor-task-path-template "$TASK_PATH_TEMPLATE"
  --harbor-env-kwargs '{"override_claim_image": false}'
  --prompt-data "$PROMPT_DATA" --input-key prompt --rollout-global-dataset
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "$MAX_RESP" --rollout-temperature 1.0 --global-batch-size "$GLOBAL_BATCH_SIZE"
  --actor-num-nodes 1 --actor-num-gpus-per-node "$GPUS" --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static "$SGLANG_MEM" --sglang-disable-cuda-graph
  --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP" --sequence-parallel
  --use-dynamic-batch-size --max-tokens-per-gpu 9216
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
  --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash --seed 42
)

# --- trial mode ---
if [ "$MODE" = local ]; then
  export E2B_API_KEY="${E2B_API_KEY:?MODE=local needs E2B_API_KEY (ACK sandbox admin key)}"
  export E2B_API_URL="${E2B_API_URL:-http://sandbox-manager.sandbox-system:8080}"
  export E2B_SANDBOX_URL="${E2B_SANDBOX_URL:-http://sandbox-gateway.sandbox-system:7788}"
  export E2B_VALIDATE_API_KEY="${E2B_VALIDATE_API_KEY:-false}"
  ARGS+=( --harbor-use-local-trial --harbor-env-import-path harbor.environments.e2b:E2BEnvironment )
elif [ "$MODE" = kuberl ]; then
  ARGS+=( --harbor-server-url "$KUBE_RL" --harbor-max-retries "${HARBOR_MAX_RETRIES:-3}" )
else
  echo "ERROR: MODE must be 'local' or 'kuberl' (got '$MODE')"; exit 1
fi

# --- deployment ---
if [ "$DEPLOY" = colocate ]; then
  ARGS+=( --colocate )
elif [ "$DEPLOY" = disagg ]; then
  ROLLOUT_GPUS="${ROLLOUT_GPUS:?DEPLOY=disagg needs ROLLOUT_GPUS (dedicated rollout GPUs)}"
  echo "[check] disaggregated: total GPUs = actor($GPUS) + rollout($ROLLOUT_GPUS) = $((GPUS + ROLLOUT_GPUS)); RayCluster must have them."
  ARGS+=( --rollout-num-gpus "$ROLLOUT_GPUS" )
else
  echo "ERROR: DEPLOY must be 'colocate' or 'disagg' (got '$DEPLOY')"; exit 1
fi

# --- optional extras ---
[ "$APPLY_CHAT_TEMPLATE" = 1 ] && ARGS+=( --apply-chat-template )
[ -n "$HARBOR_AGENT_KWARGS" ] && ARGS+=( --harbor-agent-kwargs "$HARBOR_AGENT_KWARGS" )

echo "[run] MODE=$MODE DEPLOY=$DEPLOY preset=$MODEL_PRESET TP=$TP PP=$PP GPUS=$GPUS adapter=$HEAD_IP:${HARBOR_ADAPTER_PORT:-18001} sglang_mem=$SGLANG_MEM apply_chat_template=$APPLY_CHAT_TEMPLATE"
exec python train_remote_agent.py "${ARGS[@]}" "$@"
