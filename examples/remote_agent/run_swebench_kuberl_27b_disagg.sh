#!/usr/bin/env bash
# Qwen3.6-27B RL over kube-rl (remote mode B) with DISAGGREGATED deployment
# (训推分离): actor and rollout run on SEPARATE GPUs -- no --colocate, no
# offload/onload rotation. Runnable from /root/slime.
#
# GPU accounting: total GPUs = GPUS (actor, TP4xPP2=8) + ROLLOUT_GPUS (>=4, TP4).
# A single 8-GPU head is NOT enough; add a worker group to the RayCluster
# (workerGroupSpecs with >= ROLLOUT_GPUS GPUs, same toleration) so Ray can
# schedule the extra bundles.
#
# What it does: starts slime training with the in-process OpenAIAdapter; each
# sample is submitted to the kube-rl server (which runs the swe-agent in an ACK
# sandbox), the agent's OpenAI calls come back to the adapter, tokens are
# captured, and one GRPO step runs.
#
# The colocated counterpart is examples/remote_agent/run_swebench_kuberl_27b.sh.
#
# Prerequisites (fail-fast checks below will tell you if any are missing):
#   1. RayCluster head pod with memory >= ~800Gi (27B Megatron init needs it)
#      PLUS enough cluster GPUs for GPUS + ROLLOUT_GPUS (worker group, see above).
#   2. mbridge installed (weight bridging):
#        pip install "git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps
#   3. HF ckpt on LOCAL disk (ossfs is slow):   cp -r /var/model/Qwen3.6-27B /root/Qwen3.6-27B
#   4. Converted Megatron dist ckpt on LOCAL disk:
#        source scripts/models/qwen3.5-27B.sh
#        PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
#          --hf-checkpoint /root/Qwen3.6-27B --save /root/Qwen3.6-27B_torch_dist
#   5. A SandboxSet pool for the task (ns default), with runtimes:[{name: agent-runtime}]:
#        kubectl get sandboxset -n default   # e.g. slime-sbx-astropy-14309, AVAILABLE>=N_SAMPLES
#   6. kube-rl server on the E2B environment:
#        curl -s $KUBE_RL/api/v1/capabilities/environments  # importPath == harbor.environments.e2b:E2BEnvironment
#
# Usage:
#   cd /root/slime && bash examples/remote_agent/run_swebench_kuberl_27b_disagg.sh
# Override anything via env, e.g. NUM_ROLLOUT=3 N_SAMPLES=4 bash examples/remote_agent/run_swebench_kuberl_27b_disagg.sh
set -euo pipefail
cd "$(dirname "$0")/../.." 2>/dev/null || cd /root/slime   # -> repo root (/root/slime)

# ---------------------------------------------------------------------------
# Config (all overridable via env)
# ---------------------------------------------------------------------------
MODEL_PRESET="${MODEL_PRESET:-qwen3.5-27B}"                       # scripts/models/<preset>.sh -> MODEL_ARGS
HF_CKPT="${HF_CKPT:-/root/Qwen3.6-27B}"                           # tokenizer/config (LOCAL disk)
REF_LOAD="${REF_LOAD:-/root/Qwen3.6-27B_torch_dist}"             # Megatron dist ckpt (LOCAL disk)
MODEL_NAME="${MODEL_NAME:-openai/Qwen3.6-27B}"                    # name advertised to the swe-agent
KUBE_RL="${KUBE_RL:-http://kube-rl.kube-rl.svc.cluster.local:8080}"
SANDBOX_SET="${SANDBOX_SET:-slime-sbx-astropy-14309}"            # pre-created pool with the task image
TASK_NAME="${TASK_NAME:-astropy__astropy-14309}"                 # dataset dir under the task-path template
TASK_PATH_TEMPLATE="${TASK_PATH_TEMPLATE:-/var/model-dataset/swe-bench-verified/{instance_id}}"
PROMPT_DATA="${PROMPT_DATA:-/root/slime/examples/remote_agent/prompts_27b.jsonl}"

# parallelism / memory (27B actor: TP=4 [num_query_groups=4] x PP=2 = 8 GPUs; DP=1)
GPUS="${GPUS:-8}"; TP="${TP:-4}"; PP="${PP:-2}"
# disaggregated resources: rollout runs on its OWN GPUs (>= ROLLOUT_GPUS_PER_ENGINE, TP4)
ROLLOUT_GPUS="${ROLLOUT_GPUS:?set ROLLOUT_GPUS (dedicated rollout GPUs, e.g. 4)}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-4}"
SGLANG_MEM="${SGLANG_MEM:-0.8}"                                   # rollout GPUs are dedicated: no need to leave room for training
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"; N_SAMPLES="${N_SAMPLES:-2}"; GBS="${GBS:-2}"; MAX_RESP="${MAX_RESP:-2048}"

# ---------------------------------------------------------------------------
# Prereq checks (fail fast with a clear hint)
# ---------------------------------------------------------------------------
[ -f "$HF_CKPT/config.json" ] || { echo "ERROR: HF ckpt not found at $HF_CKPT. Run: cp -r /var/model/Qwen3.6-27B $HF_CKPT"; exit 1; }
[ -f "$REF_LOAD/latest_checkpointed_iteration.txt" ] || { echo "ERROR: dist ckpt not found at $REF_LOAD. Convert first (see header step 4)."; exit 1; }
python -c "import mbridge" 2>/dev/null || { echo "ERROR: mbridge not importable. pip install the pinned mbridge (see header step 2)."; exit 1; }
TOTAL_GPUS=$((GPUS + ROLLOUT_GPUS))
[ "$((TOTAL_GPUS % ROLLOUT_GPUS_PER_ENGINE))" -eq 0 ] || echo "[warn] ROLLOUT_GPUS($ROLLOUT_GPUS) not divisible by ROLLOUT_GPUS_PER_ENGINE($ROLLOUT_GPUS_PER_ENGINE)"
echo "[check] disaggregated needs $TOTAL_GPUS cluster GPUs in total (actor=$GPUS + rollout=$ROLLOUT_GPUS); make sure the RayCluster has them (head + worker group)."

# prompt data: message-list format (the 27B is multimodal -> slime requires list + --apply-chat-template)
# can be generated by: examples/remote_agent/convert_swebench_tasks_to_prompts.py --prompt-as-messages
if [ ! -f "$PROMPT_DATA" ]; then
  echo "[info] writing default prompt file: $PROMPT_DATA"
  cat > "$PROMPT_DATA" <<JSON
{"prompt": [{"role": "user", "content": "Fix the bug described in the task instruction in the mounted repository. Read the instruction, locate the root cause, and edit the source so the failing tests pass."}], "task_name": "$TASK_NAME", "metadata": {"sandbox_set_name": "$SANDBOX_SET"}}
JSON
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
source "scripts/models/${MODEL_PRESET}.sh"                       # defines MODEL_ARGS
export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True          # cut fragmentation (27B is tight on GPU)

HEAD_IP="$(hostname -i | awk '{print $1}')"                      # endpoint A the kube-rl sandbox dials back to
echo "[run] 27B via kube-rl (disaggregated): adapter_host=$HEAD_IP server=$KUBE_RL pool=$SANDBOX_SET TP=$TP PP=$PP rollout_gpus=$ROLLOUT_GPUS"

# NOTE: mode B needs NO E2B_* on the slime side (the kube-rl worker runs the sandbox).
python train_remote_agent.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CKPT" --ref-load "$REF_LOAD" \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --harbor-server-url "$KUBE_RL" --harbor-max-retries 3 \
  --harbor-adapter-public-host "$HEAD_IP" --harbor-adapter-port 18001 \
  --harbor-agent-name swe-agent --harbor-model-name "$MODEL_NAME" \
  --harbor-agent-kwargs '{"per_instance_cost_limit": 0, "total_cost_limit": 0}' \
  --harbor-task-path-template "$TASK_PATH_TEMPLATE" \
  --harbor-env-kwargs '{"override_claim_image": false}' \
  --prompt-data "$PROMPT_DATA" --input-key prompt --apply-chat-template --rollout-global-dataset \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size 1 --n-samples-per-prompt "$N_SAMPLES" \
  --rollout-max-response-len "$MAX_RESP" --rollout-temperature 1.0 --global-batch-size "$GBS" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE" \
  --sglang-mem-fraction-static "$SGLANG_MEM" --sglang-disable-cuda-graph \
  --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP" --sequence-parallel \
  --use-dynamic-batch-size --max-tokens-per-gpu 9216 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl \
  --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 --attention-backend flash --seed 42 \
  "$@"
