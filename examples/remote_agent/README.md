# Remote-Agent RL with Harbor

Train a **multi-turn agent** with slime. The agent runs in-process through
[Harbor](https://github.com/alibaba/harbor)'s `Trial`, and its OpenAI API calls are
served by an **in-process `OpenAIAdapter`** that captures token-level data
(`token_ids`, logprobs, loss mask) at generation time for RL.

The adapter runs inside the `RolloutManager` actor (where `generate_with_harbor`
executes), so all concurrent trials share one adapter endpoint and tokens are
captured in place — **no separate proxy actor and no post-hoc token
reconstruction**. The adapter reaches the SGLang engines through the sglang
router that slime starts automatically (`--sglang-router-ip/-port`).

## Architecture

```
Slime Ray Cluster
┌────────────────────────────────┐
│  train_remote_agent.py         │
│    └── RolloutManager          │
│         └── generate_rollout   │
│              └── generate_with_harbor
│                   │            │
│                   ▼            │
│  ┌──────────────────────────┐ │
│  │  OpenAIAdapter           │ │   endpoint A: agent's OpenAI calls
│  │  (in-process, aiohttp)   │◀┼── (OPENAI_BASE_URL, Bearer=session_id)
│  │  + TrajectoryManager     │ │
│  │  → sglang router (HTTP)  │─┼── endpoint B: token generation
│  └──────────┬───────────────┘ │
│             │ runs the agent   │
│             ▼                  │
│  ┌──────────────────────────┐ │
│  │  harbor Trial (in-proc)  │ │
│  │  → Agent (e.g. swe-agent)│ │
│  └──────────────────────────┘ │
└────────────────────────────────┘
```

**How it works**

1. **Adapter startup** — on the first `generate_with_harbor` call, an
   `OpenAIAdapter` (aiohttp) is started lazily inside the `RolloutManager` actor,
   bound to a fixed port on the head node (`--harbor-adapter-port`). It reads the
   sglang router address from `args.sglang_router_ip/port`.
2. **Generate** — `generate_with_harbor` replaces the default generate function.
   For each sample it opens an adapter session keyed by the sample's
   `session_id`, points the agent at the adapter via `OPENAI_BASE_URL` (with the
   session id carried as the `OPENAI_API_KEY` Bearer token), runs the agent's
   `harbor.trial.Trial` in-process, then calls `finish_session(sid)` to drain the
   captured trajectory into training `Sample`s.
3. **Token capture** — each turn's messages are rendered to token ids and sent to
   sglang `/generate`; `TrajectoryManager` records the exact `output_ids` and
   logprobs. Generated tokens get `loss_mask=1`; prompt/tool/user context gets
   `loss_mask=0` — the key to multi-turn agent RL.

## Requirements

- The `harbor` package (drives the external agent):
  `pip install git+https://github.com/alibaba/harbor.git`
- Docker available on the node, for the default `local_docker` environment.
- A model checkpoint: an HF directory (config + tokenizer + weights) for
  `--hf-checkpoint`, and a Megatron `torch_dist` checkpoint for `--ref-load`
  (convert with `tools/convert_hf_to_torch_dist.py`; see `prepare_model.sh`).

## Quick start

Everything goes through `examples/remote_agent/run_swebench.sh`, configured via
environment variables that are assembled into a `train_remote_agent.py` command.
Two orthogonal switches:

- `DEPLOY`: `colocate` (train + rollout share GPUs) | `disagg` (dedicated rollout GPUs)
- model: `MODEL_PRESET` + parallelism (`TP`/`PP`) + multimodal (`APPLY_CHAT_TEMPLATE`)

```bash
# 0.5B smoke, colocate, 2 GPUs
DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 \
  HF_CKPT=/path/to/Qwen2.5-0.5B-Instruct \
  REF_LOAD=/path/to/Qwen2.5-0.5B_torch_dist \
  bash examples/remote_agent/run_swebench.sh

# disaggregated: dedicated rollout GPUs (total = GPUS + ROLLOUT_GPUS)
DEPLOY=disagg GPUS=4 TP=2 ROLLOUT_GPUS=4 GLOBAL_BATCH_SIZE=2 \
  HF_CKPT=... REF_LOAD=... bash examples/remote_agent/run_swebench.sh
```

Calling `train_remote_agent.py` directly:

```bash
python train_remote_agent.py \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-agent-name swe-agent \
  --harbor-model-name openai/qwen-max \
  --harbor-env-import-path harbor.environments.local_docker:LocalDockerEnvironment \
  --harbor-task-path-template '/data/tasks/{instance_id}' \
  --harbor-adapter-public-host <head-node-ip> --harbor-adapter-port 18001 \
  --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \
  ...   # remaining slime training args
```

## Parameters (`run_swebench.sh`)

### Switch: `DEPLOY`

- `colocate` (default): adds `--colocate`; training (actor) and inference (sglang)
  share the same GPUs via offload/onload. Rollout reuses the actor's `GPUS`; do
  **not** set `ROLLOUT_GPUS`.
- `disagg`: no `--colocate`; inference gets dedicated GPUs
  (`--rollout-num-gpus $ROLLOUT_GPUS`). **Requires `ROLLOUT_GPUS`**; total GPUs =
  `GPUS` (actor) + `ROLLOUT_GPUS` (rollout). `SGLANG_MEM` defaults to `0.5`
  (colocate) / `0.8` (disagg).

### Model

| Var | Default | Maps to / effect | Constraints |
|---|---|---|---|
| `MODEL_PRESET` | `qwen2.5-0.5B` | `source scripts/models/<preset>.sh` → `MODEL_ARGS` | sets `num_query_groups` (caps `TP`), `num_layers` (caps `PP`) |
| `HF_CKPT` | — | `--hf-checkpoint` (tokenizer + config) | must be a complete HF dir (`config.json` + tokenizer). In `disagg`, must be on **shared storage** reachable by all nodes (the sglang engines run on the rollout node) |
| `REF_LOAD` | — | `--ref-load` (Megatron dist checkpoint) | produced by `tools/convert_hf_to_torch_dist.py` |
| `MODEL_NAME` | `openai/Qwen2.5-0.5B-Instruct` | `--harbor-model-name` (name advertised to the agent) | naming only; generation goes through the adapter |
| `APPLY_CHAT_TEMPLATE` | `0` | `1` adds `--apply-chat-template` | set `1` for multimodal models; then `PROMPT_DATA` prompts must be message lists |

### Parallelism and GPUs

| Var | Default | Maps to | Constraints |
|---|---|---|---|
| `TP` | `2` | `--tensor-model-parallel-size` | must divide `num_query_groups` |
| `PP` | `1` | `--pipeline-model-parallel-size` | must divide `num_layers` |
| `GPUS` | `2` | `--actor-num-gpus-per-node` | must be divisible by `TP*PP`; `DP = GPUS/(TP*PP)` |
| `ROLLOUT_GPUS_PER_ENGINE` | `=TP` | `--rollout-num-gpus-per-engine` | usually `= TP` |
| `ROLLOUT_GPUS` | — (disagg only) | `--rollout-num-gpus` | multiple of `ROLLOUT_GPUS_PER_ENGINE`; engines = `ROLLOUT_GPUS/ROLLOUT_GPUS_PER_ENGINE` |
| `SGLANG_MEM` | `0.5`/`0.8` | `--sglang-mem-fraction-static` | lower it under colocate to leave room for training |
| `HARBOR_ADAPTER_PORT` | `18001` | `--harbor-adapter-port` | pick a port outside the sglang-router range (3000–4000) |

### Batch and sampling

| Var | Default | Maps to | Constraints |
|---|---|---|---|
| `NUM_ROLLOUT` | `1` | `--num-rollout` | RL steps; use 1 for a smoke test |
| `ROLLOUT_BATCH_SIZE` | `1` | `--rollout-batch-size` | prompts per step; samples/step = `ROLLOUT_BATCH_SIZE × N_SAMPLES` |
| `N_SAMPLES` | `2` | `--n-samples-per-prompt` | group size (GRPO) |
| `GLOBAL_BATCH_SIZE` | `2` | `--global-batch-size` | must be a multiple of `DP = GPUS/(TP*PP)` |
| `MAX_RESP` | `2048` | `--rollout-max-response-len` | per-turn response cap |

### Task / Harbor / data

| Var | Default | Maps to | Notes |
|---|---|---|---|
| `HARBOR_AGENT_NAME` | `swe-agent` | `--harbor-agent-name` | built-in Harbor agent; for a custom agent use `--harbor-agent-import-path` |
| `HARBOR_AGENT_KWARGS` | — | `--harbor-agent-kwargs <json>` | e.g. `'{"per_instance_cost_limit": 0}'` |
| `HARBOR_ENV_IMPORT_PATH` | `harbor.environments.local_docker:LocalDockerEnvironment` | `--harbor-env-import-path` | the Harbor environment that runs the agent |
| `TASK_NAME` | `astropy__astropy-14309` | default prompt's `task_name` | must exist under `TASK_PATH_TEMPLATE` |
| `SANDBOX_SET` | — (unset) | prompt `metadata.sandbox_set_name` | optional; only if your environment routes trials to named pools |
| `TASK_PATH_TEMPLATE` | `$(pwd)/tasks/{instance_id}` | `--harbor-task-path-template` | `{instance_id}` filled from the sample's `task_name`/`instance_id` |
| `PROMPT_DATA` | `examples/remote_agent/prompts.jsonl` | `--prompt-data` (with `--input-key prompt`) | if absent, the script writes one default sample |

### Harbor CLI flags (for direct `train_remote_agent.py` use)

| Flag | Default | Description |
|---|---|---|
| `--harbor-timeout` | `1800.0` | trial execution timeout (seconds) |
| `--harbor-agent-name` | `None` | built-in Harbor agent name (e.g. `swe-agent`) |
| `--harbor-agent-import-path` | `None` | import path for a custom agent |
| `--harbor-model-name` | `None` | model name passed to the agent |
| `--harbor-agent-kwargs` | `{}` | JSON dict of extra agent kwargs |
| `--harbor-env-overrides` | `{}` | JSON dict of env vars forwarded to the agent |
| `--harbor-env-import-path` | `harbor.environments.local_docker:LocalDockerEnvironment` | environment class import path |
| `--harbor-env-kwargs` | `{}` | JSON dict of environment constructor kwargs |
| `--harbor-task-path-template` | `/home/slime/dataset-tasks/{instance_id}` | task directory template |
| `--harbor-adapter-bind-host` | `0.0.0.0` | bind host for the in-process adapter |
| `--harbor-adapter-port` | `18001` | fixed adapter port (0 = auto) |
| `--harbor-adapter-public-host` | `None` | address the agent uses to reach the adapter (falls back to `LOCAL_IP`) |
| `--harbor-sandbox-set-key` / `--harbor-sandbox-class-key` / `--harbor-sandbox-set-name-template` | see `--help` | optional SandboxSet routing (see below) |

## SandboxSet routing (optional)

If your Harbor environment routes trials to named sandbox pools, you can select
the pool per task. Precedence:

1. an explicit `sandbox_set_name` in the sample metadata;
2. a pod-size class (`--harbor-sandbox-class-key`, default `sandbox_class`) from
   the sample metadata, formatted via `--harbor-sandbox-set-name-template`;
3. the same class read from the task's `task.toml`.

When nothing is found, the key is omitted and the environment uses its default.
Generate a multi-task prompt file with pool names using
`convert_swebench_tasks_to_prompts.py` (add `--prompt-as-messages` for multimodal
models + `--apply-chat-template`).

## Constraints cheat-sheet

1. `GPUS % (TP*PP) == 0`; `DP = GPUS/(TP*PP)`.
2. `GLOBAL_BATCH_SIZE % DP == 0` (else Megatron asserts).
3. `TP` divides `num_query_groups`; `PP` divides `num_layers`.
4. `DEPLOY=disagg` ⇒ set `ROLLOUT_GPUS`; total GPUs = `GPUS + ROLLOUT_GPUS`.
5. `DEPLOY=disagg` ⇒ `HF_CKPT`/`REF_LOAD` must be on shared storage (the sglang
   engines run on the rollout node, not the head).
6. `APPLY_CHAT_TEMPLATE=1` ⇔ `PROMPT_DATA` prompts are message lists.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `global batch size not divisible by micro×DP` | `GLOBAL_BATCH_SIZE` vs `DP = GPUS/(TP*PP)` | make GBS a multiple of DP |
| `TP`-related assert on model load | `TP` does not divide `num_query_groups` | lower `TP` |
| `hf-checkpoint … Unrecognized model / no model_type` | the dir has weights only, no config | use a complete HF checkpoint |
| `Repo id must be in the form 'repo_name'…` (disagg) | `HF_CKPT` on head-local disk, unreachable by the rollout node | put `HF_CKPT` on shared storage |
| `prompt must be a list when processor is not None` | multimodal model but prompt is not a list | set `APPLY_CHAT_TEMPLATE=1` and use message-list prompts |
| CUDA out of memory | `SGLANG_MEM` too high / `PP` too small | lower `SGLANG_MEM`, increase `PP` |
| `ImportError: run_local_trial requires the 'harbor' package` | harbor not installed | `pip install git+https://github.com/alibaba/harbor.git` |
| agent can't reach the LLM / no turns | adapter endpoint unreachable | set `--harbor-adapter-public-host` to a real IP (not 0.0.0.0), open the port |

## Files

- `run_swebench.sh` — launcher (`DEPLOY` × model switches).
- `convert_swebench_tasks_to_prompts.py` — build a prompt JSONL from Harbor task dirs.
- `prepare_model.sh` — convert an HF checkpoint to a Megatron `torch_dist` checkpoint.
- `small_prompts.jsonl` — sample prompt data (the format the pipeline consumes).
