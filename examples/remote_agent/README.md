# Slime RemoteAgent with Harbor

This example demonstrates how to run Slime training with agents executed
on a remote [Harbor](https://github.com/agent-arena/harbor) server, or
locally via in-process Trial execution.  In both modes an in-process
`OpenAIAdapter` captures token-level data (token_ids, logprobs) for RL training.

The adapter runs inside the `RolloutManager` actor (where `generate_with_harbor`
executes), so every concurrent trial shares one adapter endpoint and token
capture happens in-process — no separate proxy actor and no post-hoc token
reconstruction. The adapter reaches the SGLang engines through the sglang
router (`--sglang-router-ip/-port`, started automatically by slime).

## Architecture

### Remote Mode (Harbor Server)

```
Slime Ray Cluster                              Harbor Server (remote)
┌──────────────────────────────┐              ┌──────────────────────┐
│  train_remote_agent.py       │              │                      │
│    └── RolloutManager        │  HTTP POST   │  /api/v1/runs        │
│         └── generate_rollout │─────────────▶│    ├── pack task     │
│              └── generate()  │              │    ├── start Docker  │
│                   │          │              │    └── run Agent     │
│                   ▼          │              └──────────┬───────────┘
│  ┌──────────────────────┐   │                         │
│  │  OpenAIAdapter       │   │   OpenAI SDK            │
│  │  (in-process, aiohttp)│◀─┼───(base_url,Bearer sid)─┘
│  │  + TrajectoryManager │   │
│  │  → sglang router HTTP│   │
│  └──────────────────────┘   │
└──────────────────────────────┘
```

### Local Trial Mode

```
Slime Ray Cluster
┌──────────────────────────────┐
│  train_remote_agent.py       │
│    └── RolloutManager        │
│         └── generate_rollout │
│              └── generate()  │
│                   │          │
│                   ▼          │
│  ┌──────────────────────┐   │
│  │  OpenAIAdapter       │   │
│  │  (in-process, aiohttp)│  │
│  │  + TrajectoryManager │   │
│  │  → sglang router HTTP│   │
│  └──────────┬───────────┘   │
│             │ OpenAI SDK    │
│             ▼               │
│  ┌──────────────────────┐   │
│  │  LocalTrialClient    │   │
│  │  harbor.Trial.run()  │   │
│  │  → Agent (in-proc)   │   │
│  └──────────────────────┘   │
└──────────────────────────────┘
```

## Quick Start

All runs go through the unified launcher `run_swebench.sh` (three switches:
`MODE`=local|kuberl, `DEPLOY`=colocate|disagg, and the model via env). Full
parameter reference: see the "参数详解" section below.

### Via kube-rl server (MODE=kuberl)

```bash
MODE=kuberl DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 \
  bash examples/remote_agent/run_swebench.sh
```

### In-process local trial (MODE=local)

```bash
MODE=local DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 \
  E2B_API_KEY=<ACK sandbox admin key> bash examples/remote_agent/run_swebench.sh
```

**E2B server address**: the E2B endpoints default to the in-cluster sandbox stack
(`E2B_API_URL=http://sandbox-manager.sandbox-system:8080` = control plane / the E2B
"server"; `E2B_SANDBOX_URL=http://sandbox-gateway.sandbox-system:7788` = data plane).
Override them only if your sandbox stack lives elsewhere (different namespace / out of
cluster):

```bash
MODE=local DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 \
  E2B_API_KEY=<admin key> \
  E2B_API_URL=http://<sandbox-manager-host>:8080 \
  E2B_SANDBOX_URL=http://<sandbox-gateway-host>:7788 \
  bash examples/remote_agent/run_swebench.sh
```
(`MODE=kuberl` needs no E2B vars — the kube-rl server owns the sandbox; its address is
`KUBE_RL`, default `http://kube-rl.kube-rl.svc.cluster.local:8080`.)

Or manually:

```bash
python train_remote_agent.py \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-use-local-trial \
  --harbor-agent-name swe-agent \
  --harbor-model-name openai/qwen-max \
  --harbor-task-path-template '/data/tasks/{instance_id}' \
  --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \
  ... (other training params)
```

## K8s permissions / `kubeconfig` (non-E2B only)

The E2B modes (kube-rl / local-trial) need **no** kubeconfig — slime only talks HTTP
to `sandbox-manager` / `kube-rl`, which own the sandbox Pod lifecycle. A `kubeconfig`
is only needed in **non-E2B** environments where harbor creates K8s **Pods/Jobs**
directly. Prefer **in-cluster RBAC** over a mounted kubeconfig: grant the RayCluster
head ServiceAccount (`rayclustertest`) a `Role`/`RoleBinding` for pods/jobs in the
target namespace, so the in-pod client uses the auto-mounted SA token (no kubeconfig
file, no `KUBECONFIG`). Concrete YAML: see the Runbook §8.4 section below.

## Parameters

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--harbor-timeout` | `1800.0` | Timeout (seconds) for task execution |
| `--harbor-agent-name` | `None` | Built-in Harbor agent name (e.g. `swe-agent`) |
| `--harbor-agent-import-path` | `None` | Python import path for custom agent |
| `--harbor-model-name` | `None` | LLM model name for the agent |
| `--harbor-agent-kwargs` | `{}` | JSON dict of extra agent kwargs |
| `--harbor-env-overrides` | `{}` | JSON dict of env vars for the agent |
| `--harbor-env-import-path` | `harbor.environments.local_docker:LocalDockerEnvironment` | Environment class import path |
| `--harbor-env-kwargs` | `{}` | JSON dict of environment kwargs |
| `--harbor-task-path-template` | `/home/slime/dataset-tasks/{instance_id}` | Task directory template |
| `--harbor-adapter-bind-host` | `0.0.0.0` | Bind host for the in-process OpenAI adapter |
| `--harbor-adapter-port` | `18001` | Fixed adapter port (avoid the router's 3000-4000 range; 0 = auto) |
| `--harbor-adapter-public-host` | `None` | Head-node address the sandbox uses to reach the adapter (falls back to `LOCAL_IP`) |
| `--harbor-max-retries` | `3` | Max retry attempts on failure (remote mode only) |
| `--harbor-retry-base-delay` | `2.0` | Base delay (seconds) for exponential backoff (remote mode only) |
| `--harbor-use-local-trial` | `False` | **Run Trial locally instead of remote Harbor** |

### Remote Mode Only Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--harbor-server-url` | `http://localhost:8080` | Harbor Agent Run server URL |
| `LOCAL_IP` env var | `0.0.0.0` | IP the Harbor containers can reach |

## How It Works

1. **Adapter startup**: `generate_with_harbor` lazily starts an `OpenAIAdapter`
   (aiohttp) inside the `RolloutManager` actor on the first call, bound to a
   fixed port (`--harbor-adapter-port`) on the head node. It reads the sglang
   router address from `args.sglang_router_ip/port` (started automatically by
   slime in the same process).

2. **Generate function**: `generate_with_harbor` replaces the default generate
   function. For each sample:
   - Opens an adapter session keyed by the sample's `session_id` (`sid`)
   - Points the agent at the adapter via `OPENAI_BASE_URL`, carrying the `sid`
     as `OPENAI_API_KEY` (Bearer)
   - **Remote mode**: Submits the task to the Harbor HTTP server and waits for completion (with retry)
   - **Local mode**: Runs `harbor.trial.trial.Trial` directly in the current process
   - Calls `finish_session(sid)` to drain the captured trajectory into training
     `Sample` objects (tokens, `rollout_log_probs`, `loss_mask`)

3. **Token capture**: each turn's messages are rendered to token ids and sent to
   sglang `/generate`; the `TrajectoryManager` records exact `output_ids` and
   logprobs. Generated tokens get `mask=1` (participate in loss), while prompt /
   tool / user context gets `mask=0`. This is the key to RL training with
   multi-turn agents.

## Mode Comparison

| Aspect | Remote Mode | Local Trial Mode |
|--------|-------------|------------------|
| Harbor server required | Yes | No |
| Agent execution | Remote Docker/K8s | In-process |
| `LOCAL_IP` needed | Yes | No (uses 127.0.0.1) |
| Retry support | Yes (exponential backoff) | No (direct execution) |
| Debugging | Harder (remote logs) | Easy (pdb, print) |
| Production-ready | Yes | Development/testing only |
| `harbor` package needed | No (HTTP only) | Yes (Trial class) |

## Harbor Agent Configuration

### Using a built-in agent

```bash
--harbor-agent-name swe-agent \
--harbor-model-name openai/qwen-max
```

### Using a custom agent

```bash
--harbor-agent-import-path my_agents.swe:SWEAgent \
--harbor-model-name openai/qwen-max \
--harbor-agent-kwargs '{"total_cost_limit": 0, "per_instance_cost_limit": 0}'
```

## Choosing a Mode

- **Development / debugging**: Use `--harbor-use-local-trial`. You can
  set breakpoints in the agent code and step through execution.
- **Production training**: Use remote mode with a Harbor server. This
  gives Docker isolation, resource management, and the ability to scale
  across multiple machines.


---

## 参数详解：`run_swebench.sh`


`examples/remote_agent/run_swebench.sh` 一个脚本覆盖全部远程 agent RL 组合，全部通过**环境变量**配置，
最终拼成 `train_remote_agent.py` 的命令行。本文逐参数说明**取值、作用（映射的 CLI flag）、以及参数之间的关联约束**。

- 用法：`VAR=值 ... bash examples/remote_agent/run_swebench.sh [额外flag透传]`
- 三个正交开关：`MODE`（提交方式）、`DEPLOY`（GPU 布局）、模型（`MODEL_PRESET`+并行度+多模态）。
- 端到端流程见下文「端到端 Runbook」节。

---

## 1. 开关

### `MODE` — trial 提交方式
- **取值**：`local`（默认）｜`kuberl`
- **作用**：
  - `local`：进程内跑 harbor Trial（`--harbor-use-local-trial` + `--harbor-env-import-path harbor.environments.e2b:E2BEnvironment`）；sandbox 由 slime 进程内的 E2B SDK 直接 claim。
  - `kuberl`：把 trial POST 给 kube-rl server（`--harbor-server-url $KUBE_RL --harbor-max-retries N`）；sandbox 由 kube-rl worker 起。
- **关联**：
  - `local` ⇒ **必须** `E2B_API_KEY`，并用到 `E2B_API_URL`/`E2B_SANDBOX_URL`/`E2B_VALIDATE_API_KEY`。
  - `kuberl` ⇒ slime 侧**不需要**任何 `E2B_*`；用到 `KUBE_RL`、`HARBOR_MAX_RETRIES`。
  - 两种模式的 token 捕获路径完全相同（进程内 adapter），差异只在"谁起 sandbox"。

### `DEPLOY` — 训练/推理的 GPU 布局
- **取值**：`colocate`（默认）｜`disagg`
- **作用**：
  - `colocate`：加 `--colocate`，训练(actor)与推理(sglang)**共享同一组 GPU**，靠显存 offload/onload 轮转。
  - `disagg`：不加 `--colocate`，推理独占 GPU（`--rollout-num-gpus $ROLLOUT_GPUS`）。
- **关联**：
  - `disagg` ⇒ **必须** `ROLLOUT_GPUS`；**总卡数 = `GPUS`(actor) + `ROLLOUT_GPUS`(rollout)**，超过 head 的 8 卡需给 RayCluster 加 worker 组。
  - `colocate` 下 rollout 复用 actor 的 `GPUS`，**不要**设 `ROLLOUT_GPUS`。
  - `SGLANG_MEM` 默认随此开关变（colocate 0.5 / disagg 0.8），见下。

---

## 2. 模型

| 参数 | 取值 / 默认 | 作用（映射）| 关联与约束 |
|---|---|---|---|
| `MODEL_PRESET` | 默认 `qwen2.5-0.5B`；如 `qwen3.5-27B` | `source scripts/models/<preset>.sh` 得到 `MODEL_ARGS`（层数/hidden/GQA/vocab 等结构参数）| 决定 `num_query_groups`（约束 `TP` 上限）、`num_layers`（约束 `PP`）、是否多模态 |
| `HF_CKPT` | 默认 `/root/Qwen2.5-0.5B-Instruct` | `--hf-checkpoint`：tokenizer + config | **必须是完整 HF 目录**（含 `config.json`/tokenizer）；**放本地盘**（`/root/...`），ossfs 随机读极慢。脚本会检查 `config.json` 存在 |
| `REF_LOAD` | 默认 `/var/model/Qwen2.5-0.5B_torch_dist` | `--ref-load`：Megatron dist 参考权重 | 由 `tools/convert_hf_to_torch_dist.py` 转出；脚本检查 `latest_checkpointed_iteration.txt` 存在。27B 也放本地盘 |
| `MODEL_NAME` | 默认 `openai/Qwen2.5-0.5B-Instruct` | `--harbor-model-name`：告诉 swe-agent 用的模型名 | 仅命名用；实际生成走 adapter，不影响权重 |
| `APPLY_CHAT_TEMPLATE` | `0`（默认）｜`1` | `1` 时加 `--apply-chat-template` | **多模态模型必须置 1**，且 `PROMPT_DATA` 的 `prompt` **必须是消息 list**（否则 `prompt must be a list when processor is not None`）。用 `convert_swebench_tasks_to_prompts.py --prompt-as-messages` 生成 |

> **mbridge 依赖**：`qwen3_5`（27B）等模型的权重桥接需要 `mbridge` 包（镜像未内置）。脚本会 `import mbridge` 做**软告警**；缺失时 27B 会在 `update_weights` 阶段失败。安装：`pip install "git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps`。0.5B 不需要。

---

## 3. 并行度与 GPU 资源

| 参数 | 取值 / 默认 | 作用（映射）| 关联与约束 |
|---|---|---|---|
| `TP` | 默认 `2` | `--tensor-model-parallel-size` | **必须整除 `num_query_groups`**：0.5B=2 ⇒ `TP≤2`；27B=4 ⇒ `TP≤4` |
| `PP` | 默认 `1` | `--pipeline-model-parallel-size` | **必须整除 `num_layers`**（27B 64 层 ⇒ PP∈{1,2,4,8}）；PP>1 可降低单卡显存，用于大模型 |
| `GPUS` | 默认 `2` | `--actor-num-gpus-per-node`（actor 卡数）| **必须能被 `TP*PP` 整除**；数据并行 **DP = `GPUS/(TP*PP)`** |
| `ROLLOUT_GPUS_PER_ENGINE` | 默认 `=TP` | `--rollout-num-gpus-per-engine`（每个 sglang 引擎的 GPU）| 一般 = `TP`（引擎与模型同 TP）；须整除可用推理卡数 |
| `ROLLOUT_GPUS` | disagg **必填** | `--rollout-num-gpus`（独占推理卡数）| 仅 `DEPLOY=disagg`；应为 `ROLLOUT_GPUS_PER_ENGINE` 的倍数；引擎数 = `ROLLOUT_GPUS/ROLLOUT_GPUS_PER_ENGINE` |
| `SGLANG_MEM` | colocate `0.5` / disagg `0.8` | `--sglang-mem-fraction-static`（sglang 静态显存占比）| colocate 下越高留给训练越少；**27B colocate 建议 0.2**（给 27B 优化器留显存，否则 CUDA OOM）。disagg 推理独占卡可高 |
| `HARBOR_ADAPTER_PORT` | 默认 `18001` | `--harbor-adapter-port`（端点A 端口）| **避开 sglang router 的 3000–4000**；须能被（集群外的）sandbox 访问 |

> **派生量**：`MP = TP*PP`（模型并行度）；`DP = GPUS/MP`（数据并行度）；colocate 总卡 = `GPUS`；disagg 总卡 = `GPUS + ROLLOUT_GPUS`。脚本会对 `GPUS%MP` 和 `GLOBAL_BATCH_SIZE%DP` 做告警。

---

## 4. 批量与采样

| 参数 | 取值 / 默认 | 作用（映射）| 关联与约束 |
|---|---|---|---|
| `NUM_ROLLOUT` | `1` | `--num-rollout`（RL 步数）| 冒烟用 1；训练调大 |
| `ROLLOUT_BATCH_SIZE` | `1` | `--rollout-batch-size`（每步 prompt 数）| 每步样本数 = `ROLLOUT_BATCH_SIZE × N_SAMPLES` |
| `N_SAMPLES` | `2` | `--n-samples-per-prompt`（每 prompt 采样数）| **同一任务并发的 sandbox 数**：SandboxSet 池 `replicas` **必须 ≥ 它**，否则 claim 超时 |
| `GLOBAL_BATCH_SIZE` | `2` | `--global-batch-size`（训练全局 batch）| **必须是 DP=`GPUS/(TP*PP)` 的倍数**，否则 Megatron 断言失败；且不应超过每步样本数 |
| `MAX_RESP` | `2048` | `--rollout-max-response-len` | 影响单条轨迹上限与 KV/显存 |

---

## 5. 任务 / Harbor / 数据

| 参数 | 取值 / 默认 | 作用 | 关联与约束 |
|---|---|---|---|
| `HARBOR_AGENT_NAME` | `swe-agent` | `--harbor-agent-name`（harbor 内置 agent）| — |
| `HARBOR_AGENT_KWARGS` | 空 | 非空时加 `--harbor-agent-kwargs <json>` | 如 `'{"per_instance_cost_limit": 0, "total_cost_limit": 0}'` |
| `TASK_NAME` | `astropy__astropy-14309` | 默认 prompt 里的 `task_name` | 须在 `TASK_PATH_TEMPLATE` 下存在对应任务目录 |
| `SANDBOX_SET` | `slime-sbx-astropy-14309` | 默认 prompt 的 `metadata.sandbox_set_name` | **必须**指向一个已建、镜像与任务配套、带 `runtimes:[{name: agent-runtime}]` 的 SandboxSet 池 |
| `TASK_PATH_TEMPLATE` | `/var/model-dataset/swe-bench-verified/{instance_id}` | `--harbor-task-path-template` | `{instance_id}` 由样本的 `task_name`/`instance_id` 填充 |
| `PROMPT_DATA` | `examples/remote_agent/prompts.jsonl` | `--prompt-data`（配 `--input-key prompt`）| 不存在时脚本自动写**一条**默认样本；格式随 `APPLY_CHAT_TEMPLATE`（0=string / 1=消息 list）。多任务用 converter 生成 |

## 6. MODE 专属

| 参数 | 适用 | 默认 | 作用 |
|---|---|---|---|
| `KUBE_RL` | kuberl | `http://kube-rl.kube-rl.svc.cluster.local:8080` | `--harbor-server-url` |
| `HARBOR_MAX_RETRIES` | kuberl | `3` | `--harbor-max-retries` |
| `E2B_API_KEY` | local **必填** | — | ACK sandbox admin key（= sandbox-manager 的 `--e2b-admin-key`）|
| `E2B_API_URL` | local | `http://sandbox-manager.sandbox-system:8080` | 控制面 |
| `E2B_SANDBOX_URL` | local | `http://sandbox-gateway.sandbox-system:7788` | 数据面 |
| `E2B_VALIDATE_API_KEY` | local | `false` | 是否校验 key |

---

## 7. 关联约束速查（硬性，违反会报错/告警）

1. `GPUS % (TP*PP) == 0`；`DP = GPUS/(TP*PP)`。
2. `GLOBAL_BATCH_SIZE % DP == 0`（否则 Megatron 断言：`global batch size not divisible by micro×DP`）。
3. `TP ≤ num_query_groups` 且整除它（0.5B≤2 / 27B≤4）。
4. `PP` 整除 `num_layers`（27B: 64）。
5. `DEPLOY=disagg` ⇒ 必填 `ROLLOUT_GPUS`；总卡 = `GPUS+ROLLOUT_GPUS`，RayCluster 要有这么多卡。
6. `DEPLOY=colocate` ⇒ 不设 `ROLLOUT_GPUS`；rollout 复用 `GPUS`。
7. `MODE=local` ⇒ 必填 `E2B_API_KEY`。
8. `APPLY_CHAT_TEMPLATE=1` ⇔ `PROMPT_DATA` 为消息 list（多模态模型两者要一致）。
9. `SandboxSet.replicas ≥ N_SAMPLES`；`SANDBOX_SET`/`TASK_NAME` 与池镜像/数据集配套。
10. `HF_CKPT`/`REF_LOAD` 放本地盘；`REF_LOAD` 需先转换生成。
11. 27B：`mbridge` 已装、`SGLANG_MEM` 低（colocate 0.2）、host `memory≥800Gi`。

---

## 8. 常用配方（复现原来的 4 个脚本）

```bash
# 0.5B / local-trial / colocate / 2 GPU  (原 run_swebench_e2b.sh)
MODE=local DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 E2B_API_KEY=<key> \
  bash examples/remote_agent/run_swebench.sh

# 0.5B / local-trial / 训推分离       (原 run_swebench_e2b_disagg.sh)
MODE=local DEPLOY=disagg GPUS=2 TP=2 ROLLOUT_GPUS=2 GLOBAL_BATCH_SIZE=2 E2B_API_KEY=<key> \
  bash examples/remote_agent/run_swebench.sh

# 27B / kube-rl / colocate / 8 GPU     (原 run_swebench_kuberl_27b.sh)
MODE=kuberl DEPLOY=colocate MODEL_PRESET=qwen3.5-27B \
  HF_CKPT=/root/Qwen3.6-27B REF_LOAD=/root/Qwen3.6-27B_torch_dist MODEL_NAME=openai/Qwen3.6-27B \
  GPUS=8 TP=4 PP=2 SGLANG_MEM=0.2 APPLY_CHAT_TEMPLATE=1 \
  bash examples/remote_agent/run_swebench.sh

# 27B / kube-rl / 训推分离              (原 run_swebench_kuberl_27b_disagg.sh)
MODE=kuberl DEPLOY=disagg MODEL_PRESET=qwen3.5-27B \
  HF_CKPT=/root/Qwen3.6-27B REF_LOAD=/root/Qwen3.6-27B_torch_dist MODEL_NAME=openai/Qwen3.6-27B \
  GPUS=8 TP=4 PP=2 ROLLOUT_GPUS=4 SGLANG_MEM=0.8 APPLY_CHAT_TEMPLATE=1 \
  HARBOR_AGENT_KWARGS='{"per_instance_cost_limit": 0, "total_cost_limit": 0}' \
  bash examples/remote_agent/run_swebench.sh
```

---

## 9. 参数 → 常见现象（排错）

| 现象 | 多半是哪个参数 | 处理 |
|---|---|---|
| `global batch size ... not divisible by micro×DP` | `GLOBAL_BATCH_SIZE` vs `GPUS/(TP*PP)` | 让 GBS 为 DP 的倍数 |
| 模型加载/切分报错、`TP` 相关 assert | `TP` 未整除 `num_query_groups` | 降 `TP`（0.5B≤2 / 27B≤4）|
| `prompt must be a list when processor is not None` | `APPLY_CHAT_TEMPLATE`/`PROMPT_DATA` 不匹配 | 多模态：置 `APPLY_CHAT_TEMPLATE=1` 且 prompt 用消息 list |
| CUDA out of memory（27B）| `SGLANG_MEM` 太高 / `PP` 太小 | colocate 降到 0.2、加 `PP=2` |
| pod OOMKilled（主机内存）| RayCluster head `memory` | 调大到 ~800Gi（27B）|
| `ModuleNotFoundError: mbridge` | 缺 mbridge | 装固定版 mbridge |
| sandbox claim 超时 | `N_SAMPLES` > 池 `replicas`，或 `SANDBOX_SET` 不存在 | 扩池 replicas / 建对应池 |
| kube-rl `404 page not found` | `override_claim_image`（脚本已固定 false）/ 未预建池 | 用预建池 + `sandbox_set_name` |
| swe-agent 连不到 LLM / 无 turn | 端点A 不可达（`HARBOR_ADAPTER_PORT`/网络）| 确认 head IP 非 0.0.0.0 且端口对 sandbox 可达 |
| 加载模型很慢/卡住 | `HF_CKPT`/`REF_LOAD` 在 ossfs | 先 `cp` 到 `/root` 本地盘 |


---

## 端到端 Runbook（ACK sandbox + E2B 实测）


> 这是**迁移后**（去掉 TokenProxy、改用进程内 `OpenAIAdapter` + `TrajectoryManager`）在 config-sing
> （新加坡）集群上的实测流程与证据。分支 `feat/remote-agent-adapter`，镜像
> `registry-ap-southeast-1-vpc.ack.aliyuncs.com/dev/slime:2.47.1.0b8fe634`。
> 与旧 TokenProxy 版的区别：不再有独立
> proxy actor / REST / token 重建；token 在生成时由进程内 adapter 直接捕获，`finish_session` 出 `Sample`。

链路：

```
 slime (train_remote_agent.py = 薄封装 → train.train)  @ RayCluster head (config-sing)
   ├─ SGLang 引擎  ← in-process OpenAIAdapter(:18001, 端点A) ─ sglang router(端点B) ←┐
   ├─ Megatron actors (GRPO)                                                          │ OpenAI API
   └─ generate_with_harbor ─ harbor Trial ─ E2BEnvironment ─┐                         │ (base_url=adapter, Bearer=sid)
        (A: 进程内 local-trial / B: 经 kube-rl server)      ▼                          │
        ACK sandbox-manager(:8080) ── claim ── Sandbox pod(swe-agent 在此运行) ────────┘
```

## 0. 实测环境值

| 项 | 值 |
|---|---|
| kubeconfig | `~/.kube/config-sing` |
| RayCluster | `raycluster-slime`，ns `default`，head 8×GPU(~96GB)、200Gi 内存(27B 需扩到 800Gi) |
| head IP（`--harbor-adapter-public-host`） | `hostname -i`（如 `10.0.38.199`；**不能 0.0.0.0**）|
| 镜像 | `registry-ap-southeast-1-vpc.ack.aliyuncs.com/dev/slime:2.47.1.<commit>`（内部 CI 按 commit 打 tag）|
| sandbox 控制面 / 数据面 | `sandbox-manager.sandbox-system:8080` / `sandbox-gateway.sandbox-system:7788` |
| kube-rl server | `http://kube-rl.kube-rl.svc.cluster.local:8080`（环境 importPath=`harbor.environments.e2b:E2BEnvironment`）|
| E2B admin key | `sandbox-manager` 部署参数 `--e2b-admin-key=...`（运行时以 `E2B_API_KEY` 传入）|
| PVC | `ym-models`→`/var/model`（**ossfs**，随机读慢）、`ym-dataset`→`/var/model-dataset` |
| 镜像拉取 secret | `acr-pro-registry`（default 与 kube-rl 两 ns 都有）|

> **关键**：`/var/model` 是 **ossfs**（对象存储 FUSE），大模型随机读极慢。大模型转换/加载**先 `cp` 到本地
> overlay 盘 `/root`（节点约 2TB、pod overlay 约 1.5T 空闲）** 再用，快一个量级。

## 1. 拉起 RayCluster

`/tmp/raycluster-slime.yaml`（要点）：`serviceAccount rayclustertest` + `ClusterRoleBinding→kuberay-operator`；
head 容器 `command:[service, ssh, start]`、`nvidia.com/gpu:8`、`memory`（0.5B 用 200Gi；**27B 用 800Gi**）、
`imagePullSecrets:[regcred-cn-hangzhou, regcred-ap-southeast, ap-southeast-1-vpc]`、
`toleration key=node-role.alibabacloud.com/lingjun`、挂 `ym-models→/var/model`、`ym-dataset→/var/model-dataset`。

```bash
export KUBECONFIG=~/.kube/config-sing
kubectl apply -f /tmp/raycluster-slime.yaml
kubectl wait --for=condition=Ready pod -n default -l ray.io/node-type=head --timeout=180s
POD=$(kubectl get pod -n default -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n default $POD -- bash -lc 'hostname -i; cd /root/slime && git log -1 --oneline'
```

## 2. K8s 前置

- sandbox 栈（`sandbox-system`：manager+gateway）已装；E2B admin key 从 manager 参数取。
- **SandboxSet 池**（每任务镜像一个，带 `runtimes:[{name: agent-runtime}]` 注入 envd，`acr-pro-registry`）：

```bash
cat <<YAML | kubectl apply -f -
apiVersion: agents.kruise.io/v1alpha1
kind: SandboxSet
metadata: {name: slime-sbx-astropy-14309, namespace: default}
spec:
  replicas: 2                                   # >= n_samples_per_prompt
  runtimes: [{name: agent-runtime}]
  template:
    metadata: {labels: {app: sandbox, slimeset: slime-sbx-astropy-14309}}
    spec:
      automountServiceAccountToken: false
      imagePullSecrets: [{name: acr-pro-registry}]
      containers:
      - {name: main, command: [sleep, infinity], imagePullPolicy: IfNotPresent,
         image: "yueming-acr-registry.cn-hongkong.cr.aliyuncs.com/swebench-verified/astropy-astropy-14309:20260601",
         securityContext: {privileged: true, runAsUser: 0},
         resources: {requests: {cpu: "1", memory: 4Gi, ephemeral-storage: 10Gi}}}
YAML
kubectl get sandboxset -n default          # AVAILABLE 达到 REPLICAS
```
> 镜像名映射：任务 `astropy__astropy-14309` → 镜像 `astropy-astropy-14309`（也可从任务 `task.toml` 的 `[environment].docker_image` 读）。

## 3. 模型 / 数据（0.5B 冒烟）

`/var/model/Qwen2.5-0.5B-Instruct` 只有权重、缺 config/tokenizer → 下载完整 HF ckpt 到本地：
```bash
kubectl exec -n default $POD -- bash -lc 'hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct'
```
`/var/model/Qwen2.5-0.5B_torch_dist` 已是转好的 Megatron dist ckpt（`--ref-load`）。
`examples/remote_agent/small_prompts.jsonl`（每行一 JSON，`task_name`+`metadata.sandbox_set_name` 路由到池）：
```json
{"prompt": "Fix the bug ...", "task_name": "astropy__astropy-14309", "metadata": {"sandbox_set_name": "slime-sbx-astropy-14309"}}
```
该文件可由 `examples/remote_agent/convert_swebench_tasks_to_prompts.py` 从任务目录生成（在 head pod 内跑，两路径均可配；
27B 多模态加 `--prompt-as-messages`）：
```bash
kubectl exec -n default $POD -- python /root/slime/examples/remote_agent/convert_swebench_tasks_to_prompts.py \
  --dataset-root /var/model-dataset/swe-bench-verified \
  --output /root/slime/examples/remote_agent/small_prompts.jsonl --limit 1
```
要点：
- prompt 正文优先取 `task.toml` 的 `problem_statement`；**新版任务布局放在同级 `instruction.md`**（脚本自动 fallback，实测已遇）。
- `sandbox_set_name` = 前缀（默认 `slime-sbx-`）+ 短名；`--set-name-from suffix`（默认，取 `__` 后半，如 `flask-5014`，与 §2 命名一致）或 `image`（取 `docker_image` 短名）。**生成名必须与已建 SandboxSet 同名**。
- `--limit N` 按目录字典序取前 N 个；指定单任务可只拷该任务目录后用 `--dataset-root` 指向它（如 `pallets__flask-5014` 实测）。

## 4. 运行 —— 两种模式（都实测通过）

统一启动器 `examples/remote_agent/run_swebench.sh` 用环境变量切换：`MODE`(local|kuberl)、`DEPLOY`(colocate|disagg)、模型。
参数详解见上文「参数详解」节。两模式共同点：`--harbor-adapter-public-host $(hostname -i)`（端点A，非 0.0.0.0）、`override_claim_image:false`、每 prompt 的 `metadata.sandbox_set_name`。
**DP = `GPUS/(TP*PP)`，`GLOBAL_BATCH_SIZE` 须为 DP 的倍数**（2 GPU、TP2→DP1，`GLOBAL_BATCH_SIZE=2` 即可）。

> 本节 A/B 均为 **colocate**（训推共享 GPU、轮转 offload）。训推分离见 §4.1。

### 模式 A —— 进程内 local-trial（`MODE=local`）
```bash
kubectl exec -n default $POD -- bash -lc '
cd /root/slime
export MODE=local DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 E2B_API_KEY=<admin key>
setsid bash examples/remote_agent/run_swebench.sh > /root/slime/run.log 2>&1 </dev/null & disown'
```

### 模式 B —— 经 kube-rl server（`MODE=kuberl`）
先冒烟（curl + oracle，验证 claim/envd/exec，**不需 LLM**；用 `sandbox_set_name`+`override_claim_image:false`，`true` 会触发 ACK 不支持的模板构建 → `404`）：
```bash
cd /var/model-dataset/swe-bench-verified && tar czf /tmp/t.tgz astropy__astropy-14309
cat > /tmp/meta.json <<'EOF'
{"job_id":"smoke","task_id":"smoke-1","task_path":"astropy__astropy-14309",
 "agent":{"name":"oracle","import_path":"harbor.agents.oracle:OracleAgent"},
 "environment_kwargs":{"sandbox_set_name":"slime-sbx-astropy-14309","override_claim_image":false}}
EOF
KUBE_RL=http://kube-rl.kube-rl.svc.cluster.local:8080
curl -s -X POST $KUBE_RL/api/v1/runs/async -F "metadata=</tmp/meta.json" -F "task_archive=@/tmp/t.tgz"   # → queued → completed, reward 1.0
```
再跑 slime（`MODE=kuberl`，slime 侧无需 E2B）：
```bash
kubectl exec -n default $POD -- bash -lc '
cd /root/slime
export MODE=kuberl DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2
setsid bash examples/remote_agent/run_swebench.sh > /root/slime/run.log 2>&1 </dev/null & disown'
```

### 4.1 分离式部署（训推分离，不带 `--colocate`）

> 本节为参数规则推导的改法（链路其余部分与 A/B 完全一致），成功判据同下文“成功证据”。

分离式 = `DEPLOY=disagg`（脚本去掉 `--colocate`，加 `--rollout-num-gpus $ROLLOUT_GPUS`），训练卡与推理卡分开划，
**总 GPU = `GPUS`(actor) + `ROLLOUT_GPUS`(rollout)**（head 只有 8 卡，超出需给 RayCluster 加 worker 组）。与 colocate 的差异：

| 项 | colocate | 分离式 |
|---|---|---|
| 开关 | `DEPLOY=colocate` | `DEPLOY=disagg`（需 `ROLLOUT_GPUS`）|
| 训推切换 | 轮转 offload/onload | 无，天然并行 |
| `SGLANG_MEM` | 需给训练留显存（0.4~0.5；27B 0.2） | rollout 独占卡，可调高（如 0.8）|
| `GLOBAL_BATCH_SIZE` | 须为 DP 的倍数 | 同左，DP = `GPUS/(TP*PP)` |

**模式 A（local-trial）分离式，0.5B，共 4 卡**（actor 2 卡 TP=2→DP=1；rollout 2 卡 TP=2 一个引擎）：
```bash
kubectl exec -n default $POD -- bash -lc '
cd /root/slime
export MODE=local DEPLOY=disagg GPUS=2 TP=2 ROLLOUT_GPUS=2 SGLANG_MEM=0.8 GLOBAL_BATCH_SIZE=2 E2B_API_KEY=<admin key>
setsid bash examples/remote_agent/run_swebench.sh > /root/slime/run_disagg.log 2>&1 </dev/null & disown'
```

**模式 B（kube-rl）分离式**：同上把 `MODE=local ... E2B_API_KEY=...` 换成 `MODE=kuberl`（其余不变）。

**27B 分离式（8 卡刚好）**：actor 4 卡 TP=4 + rollout 4 卡 TP=4：
```bash
export MODE=kuberl DEPLOY=disagg MODEL_PRESET=qwen3.5-27B \
  HF_CKPT=/root/Qwen3.6-27B REF_LOAD=/root/Qwen3.6-27B_torch_dist MODEL_NAME=openai/Qwen3.6-27B \
  GPUS=4 TP=4 ROLLOUT_GPUS=4 SGLANG_MEM=0.8 APPLY_CHAT_TEMPLATE=1 GLOBAL_BATCH_SIZE=2
setsid bash examples/remote_agent/run_swebench.sh > /root/slime/run_27b_disagg.log 2>&1 </dev/null &
```
> 注意 27B actor 用 `TP4×PP2=8` 卡时无法再拆出 rollout 卡；上面 actor 用 `TP4×PP1=4` 卡（DP1）+ rollout 4 卡凑满 8。主机内存仍需 head `memory` 800Gi。rollout 想要更多卡只能给 RayCluster 加 `workerGroupSpecs`（同 toleration + `nvidia.com/gpu`）。

### 成功证据（两模式一致的因果链，带时间戳）
```
[Harbor] adapter service ready: adapter_url=http://<head>:18001 sglang_url=http://<head>:<router>
[Harbor][sid] Submitting trial (local/remote mode)... → Trial completed status=completed → Finished samples=1
SGLangEngine … "POST /generate HTTP/1.1" 200 OK            # agent 的调用到达 adapter→sglang
Finish rollout: [... <tool_call>{...}</tool_call> ...]      # 捕获到 agent 真实生成
rollout 0: response_lengths=.., rollout_log_probs=-0.x, total_lengths=..
step 0:    train_rollout_logprob_abs_diff=0.016~0.019, grad_norm=0.0, global_batch_size=2
```
**判读**：`finish_session` 出非空 `Sample` + `train_rollout_logprob_abs_diff≈0.02`（actor 重算 logprob 与捕获值几乎一致）
= token/loss_mask 捕获正确、GRPO 前向真实执行。`grad_norm=0` 是 0.5B 解不出题、reward 无方差所致（模型问题，非链路）。
> adapter 对成功的 `/v1/chat/completions` access log 被 `FilteredAccessLogger` 抑制；"agent→adapter" 由随后的 `/generate` 调用 + 非空捕获间接但充分证明。

## 5. 换大模型（Qwen3.6-27B，多模态）实测要点

27B `Qwen3.6-27B` 是 `qwen3_5` **多模态**（`Qwen3_5ForConditionalGeneration`），但有文本 spec/preset `qwen3.5-27B.sh`；`num_query_groups=4` → **TP≤4**（8 GPU 用 TP=4）。踩过的坑与解法：

1. **转换需 `mbridge`**（仓库声明依赖，`docker/Dockerfile:38` 固定到 `ISEEKYAN/mbridge@89eb108`，workspace 镜像未装）：
   ```bash
   pip install "git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps
   ```
2. **ossfs 读慢 → 先 copy 到本地**：`cp -r /var/model/Qwen3.6-27B /root/Qwen3.6-27B`（~61G，约 2.5min），再转换/加载。
3. **转换（写本地）**：
   ```bash
   source scripts/models/qwen3.5-27B.sh
   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
     --hf-checkpoint /root/Qwen3.6-27B --save /root/Qwen3.6-27B_torch_dist
   # 完成标志：日志 "successfully saved" + latest_checkpointed_iteration.txt == release
   ```
   vision 权重由 `qwen3_5` bridge 干净丢弃（`dist_ckpt_strictness=assume_ok_unexpected`）。**务必等转换真正结束再启动训练**（否则读到半成品 ckpt）。
4. **多模态数据断言**：模型带 processor → slime 数据管线要求 `prompt` 是 **list**。把 jsonl 的 `prompt` 改成消息列表并加 `--apply-chat-template`（纯文本→ processor 产出无多模态输入，走文本路径）：
   ```json
   {"prompt": [{"role":"user","content":"Fix the bug ..."}], "task_name": "astropy__astropy-14309", "metadata": {"sandbox_set_name": "slime-sbx-astropy-14309"}}
   ```
5. **200GiB pod 内存不够**：27B Megatron init 时 8 个 actor ×~20GB + sglang → 194.8/200GB → Ray OOM kill。GPU 显存没问题（TP4，13GB/卡余 80GB）。**把 RayCluster head `memory` 调到 800Gi（节点 ~2TB）后重建 head pod**。用 `0b8fe634` 镜像重建则 `train_remote_agent.py` 修复已内置，无需再打补丁；但 `/root` 本地状态（mbridge、ckpt、HF 副本）会丢，需重装/重拷/重转（HF 仍在 ossfs 持久）。

6. **GPU 显存不够（CUDA OOM）**：27B colocate 时 sglang 与 27B 优化器抢显存；**TP 上限 4，单靠 TP 差 ~2GB**。解法：加 `PP=2`（拆层降单卡权重）+ 降 `SGLANG_MEM=0.2`。

27B colocate 启动命令（8 GPU / **TP4×PP2** / `SGLANG_MEM=0.2`，实测通过的配置）：
```bash
export MODE=kuberl DEPLOY=colocate MODEL_PRESET=qwen3.5-27B \
  HF_CKPT=/root/Qwen3.6-27B REF_LOAD=/root/Qwen3.6-27B_torch_dist MODEL_NAME=openai/Qwen3.6-27B \
  GPUS=8 TP=4 PP=2 SGLANG_MEM=0.2 APPLY_CHAT_TEMPLATE=1 GLOBAL_BATCH_SIZE=2
setsid bash examples/remote_agent/run_swebench.sh > /root/slime/run_27b.log 2>&1 </dev/null &
```
（`MODE=local` 亦可，加 `E2B_API_KEY=<admin key>` 即可。）

## 6. 踩坑速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `ImportError: update_tracking_open_metrics` / `AttributeError: critic_train_only` | e2b base 的 `train_remote_agent.py` 引用了已删符号/旧 train 循环 | 已修：`train_remote_agent.py` 改为薄封装 `from train import train`（commit `0b8fe634`）|
| `global batch size (2) not divisible by micro(1)×DP(4)` | 8 GPU TP2 → DP4 | `global-batch-size` 取 DP 的倍数;或用 2 GPU(DP1)|
| `hf-checkpoint … Unrecognized model / no model_type` | 该目录只有权重、缺 config | 下载完整 HF ckpt(`hf download`)|
| kube-rl `404 page not found` | `override_claim_image:true` 触发 ACK 不支持的模板构建 | 预建池 + `sandbox_set_name` + `override_claim_image:false`|
| `mbridge` ModuleNotFound（转换时）| workspace 镜像未装 | pip 装仓库固定版 mbridge |
| 转换 `world_size <= args.num_layers` TypeError | `MODEL_ARGS` 数组没进 `bash -c` 子壳 | 写成脚本 `source preset` 后再跑 |
| 转换卡在权重加载不动 | 从 ossfs 随机读 | 先 `cp` 到本地盘再转 |
| `prompt must be a list when processor is not None` | 多模态模型加载了 processor | prompt 改消息 list + `--apply-chat-template`|
| pod OOMKilled(200GiB) | 大模型 Megatron init 主机内存超限 | 调大 RayCluster head `memory`(节点 2TB)重建 |
| 分离式启动卡住、placement group 一直 pending | actor 卡 + rollout 卡 超过集群可用 GPU | 缩减两边卡数，或给 RayCluster 加 `workerGroupSpecs` |

## 7. 清理
```bash
kubectl delete raycluster raycluster-slime -n default
kubectl delete sandboxset slime-sbx-astropy-14309 -n default
```

---

## 8. 附录：K8s / 镜像准备细节（合并自旧 TokenProxy 版 runbook）

> 以下是从旧 TokenProxy 版 runbook 合并进来、仍适用的准备细节。
> 以下是从旧版合并进来、仍适用的准备细节。

### 8.1 镜像拉取 secret（任务镜像在私有 HK 仓库）
在 sandbox 所在 ns（`default`；kube-rl 模式还需 `kube-rl` ns）建 `acr-pro-registry`：
```bash
kubectl create secret docker-registry acr-pro-registry \
  --docker-server=yueming-acr-registry.cn-hongkong.cr.aliyuncs.com \
  --docker-username='<user>' --docker-password='<pwd>' -n default
```

### 8.2 workspace 镜像关键片段（`docker/Dockerfile.workspace`）
```dockerfile
RUN cd /root/Megatron-LM && pip install -e .
# pip -e 不会把 megatron.training 注册进 megatron-core 命名空间，且 Ray worker 不继承
# driver 的 PYTHONPATH —— 用 .pth 把 Megatron-LM 加进所有 python 进程的 sys.path
RUN echo /root/Megatron-LM > "$(python -c 'import site; print(site.getsitepackages()[0])')/zzz_megatron_lm.pth"
RUN pip install -e . --no-deps          # slime（本分支）
# 带 ACK sandbox + E2B 支持的 harbor（sandbox_set_name / override_claim_image）
RUN git clone --depth 1 -b feat/ack-sandbox-image-override https://github.com/alibaba/harbor.git /root/harbor \
    && pip install -e /root/harbor && pip install kubernetes_asyncio
```
> ⚠️ 该镜像**未内置 `mbridge`**（27B 权重桥接需要）；建议在此加一行
> `pip install "git+https://github.com/ISEEKYAN/mbridge.git@89eb108..." --no-deps`，否则 27B 需运行时手动装。

### 8.3 kube-rl 模式 B 自动建的池形态（ns `start-rl`/`kube-rl`，供对照）
```yaml
apiVersion: agents.kruise.io/v1alpha1
kind: SandboxSet
metadata: {name: swe-bench-swebench-verified-pallets-flask-5014, namespace: start-rl}
spec:
  replicas: 1
  template:
    spec:
      serviceAccountName: kube-rl-server        # 无 runtimes 时需手动补 agent-runtime
      containers:
      - {name: main, command: [sleep, infinity],
         image: rl-lab-registry-vpc.ap-southeast-1.cr.aliyuncs.com/swebench-verified/pallets-flask-5014:20260601,
         securityContext: {privileged: true, runAsUser: 0}}
```
> 实测：kube-rl 与 mode A 共用同一 `sandbox-manager`，故在 `default` ns 预建的池两种模式都能 claim；
> `override_claim_image:true` 会触发 ACK 不支持的模板构建（404），务必用**预建池 + `sandbox_set_name` + `override_claim_image:false`**。

### 8.4 K8s 权限 / kubeconfig（仅**非 E2B** 环境需要）

- **E2B 环境（本 runbook 的两种模式）不需要 kubeconfig**：slime/harbor 只通过 HTTP 调用 `sandbox-manager`（mode A）或 `kube-rl` server（mode B），sandbox 的 Pod 生命周期由它们负责，slime 侧无需直接访问 K8s API。
- **非 E2B 环境**（harbor 的环境后端直接在 K8s 里**创建 Pod / Job** 来跑 agent，例如 kubernetes-native 环境）：进程需要 K8s API 访问权限，这就是 `examples/remote_agent/kubeconfig` 的用途。
- **推荐做法：不要挂静态 kubeconfig（凭证入库/入镜像有安全隐患），改用 RayCluster ServiceAccount 的 in-cluster RBAC**——给 head 用的 SA（本 runbook 里是 `rayclustertest`）在目标 ns 建 `Role`/`RoleBinding`，Pod 内的客户端会自动用挂载的 SA token（in-cluster config），无需任何 kubeconfig 文件、也无需设 `KUBECONFIG`。

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: {name: slime-pod-job-manager, namespace: default}   # = 创建 agent Pod/Job 的目标 ns
rules:
- {apiGroups: [""],      resources: [pods, pods/log, pods/exec], verbs: [create, get, list, watch, delete]}
- {apiGroups: ["batch"], resources: [jobs],                      verbs: [create, get, list, watch, delete]}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: {name: slime-pod-job-manager, namespace: default}
roleRef: {apiGroup: rbac.authorization.k8s.io, kind: Role, name: slime-pod-job-manager}
subjects:
- {kind: ServiceAccount, name: rayclustertest, namespace: default}   # RayCluster head 的 SA
```
> 应用后即可**从仓库/镜像移除 `kubeconfig`**（并加入 `.gitignore`）；如目标 ns 与 head 不同 ns，Role/RoleBinding 建到目标 ns、subject 仍指向 head SA 即可。


---

## 容量说明与 FAQ（合并自通用指南）

- **每镜像一个池、`override_claim_image=false`**：当前 ACK controller 的 per-claim 镜像覆盖有问题（openkruise/agents 上游跟进中），修复前按镜像拆池。
- **显存/内存**：单节点 Megatron+SGLang 共置很吃内存；3B/7B 全量共置可能把 200GiB pod OOM。建议从 0.5B 起步、调低 `SGLANG_MEM`、`--ref-load` 放持久卷。
- **Agent cost limit**：`--harbor-agent-kwargs '{"total_cost_limit":0,...}'` 没问题——SWE-agent 把 `0` 当**无限制**，不是「无预算」。

### FAQ：`ModuleNotFoundError: No module named 'megatron.training'`（Megatron 训练 actor 中）
`import megatron` 解析到 dist-packages 的 `megatron-core`（不含 `megatron.training`）；`pip install -e /root/Megatron-LM` 不注册进该命名空间；且 Ray worker actor **不继承** driver 的 `PYTHONPATH`。修复（已烘进 `docker/Dockerfile.workspace`）：用 `.pth` 把 Megatron-LM 加进所有 Python 进程的 `sys.path`：
```bash
echo /root/Megatron-LM > "$(python -c 'import site; print(site.getsitepackages()[0])')/zzz_megatron_lm.pth"
python -c "from megatron.training.arguments import parse_args; print('ok')"   # 验证
```
