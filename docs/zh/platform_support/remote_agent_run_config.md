# `run_swebench.sh` 参数详解

`examples/remote_agent/run_swebench.sh` 一个脚本覆盖全部远程 agent RL 组合，全部通过**环境变量**配置，
最终拼成 `train_remote_agent.py` 的命令行。本文逐参数说明**取值、作用（映射的 CLI flag）、以及参数之间的关联约束**。

- 用法：`VAR=值 ... bash examples/remote_agent/run_swebench.sh [额外flag透传]`
- 三个正交开关：`MODE`（提交方式）、`DEPLOY`（GPU 布局）、模型（`MODEL_PRESET`+并行度+多模态）。
- 端到端流程见 [`ack_sandbox_e2b_adapter_runbook.md`](./ack_sandbox_e2b_adapter_runbook.md)。

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
