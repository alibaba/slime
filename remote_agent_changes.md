# `feat/remote-agent-adapter` 相对 upstream 的改动汇总

> 本文汇总当前分支 `feat/remote-agent-adapter` 相对 **THUDM upstream/main** 的**全部改动**。
> 分支落后 upstream 8 个 commit、领先 11 个；这 11 个里前 8 个是 Harbor/ACK-E2B 接入的基础工作，
> 最后 3 个是本次的 **adapter 迁移**（去掉 TokenProxy → 进程内 `OpenAIAdapter`）。
> 规模：**30 文件，+4082 / −4**（`git diff upstream/main...HEAD`）。

vanilla upstream slime **没有** `slime/rollout/remote_agent/`、Harbor 接入、ACK sandbox/E2B 支持；
这些都是本分支新增。`slime/agent/`（OpenAIAdapter/TrajectoryManager 本体）**已在 upstream**，本分支只是**复用**它。

---

## 1. 当前架构（迁移后）

```
 slime (train_remote_agent.py = 薄封装 → train.train)  @ RayCluster head
   ├─ SGLang 引擎  ← in-process OpenAIAdapter(:端口, 端点A) ─ sglang router(端点B) ←┐
   ├─ Megatron actors (GRPO)                                                        │ OpenAI API
   └─ generate_with_harbor ─ harbor Trial ─ E2BEnvironment ─┐                       │ (base_url=adapter, Bearer=sid)
        (A: 进程内 local-trial / B: 经 kube-rl server)      ▼                        │
        ACK sandbox-manager ── claim ── Sandbox pod(swe-agent 在此运行) ─────────────┘
```
- **端点A**：进程内 adapter 的 OpenAI 端点，绑定 head 固定端口，供（可能在集群外的）sandbox agent 回连。
- **端点B**：sglang router（`args.sglang_router_ip/port`，slime 自带），adapter 读它访问引擎。
- token 在生成时由 `TrajectoryManager` 直接捕获，`finish_session` 出训练 `Sample`——**无独立 proxy actor、无 REST、无事后 token 重建**。

---

## 2. 改动分区说明

### 2.1 新增核心模块 `slime/rollout/remote_agent/`
| 文件 | 说明 |
|---|---|
| `adapter_service.py`（新增 149 行）| `HarborAdapterService` 单例：进程内起 `OpenAIAdapter`（aiohttp），构造 tokenizer / `sglang_url`（端点B）/ 固定端口（端点A）、router 就绪探测、大 `fork_threshold` 保持 1 trial→1 Sample。|
| `generate.py` | `generate_with_harbor`：`open_session`→注入 `OPENAI_BASE_URL`+`OPENAI_API_KEY=sid`（Bearer）→提交 trial（local/remote）→`finish_session`→返回 `list[Sample]`；`finally drop_session`。含 SandboxSet 路由（`_resolve_sandbox_set_name`/`_read_task_sandbox_class`）与远程重试 `_submit_with_retry`。返回类型 `Sample|list[Sample]`。|
| `harbor_client.py` | `HarborClient`（向 kube-rl `POST /api/v1/runs` 提交 multipart：task tar + JSON `AgentRunRequest`）、`run_local_trial`（进程内跑 `harbor.trial.Trial`）、`HarborAgentConfig`/`HarborVerifierConfig`/`HarborRunResult`。|
| `__init__.py` | 导出上述（**不再**导出任何 proxy 符号）。|

### 2.2 核心 slime 触点（Harbor 接入需要的最小改动）
| 文件 | 改动 |
|---|---|
| `slime/backends/sglang_utils/sglang_engine.py`（+63）| 新增引擎 `generate(...)` Ray RPC（旧 TokenProxy 用；**迁移后 adapter 走 HTTP router，不再用这个 RPC**，见 §4 遗留）。|
| `slime/ray/placement_group.py`（+8）| 把 `E2B_*`/`HARBOR_*` 环境变量转发进 RolloutManager actor（local-trial 的 sandbox 客户端在 actor 内需要）。|
| `slime/ray/rollout.py`（+21）| `get_engine_handles()`、`get_metrics_router_addr()`（后者的调用方已随迁移移除，**方法现无引用**，见 §4）。|
| `slime/utils/data.py`（+10）| `--multimodal-keys` 相关的保留键处理（多模态数据）。|
| `slime/utils/arguments.py`（+162）| `add_harbor_arguments`：全部 `--harbor-*` 参数（server/agent/env/task/**adapter-bind-host/port/public-host**/retry/sandbox-set 路由/local-trial）。|
| `slime/backends/sglang_utils/arguments.py`（+13）| sglang 侧配套参数。|

### 2.3 训练入口 `train_remote_agent.py`
迁移后化简为**薄封装**：`from train import train; train(parse_args())`。不再启动/停止任何 proxy；
adapter 由 `generate_with_harbor` 首次调用懒启动。（修掉了 e2b base 遗留的 `update_tracking_open_metrics` /
`args.critic_train_only` 崩溃——那是旧 train 循环引用了新 base 已删的符号。）

### 2.4 打包 / 环境
| 文件 | 改动 |
|---|---|
| `docker/Dockerfile.workspace`（+54）| workspace 镜像：装 Megatron(-e)、加 `.pth` 让 Ray worker 能 import `megatron.training`、装 slime(-e --no-deps)、shortuuid、harbor（ACK sandbox over E2B）。|
| `.gitmodules`（+3）| 新增子模块 `harbor` → `git@gitlab.alibaba-inc.com:eml/harbor.git`（**内部地址**）。|
| `.gitignore`（+1）| 忽略项。|

### 2.5 文档
- `docs/{en,zh}/platform_support/ack_sandbox_e2b.md`（各 +240 左右）：ACK sandbox + E2B 通用指南（已改述为 adapter 链路）。
- `docs/zh/platform_support/ack_sandbox_e2b_adapter_runbook.md`：adapter 版端到端实测 runbook（mode A/B、27B、踩坑表）。
- `remote_agent_design.md`（+1322）：设计文档（顶部已加"部分过时/TokenProxy→adapter"横幅）。

### 2.6 示例 / 脚本、测试
见 §5 脚本审查表；测试见 `tests/test_remote_agent/`（`test_adapter_capture.py` adapter 冒烟、`test_sandbox_set_routing.py` 路由单测）。

---

## 3. 本次 adapter 迁移的增量（3 个 commit，相对 e2b base）
| commit | 内容 |
|---|---|
| `db96eb8d` | **用进程内 OpenAIAdapter 替换 TokenProxy**：新增 `adapter_service.py`；重写 `generate.py`；args 删 `--harbor-proxy-host/-port`/`--harbor-disable-reconstruct`、加 `--harbor-adapter-*`；删 `proxy.py`(1246 行) + `test_proxy_pipeline.py` + `test_standalone_proxy.py` + `debug_multiturn.py`；文档/示例改述。|
| `9200571f` | `docker/Dockerfile.workspace` 基础镜像 tag 固定到 `nightly-dev-20260804a`。|
| `0b8fe634` | `train_remote_agent.py` 改薄封装（修 e2b base 遗留崩溃）。|

**验证**：0.5B 在 mode A/B 双模式端到端跑通（`train_rollout_logprob_abs_diff≈0.016/0.019`，token 捕获正确）；
27B（`Qwen3.6-27B`，多模态，TP4×PP2、800Gi）产出连贯有效轨迹。详见 adapter runbook。

---

## 4. 已知 gap / 需注意（非阻塞，建议后续处理）
1. **`mbridge` 未进镜像**：`Dockerfile.workspace` 没装 `mbridge`，27B 的 HF→dist 转换/权重桥接会 `ModuleNotFound`。当前靠手动 `pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb108... --no-deps`。**建议在 Dockerfile.workspace 加这行**。
2. **`examples/remote_agent/kubeconfig` 被提交进仓库**（含 clusters/users/contexts）——**安全隐患**，建议移除并加入 `.gitignore`。
3. **遗留死代码（迁移后不再使用，但未清）**：`sglang_engine.py` 的 `generate()` Ray RPC、`rollout.py` 的 `get_metrics_router_addr()`——旧 TokenProxy 路径的产物，adapter 链路不用。可留可删。
4. **`harbor` 子模块指向内部 gitlab**：若要合并回公开 upstream 需处理。
5. **`run_swebench_e2b.sh` / `run_swebench_e2b_disagg.sh` 头部注释仍写 "TokenProxy"**：脚本本身是对的（已用 `--harbor-adapter-public-host`），仅注释措辞过时。

---

## 5. 脚本审查表（`examples/remote_agent/`）

用户已在两个 converter 的头部把"旧 Harbor/TokenProxy 链路"与"ACK E2B adapter 链路"划清。据此分类：

### ✅ 保留（当前 adapter 链路，已测）
| 脚本 | 用途 |
|---|---|
| `run_swebench_e2b.sh` | mode A（进程内 local-trial）colocate。⚠️头部注释"TokenProxy"待改。|
| `run_swebench_e2b_disagg.sh` | mode A 训推分离。⚠️同上。|
| `run_swebench_kuberl_27b.sh` | mode B（kube-rl）27B colocate。|
| `run_swebench_kuberl_27b_disagg.sh` | mode B 27B 训推分离。|
| `convert_swebench_tasks_to_prompts.py` | adapter 链路的 prompt 转换（带 `sandbox_set_name`、`--prompt-as-messages`）。|
| `prepare_model.sh` / `swe-bench-verified.jsonl` / `README.md` | 模型准备 / 样例数据 / 说明。|

### 🗑 疑似废弃（旧 Harbor/TokenProxy 链路，已被上面的 adapter 脚本取代；这一组互相自洽，一起删不影响 adapter 链路）
| 脚本 | 为何废弃 |
|---|---|
| `harbor_qwen.sh` | 旧通用远程示例；占位符 `/path/to/...`、无 preset/MODEL_ARGS、无 sandbox_set 路由。|
| `harbor_local_trial.sh` | 旧 local-trial 示例；注释仍写 TokenProxy、占位符 checkpoint。|
| `run_swebench_harbor.sh` | 旧通用远程；占位符默认、`global-batch-size 1` vs 8 卡 DP 不整除会崩、无 preset。|
| `run_swebench_local.sh` | 旧通用 local；与 `run_swebench_e2b.sh`(mode A) 重叠、无 e2b/sandbox_set 路由。|
| `convert_swebench_to_prompts.py` | 仅服务上面 `run_swebench_harbor.sh` + `harbor_qwen.sh`（旧链路 metadata，无 sandbox_set_name）。|

### 🚩 建议移除（非"脚本失效"，而是不该入库）
| 文件 | 原因 |
|---|---|
| `kubeconfig` | 集群凭证文件被提交，安全隐患；建议删 + `.gitignore`。|

> 我**没有删除任何文件**。上面 🗑 / 🚩 是候选，等你确认哪些可删，我再动手。
