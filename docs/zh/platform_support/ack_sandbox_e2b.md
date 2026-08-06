# 在 ACK sandbox 上运行 slime 远程 agent 强化学习（E2B）

这是一份经过实测、可端到端跑通的流程：每次 rollout 在 **ACK sandbox 内运行真实的编码 agent**，
通过 **E2B 协议** 驱动；agent 的 LLM 调用经 slime 的 OpenAI adapter 路由到 SGLang 引擎，采集 token 级
数据后用 GRPO 训练。

```
 slime (train_remote_agent.py)
   ├─ SGLang 引擎    ← OpenAI adapter（记录 token_ids/logprobs） ←─┐
   ├─ Megatron actors (GRPO)                                   │ OpenAI API（base_url = OpenAI adapter）
   └─ generate_with_harbor ─ harbor Trial ─ E2BEnvironment ─┐  │
                                                            ▼  │
                    ACK sandbox-manager（E2B API） ─ claim ─ Sandbox pod（agent 在此运行）
```

需要准备五部分：**K8s**、**代码（镜像）**、**模型**、**数据**，然后 **运行**。

---

## 1. Kubernetes 准备

### 1.1 Sandbox 组件
在 `sandbox-system` 命名空间安装 ACK sandbox 组件（OpenKruise `agents.kruise.io` CRD +
`ack-sandbox-manager` + `ack-sandbox-gateway`）。slime 使用两个 E2B 端点：

| 平面 | 端点 | 用途 |
|------|------|------|
| 控制面 | `http://sandbox-manager.sandbox-system:8080` | 创建/claim sandbox、模板 |
| 数据面 | `http://sandbox-gateway.sandbox-system:7788` | envd exec/文件系统路由 |

从 manager 的 `--e2b-admin-key` 获取 E2B admin key，运行时通过 `E2B_API_KEY` 传入。

### 1.2 镜像拉取 secret（私有镜像）
若任务镜像在私有仓库，在 sandbox 所在命名空间创建 `dockerconfigjson` secret（如 `acr-pro-registry`）。

### 1.3 SandboxSet 池（每个任务镜像一个）

> **重要：** 当前 ACK controller 的「claim 时按需覆盖镜像」**不可靠** —— in-place 更新镜像会让
> sandbox 卡在 `state=dead / InplaceUpdating`，claim 超时失败。因此请把任务镜像**直接烘进
> SandboxSet**，并使用 `override_claim_image=false`，即**每个任务镜像建一个 SandboxSet**，通过
> prompt 元数据（`sandbox_set_name`）把每条样本路由到对应池。

每个 SandboxSet 必须：
- 设置 `spec.runtimes: [{name: agent-runtime}]` —— 注入 E2B `envd`（否则 `AsyncSandbox.create`
  会因 envd 端口 49983 `connection refused` 失败）；
- 私有镜像需带拉取 secret；
- privileged 运行（swe-bench 镜像需要）。

```yaml
apiVersion: agents.kruise.io/v1alpha1
kind: SandboxSet
metadata:
  name: slime-sbx-pallets-flask-5014   # DNS 合法；由 prompt 元数据引用
  namespace: default
spec:
  replicas: 2                          # >= 该任务的 n_samples_per_prompt
  runtimes:
  - name: agent-runtime                # 注入 envd
  template:
    metadata:
      labels: { app: sandbox }
    spec:
      automountServiceAccountToken: false
      imagePullSecrets:
      - name: acr-pro-registry
      containers:
      - name: main
        command: ["sleep", "infinity"]
        image: <registry>/swebench-verified/pallets-flask-5014:<tag>   # 任务镜像，直接烘入
        imagePullPolicy: IfNotPresent
        securityContext: { privileged: true, runAsUser: 0 }
        resources:
          requests: { cpu: "1", memory: 4Gi, ephemeral-storage: 10Gi }
```

```bash
kubectl apply -f sandboxset-<instance>.yaml
kubectl get sandboxset -n default          # AVAILABLE 达到 REPLICAS
```

### 1.4 Ray 集群
在 Ray 集群（如 KubeRay `RayCluster`）上运行 slime，head pod 拥有 GPU 并使用第 2 步的镜像。
head pod 的 IP 必须能被 sandbox pod 访问（同集群网络）—— 它会作为 `--harbor-adapter-public-host` 传给
agent，使 sandbox 内的 agent 能访问 OpenAI adapter。

---

## 2. 代码准备（构建镜像）

用 `docker/Dockerfile.workspace` 构建镜像，它烘入：slime 代码（本分支）、`sys.path` 上的
Megatron-LM，以及带 ACK/E2B 支持的 harbor：

```dockerfile
# 带 ACK sandbox + E2B 支持的 harbor（sandbox_set_name / override_claim_image，跳过 ACK 不支持的
# 模板构建，per-claim 镜像 key）。会拉取 e2b SDK、litellm 等。
RUN git clone --depth 1 -b feat/ack-sandbox-image-override \
        https://github.com/alibaba/harbor.git /root/harbor && \
    pip install -e /root/harbor && \
    pip install kubernetes_asyncio
```

```bash
docker build -f docker/Dockerfile.workspace -t <registry>/dev/slime:<tag> .
docker push <registry>/dev/slime:<tag>
# 将 RayCluster head 指向该镜像并（重建）head pod
```

本分支同时包含适配新基础镜像（SGLang 0.5.15 / 较新 Megatron）与保证 token 采集正确所需的 slime
修复，镜像构建后无需运行时打补丁：
- `model_provider` / freeze 包装接受 Megatron 的 `config` / `pg_collection`；
- HF/SGLang 参数校验对重命名参数容错；
- `torch_memory_saver` 预加载库按 glob 定位；
- E2B/HARBOR 环境变量转发进 RolloutManager actor；
- adapter 每轮 token 渲染时将 `apply_chat_template` 输出统一转成 `list[int]`。

---

## 3. 模型准备

下载 HF 权重（用于 tokenizer、SGLang 及转换源），再转成 Megatron `torch_dist` 供 `--ref-load`：

```bash
# 1. HF 权重（hf / modelscope 均可）
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# 2. HF -> torch_dist（输出放到持久卷，避免 pod 重启后重转）
cd /root/slime
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
    --save /var/model/Qwen2.5-0.5B_torch_dist
```

其他尺寸使用对应 `scripts/models/*.sh`（如 `qwen2.5-7B.sh`）。注意 TP 必须整除 `num_query_groups`
（Qwen2.5 为 2），因此这些模型 `--tensor-model-parallel-size` ≤ 2。

---

## 4. 数据准备

### 4.1 任务目录
slime 通过 `--harbor-task-path-template` 把每条样本解析到 harbor 任务目录。swe-bench 任务
（含 `task.toml`、`environment/Dockerfile`、`solution/`、`tests/`）位于如
`/var/model-dataset/swe-bench-verified/{instance_id}`。

### 4.2 prompts（`prompts.jsonl`）
每行一个 JSON。`task_name` 选择任务目录；`metadata.sandbox_set_name` 把样本路由到其烘入镜像的
SandboxSet：

```json
{"prompt": "<问题描述>", "task_name": "pallets__flask-5014", "metadata": {"sandbox_set_name": "slime-sbx-pallets-flask-5014"}}
{"prompt": "<问题描述>", "task_name": "astropy__astropy-14309", "metadata": {"sandbox_set_name": "slime-sbx-astropy-14309"}}
```

运行时加 `--input-key prompt`。slime 会把顶层字段（`task_name` 等）并入 `sample.metadata`，
`generate_with_harbor` 解析池时优先读 `metadata.sandbox_set_name`。

---

## 5. 运行

```bash
MODE=local DEPLOY=colocate GPUS=2 TP=2 GLOBAL_BATCH_SIZE=2 \
  E2B_API_KEY=<ACK sandbox admin key> bash examples/remote_agent/run_swebench.sh
```

统一启动器 `examples/remote_agent/run_swebench.sh` 用环境变量设置所需参数（详见 [`remote_agent_run_config.md`](./remote_agent_run_config.md)），关键项：

| 参数 / 环境变量 | 取值 | 说明 |
|------------|-------|-----|
| `E2B_API_URL` | `http://sandbox-manager.sandbox-system:8080` | E2B 控制面 |
| `E2B_SANDBOX_URL` | `http://sandbox-gateway.sandbox-system:7788` | E2B 数据面（router） |
| `E2B_VALIDATE_API_KEY` | `false` | 跳过客户端 key 格式校验 |
| `--harbor-use-local-trial` | — | 进程内运行 harbor Trial |
| `--harbor-env-import-path` | `harbor.environments.e2b:E2BEnvironment` | 使用 E2B 环境 |
| `--harbor-env-kwargs` | `{"override_claim_image": false}` | 使用烘入的池镜像（不做 in-place 覆盖） |
| `--harbor-adapter-public-host` | **head pod IP**（不是 `0.0.0.0`） | 使 sandbox 内 agent 能访问 OpenAI adapter |
| `--harbor-agent-name` | `swe-agent` | 内置 SWE-agent |
| `--sglang-disable-cuda-graph` | — | SGLang 0.5.15 的 prefill CUDA graph 与 slime memory-saver 不兼容 |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | 张量并行必需 |

---

## 6. 跑通的标志

日志中应能看到每条样本的真实 token 采集与一次 GRPO 步：

```
[Harbor][...] Reconstructed sample: response_len=976, num_tokens=1026
rollout.py: perf 0: {'rollout/response_len/mean': ~1000, ...}
model.py:  step 0: {'train/loss': ..., 'train/entropy_loss': 2.10,
                    'train/train_rollout_logprob_abs_diff': 0.80, ...}
```

`entropy_loss` / `train_rollout_logprob_abs_diff` 非零说明 actor 在重建的 token 上做了真实前向。
若模型很小、解不出任务，则所有 reward 为 `0`，GRPO advantage（及 `grad_norm`）为 `0` —— 换更强的
模型、可解的任务或 shaped reward 才能获得学习信号。

---

## 7. 说明与容量

- **每个镜像一个池，`override_claim_image=false`。** 当前 ACK controller 的 per-claim 镜像覆盖有问题
  （已在 `openkruise/agents` 上游跟进），修复后可再评估。
- **显存/内存。** 单节点上 Megatron + SGLang 共置很吃内存；3B/7B 全量启动可能把 200 GiB pod OOM。
  建议从 0.5B 起步，调低 `--sglang-mem-fraction-static`，并把 `--ref-load` 检查点放在持久卷上，
  避免重启后重转。
- **Agent cost limit。** `--harbor-agent-kwargs '{"total_cost_limit":0,...}'` 没问题 —— SWE-agent
  将 `0` 视为*无限制*，而非「无预算」。

---

## 8. FAQ

### `ModuleNotFoundError: No module named 'megatron.training'`（Megatron 训练 actor 中）

现象：任务启动后 `MegatronTrainRayActor` 因 `No module named 'megatron.training'` 导入失败
（或权重转换同样报错）。

原因：`import megatron` 解析到 `dist-packages` 下安装的 **`megatron-core`** 命名空间包，其中**不含**
`megatron.training`；`pip install -e /root/Megatron-LM` 不会把 Megatron-LM 注册进该命名空间；而且
**Ray worker actor 不继承 driver 进程的 `PYTHONPATH`**，所以在启动前 `export PYTHONPATH=/root/Megatron-LM`
只能修好 driver，修不了 actor。

修复（已烘进 `docker/Dockerfile.workspace`）：通过 site-packages 下的 `.pth` 文件把 `/root/Megatron-LM`
加入**所有** Python 进程的 `sys.path`，使 `megatron.training` 在每个 Ray worker 都可导入，无需依赖
`PYTHONPATH`：

```dockerfile
RUN echo /root/Megatron-LM > "$(python -c 'import site; print(site.getsitepackages()[0])')/zzz_megatron_lm.pth"
```

若在运行中的 pod 上遇到（如未重建镜像的临时测试），运行时执行同样一行即可：

```bash
echo /root/Megatron-LM > "$(python -c 'import site; print(site.getsitepackages()[0])')/zzz_megatron_lm.pth"
python -c "from megatron.training.arguments import parse_args; print('ok')"   # 验证
```
