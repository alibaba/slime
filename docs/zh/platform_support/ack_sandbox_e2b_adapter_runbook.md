# Runbook：验证 adapter 版 remote_agent（in-process OpenAIAdapter）在 ACK 上端到端跑通

> 这是**迁移后**（去掉 TokenProxy、改用进程内 `OpenAIAdapter` + `TrajectoryManager`）在 config-sing
> （新加坡）集群上的实测流程与证据。分支 `feat/remote-agent-adapter`，镜像
> `registry-ap-southeast-1-vpc.ack.aliyuncs.com/dev/slime:2.47.1.0b8fe634`。
> 与旧 [`ack_sandbox_e2b_runbook.md`](./ack_sandbox_e2b_runbook.md)（TokenProxy 版）的区别：不再有独立
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

脚本 `examples/remote_agent/run_swebench_e2b.sh` 全参数走环境变量覆盖。两模式共同点：
`--harbor-adapter-public-host $(hostname -i)`（端点A，非 0.0.0.0）、`--harbor-env-kwargs '{"override_claim_image": false}'`、
每 prompt 的 `metadata.sandbox_set_name`。**8 GPU 时 TP=2→DP=4 需 `global-batch-size` 为 4 的倍数；用 2 GPU(TP=2→DP=1)最省，`global-batch-size=2` 即可。**

> 本节 A/B 两模式均为 **colocate**（`--colocate`，训推共享同一组 GPU、轮转 offload）。改用分离式（训推分离）见 §4.1。

### 模式 A —— 进程内 local-trial
```bash
kubectl exec -n default $POD -- bash -lc '
cd /root/slime
export E2B_API_KEY=<admin key>
export HF_CKPT=/root/Qwen2.5-0.5B-Instruct
export GPUS=2 ROLLOUT_GPUS_PER_ENGINE=2 TP=2
export ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=2 NUM_ROLLOUT=1
setsid bash examples/remote_agent/run_swebench_e2b.sh > /root/slime/run.log 2>&1 </dev/null & disown'
```

### 模式 B —— 经 kube-rl server
先冒烟（curl + oracle，验证 claim/envd/exec，**不需 LLM**；注意用 `sandbox_set_name`+`override_claim_image:false`，`true` 会触发 ACK 不支持的模板构建 → `404`）：
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
再跑 slime（内联命令，去掉 `--harbor-use-local-trial`，加 server-url）：
```bash
python train_remote_agent.py "${MODEL_ARGS[@]}" \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-server-url http://kube-rl.kube-rl.svc.cluster.local:8080 --harbor-max-retries 3 \
  --harbor-adapter-public-host $(hostname -i) --harbor-agent-name swe-agent \
  --harbor-model-name openai/Qwen2.5-0.5B-Instruct \
  --harbor-task-path-template "/var/model-dataset/swe-bench-verified/{instance_id}" \
  --harbor-env-kwargs '{"override_claim_image": false}' \
  --hf-checkpoint /root/Qwen2.5-0.5B-Instruct --ref-load /var/model/Qwen2.5-0.5B_torch_dist \
  --prompt-data .../small_prompts.jsonl --input-key prompt --rollout-global-dataset \
  --num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 2 --global-batch-size 2 \
  --colocate --actor-num-gpus-per-node 2 --rollout-num-gpus-per-engine 2 \
  --tensor-model-parallel-size 2 --sglang-mem-fraction-static 0.5 --sglang-disable-cuda-graph ... --advantage-estimator grpo ...
```

### 4.1 分离式部署（训推分离，不带 `--colocate`）

> 本节为参数规则推导的改法（链路其余部分与 A/B 完全一致），成功判据同下文“成功证据”。

分离式 = 去掉 `--colocate`，训练卡（`--actor-num-gpus-per-node`）与推理卡（`--rollout-num-gpus`）分开划，
**总 GPU = actor 卡 + rollout 卡**（head 只有 8 卡，超出需给 RayCluster 加 worker 组）。
专用脚本：`examples/remote_agent/run_swebench_e2b_disagg.sh`（与 colocate 版唯一区别：无 `--colocate`，改为
`--rollout-num-gpus $ROLLOUT_GPUS`，`SGLANG_MEM_FRACTION` 默认 0.8）。与 colocate 的差异：

| 项 | colocate | 分离式 |
|---|---|---|
| 脚本 | `run_swebench_e2b.sh`（`--colocate`） | `run_swebench_e2b_disagg.sh`（需 `ROLLOUT_GPUS`）|
| 训推切换 | 轮转 offload/onload | 无，天然并行 |
| `SGLANG_MEM_FRACTION` | 需给训练留显存（0.4~0.5） | rollout 独占卡，可调高（如 0.8）|
| `global-batch-size` | 同左 | 仍须为 DP 的倍数，DP = actor 卡 ÷ TP |

**模式 A（local-trial）分离式，0.5B，共 4 卡**（actor 2 卡 TP=2 → DP=1；rollout 2 卡 TP=2 一个引擎）：
```bash
kubectl exec -n default $POD -- bash -lc '
cd /root/slime
export E2B_API_KEY=<admin key>
export HF_CKPT=/root/Qwen2.5-0.5B-Instruct
export GPUS=2 TP=2 ROLLOUT_GPUS=2 ROLLOUT_GPUS_PER_ENGINE=2 SGLANG_MEM_FRACTION=0.8
export ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=2 NUM_ROLLOUT=1
setsid bash examples/remote_agent/run_swebench_e2b_disagg.sh > /root/slime/run_disagg.log 2>&1 </dev/null & disown'
```

**模式 B（kube-rl）分离式**：内联命令把 `--colocate --actor-num-gpus-per-node 2 ...` 换成：
```bash
  --actor-num-gpus-per-node 2 --rollout-num-gpus 2 --rollout-num-gpus-per-engine 2 \
```
（去掉 `--colocate`，其余不变）。

**27B 分离式（8 卡刚好）**：actor 4 卡 TP=4 + rollout 4 卡 TP=4：
```bash
export MODEL_PRESET=qwen3.5-27B HF_CKPT=/root/Qwen3.6-27B REF_LOAD=/root/Qwen3.6-27B_torch_dist MODEL_NAME=openai/Qwen3.6-27B
export GPUS=4 TP=4 ROLLOUT_GPUS=4 ROLLOUT_GPUS_PER_ENGINE=4 SGLANG_MEM_FRACTION=0.8
export ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=2 NUM_ROLLOUT=1
setsid bash examples/remote_agent/run_swebench_e2b_disagg.sh --apply-chat-template > /root/slime/run_27b_disagg.log 2>&1 </dev/null &
```
> 27B 主机内存要求不变（仍需 head `memory` 800Gi）。rollout 卡数想更多只能给 RayCluster 加 worker pod（yaml 加 `workerGroupSpecs`，同样 toleration + `nvidia.com/gpu`），Ray 会自动把多出的 bundle 调度过去。

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

27B mode-A 启动命令（8 GPU / TP4 / 低 mem-fraction 留余量）：
```bash
export MODEL_PRESET=qwen3.5-27B HF_CKPT=/root/Qwen3.6-27B REF_LOAD=/root/Qwen3.6-27B_torch_dist MODEL_NAME=openai/Qwen3.6-27B
export GPUS=8 TP=4 ROLLOUT_GPUS_PER_ENGINE=4 SGLANG_MEM_FRACTION=0.4
export ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=2 NUM_ROLLOUT=1
setsid bash examples/remote_agent/run_swebench_e2b.sh --apply-chat-template > /root/slime/run_27b.log 2>&1 </dev/null &
```

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

> 本 runbook 是唯一的 ACK+E2B 实测 runbook，已取代旧 `ack_sandbox_e2b_runbook.md`（TokenProxy 版）。
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
