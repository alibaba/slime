# 通过 E2B 协议将 slime 与 ACK Sandbox 集成

本文说明如何在阿里云容器服务 Kubernetes 版（ACK）上运行 slime 的 RL rollout，
并使用 **E2B 协议** 作为 slime/Harbor 与集群之间的通信格式。

整体设计为：**先使用一个公共 SandboxSet（通用的预热池），在 RL 训练时于认领
（claim）阶段将镜像替换为实例专属镜像。** 这样可以避免为每个镜像/每个任务单独
构建模板。

涉及两个 API 平面：

| 平面 | API | 时机 | 执行方 |
|------|-----|------|--------|
| 创建池 | Kubernetes sandbox CRD | 训练前，一次性 | 集群管理员 |
| 创建 sandbox（含镜像覆盖） | E2B API | 每个样本，rollout 期间 | slime → Harbor |

```
 slime rollout ── Harbor ── E2B SDK ──► 网关 ── k8s ──► 从公共 SandboxSet 认领
 (generate_with_harbor)     (E2B env)  (ack-sandbox-manager)  └─ 每次认领的镜像 = 任务镜像
```

---

## 前置条件

### 1. 在 ACK 上部署 sandbox 控制器（二选一）

- **`ack-sandbox-controller` + `ack-sandbox-manager`（推荐）** —— ACK 原生方案；
  `ack-sandbox-manager` 内置了兼容 E2B 的网关。
- **OpenKruise** —— 提供 `SandboxSet` / `SandboxClaim` / `Sandbox` CRD
  （`agents.kruise.io/v1alpha1`），Harbor 的 `ACKEnvironment` 可直接对接。

两者都支持按认领覆盖镜像（OpenKruise SandboxClaim 的
`spec.inplaceUpdate.image`；E2B 网关扩展键 `e2b.agents.kruise.io/image`）。

### 2. 实例专属镜像（harbor CLI）

每个任务/实例需要自己的容器镜像，并推送到集群可拉取的镜像仓库。**镜像准备步骤
另有单独文档** —— 简言之，harbor CLI 会基于任务的 Dockerfile（或预构建的
`docker_image`）构建并推送到你的仓库。

---

## 步骤 1 —— 创建一个公共 SandboxSet（训练前）

创建单个通用预热池，`replicas` 按 rollout 并发规模设置。

OpenKruise 示例：

```yaml
apiVersion: agents.kruise.io/v1alpha1
kind: SandboxSet
metadata:
  name: slime-common-pool
  namespace: my-namespace
spec:
  replicas: 32
  template:
    spec:
      containers:
        - name: main
          image: registry.cn-hangzhou.aliyuncs.com/my-repo/sandbox-base:latest  # 通用镜像
          command: ["sleep", "infinity"]
          securityContext: { privileged: true, runAsUser: 0 }
          resources:
            requests: { cpu: "2", memory: "4Gi" }
```

```bash
kubectl apply -f sandboxset.yaml
kubectl get sandboxset -n my-namespace
```

（若使用 `ack-sandbox-controller`，请通过其 CRD 创建等价模板，详见其文档。）

---

## 步骤 2 —— 在每个实例的 `task.toml` 中写入镜像

每个任务在 `[environment]` 下携带自己的镜像：

```toml
[environment]
docker_image = "registry.cn-hangzhou.aliyuncs.com/my-repo/sympy-12096:latest"
cpus = 2
memory_mb = 4096
```

Harbor 会按任务读取该镜像，并将其作为对公共 SandboxSet 的**按认领镜像覆盖**，
无需按镜像单独构建模板。

---

## 步骤 3 —— 通过 Harbor 路径运行训练

slime 的 Harbor 集成（`slime/rollout/remote_agent/`）驱动 Harbor，由 Harbor
创建 sandbox 并应用镜像覆盖。

```bash
python train_remote_agent.py \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-use-local-trial \
  --harbor-agent-name swe-agent \
  --harbor-model-name openai/qwen \
  --harbor-env-import-path harbor.environments.e2b:E2BEnvironment \
  --harbor-env-kwargs '{"sandbox_set_name": "slime-common-pool"}' \
  --harbor-task-path-template '/home/slime/dataset-tasks/{instance_id}' \
  --hf-checkpoint /path/to/model
```

- `--harbor-env-kwargs '{"sandbox_set_name": "slime-common-pool"}'` 用于选择公共
  池。Harbor 的 `E2BEnvironment` 会从该池认领，并通过网关键
  `e2b.agents.kruise.io/image` 传入任务的 `docker_image`（按认领镜像覆盖）。
  也可改用 `harbor.environments.ack:ACKEnvironment` 直接对接 OpenKruise CRD
  （镜像覆盖通过 `spec.inplaceUpdate.image`）。
- Harbor 侧开关（见 harbor 的 `E2BEnvironment` / `ACKEnvironment`）：
  `sandbox_set_name`（公共池）、`override_claim_image`（默认 `true`）。

### 可选：按 Pod 规格分级路由到不同池

若后续按 Pod 规格拆分池，可在每个 `task.toml` 中标注规格类别，由 slime 按任务
解析为 SandboxSet 名称：

```toml
[metadata]
sandbox_class = "large"
```

```bash
  --harbor-sandbox-class-key sandbox_class \
  --harbor-sandbox-set-name-template 'slime-pool-{sandbox_class}' \
  --harbor-sandbox-set-key sandbox_set_name
```

slime 的 `generate_with_harbor` 会读取类别（样本 metadata 或 `task.toml`），按
模板转换后注入到 `environment_kwargs`。若仅使用单个公共池，则无需此项 —— 使用
静态的 `--harbor-env-kwargs` 即可。

---

## 检查清单

| 步骤 | 平面 | 方式 |
|------|------|------|
| 1. ACK 上的 sandbox 控制器 | k8s | ack-sandbox-manager（网关）或 OpenKruise |
| 2. 一个公共 SandboxSet | k8s | `kubectl apply` |
| 3. 实例专属镜像 | —— | harbor CLI（单独文档） |
| 4. 在每个 `task.toml` 写入 `docker_image` | 数据集 | `[environment] docker_image` |
| 5. 运行训练 | slime | `generate_with_harbor` + `--harbor-env-kwargs {"sandbox_set_name": ...}` |

**核心思路**：池只创建一次；rollout 时每次认领都用任务自身的镜像覆盖，从而让单个
公共 SandboxSet 服务于所有实例。

> 注意：Harbor 驱动的 sandbox 创建目前为实验特性，未来可能变化。
