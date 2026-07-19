# Integrating slime with ACK Sandbox over the E2B protocol

This guide explains how to run slime RL rollouts against sandboxes on Alibaba
Cloud Container Service for Kubernetes (ACK), using the **E2B protocol** as the
wire format between slime/Harbor and the cluster.

The design is: **one common SandboxSet (a generic warm pool), and the
instance-specific image is swapped in at claim time during RL training.** This
avoids building a template per image / per task.

There are two API planes:

| Plane | API | When | Who |
|-------|-----|------|-----|
| Pool creation | Kubernetes sandbox CRD | once, before training | cluster admin |
| Sandbox creation (+ image override) | E2B API | per sample, during rollout | slime → Harbor |

```
 slime rollout ── Harbor ── E2B SDK ──► gateway ── k8s ──► claim from common SandboxSet
 (generate_with_harbor)     (E2B env)    (ack-sandbox-manager)   └─ per-claim image = task image
```

---

## Prerequisites

### 1. A sandbox controller on ACK (choose one)

- **`ack-sandbox-controller` + `ack-sandbox-manager` (recommended)** — the ACK-native
  stack; `ack-sandbox-manager` includes the E2B-compatible gateway.
- **OpenKruise** — provides the `SandboxSet` / `SandboxClaim` / `Sandbox` CRDs
  (`agents.kruise.io/v1alpha1`), which Harbor's `ACKEnvironment` targets directly.

Per-claim image override is supported by both (OpenKruise SandboxClaim
`spec.inplaceUpdate.image`; the E2B gateway extension key
`e2b.agents.kruise.io/image`).

### 2. Per-instance images (harbor CLI)

Each task/instance needs its own container image, built and pushed to a registry
the cluster can pull from. **Image preparation is covered in a separate
document** — in brief, the harbor CLI builds from the task's Dockerfile (or a
prebuilt `docker_image`) and pushes to your registry.

---

## Step 1 — Create one common SandboxSet (before training)

Create a single generic warm pool. Size `replicas` to your rollout concurrency.

OpenKruise example:

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
          image: registry.cn-hangzhou.aliyuncs.com/my-repo/sandbox-base:latest  # generic
          command: ["sleep", "infinity"]
          securityContext: { privileged: true, runAsUser: 0 }
          resources:
            requests: { cpu: "2", memory: "4Gi" }
```

```bash
kubectl apply -f sandboxset.yaml
kubectl get sandboxset -n my-namespace
```

(For `ack-sandbox-controller`, create the equivalent template via its CRD — see
its docs.)

---

## Step 2 — Put each instance's image in its `task.toml`

Each task carries its image under `[environment]`:

```toml
[environment]
docker_image = "registry.cn-hangzhou.aliyuncs.com/my-repo/sympy-12096:latest"
cpus = 2
memory_mb = 4096
```

Harbor reads this per task and uses it as the **per-claim image override** against
the common SandboxSet — no per-image template build.

---

## Step 3 — Run training through the Harbor path

slime's Harbor integration (`slime/rollout/remote_agent/`) drives Harbor, which
creates the sandbox and applies the image override.

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

- `--harbor-env-kwargs '{"sandbox_set_name": "slime-common-pool"}'` selects the
  common pool. Harbor's `E2BEnvironment` claims from it and passes the task's
  `docker_image` via the gateway key `e2b.agents.kruise.io/image` (per-claim
  image override). Use `harbor.environments.ack:ACKEnvironment` to talk to the
  OpenKruise CRDs directly instead (image override via `spec.inplaceUpdate.image`).
- Harbor-side controls (see harbor `E2BEnvironment` / `ACKEnvironment`):
  `sandbox_set_name` (common pool), `override_claim_image` (default `true`).

### Optional: per-task pool routing by size class

If you later split the pool by pod size, tag each `task.toml` with a class and
let slime resolve it to a SandboxSet name per task:

```toml
[metadata]
sandbox_class = "large"
```

```bash
  --harbor-sandbox-class-key sandbox_class \
  --harbor-sandbox-set-name-template 'slime-pool-{sandbox_class}' \
  --harbor-sandbox-set-key sandbox_set_name
```

slime's `generate_with_harbor` reads the class (sample metadata or `task.toml`),
converts it via the template, and injects it into `environment_kwargs`. With a
single common pool this is unnecessary — a static `--harbor-env-kwargs` suffices.

---

## Checklist

| Step | Plane | How |
|------|-------|-----|
| 1. Sandbox controller on ACK | k8s | ack-sandbox-manager (gateway) or OpenKruise |
| 2. One common SandboxSet | k8s | `kubectl apply` |
| 3. Per-instance images | — | harbor CLI (separate doc) |
| 4. `docker_image` in each `task.toml` | dataset | `[environment] docker_image` |
| 5. Run training | slime | `generate_with_harbor` + `--harbor-env-kwargs {"sandbox_set_name": ...}` |

**Mental model:** create the pool once; at rollout time each claim overrides the
image with the task's own image, so a single common SandboxSet serves every
instance.

> Note: Harbor-driven sandbox creation is experimental and may change.
