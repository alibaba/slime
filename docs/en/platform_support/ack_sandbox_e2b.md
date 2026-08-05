# Running slime RL with remote agents on ACK sandboxes (E2B)

This is a verified, end-to-end recipe for training with slime where each rollout runs a
real coding **agent inside an ACK sandbox**, driven over the **E2B protocol**. The agent's
LLM calls are routed through slime's OpenAI adapter to the SGLang engines, token-level data is
captured, and a GRPO step trains on it.

```
 slime (train_remote_agent.py)
   ├─ SGLang engines  ← OpenAI adapter (records token_ids/logprobs) ←─┐
   ├─ Megatron actors (GRPO)                                       │ OpenAI API (base_url = OpenAI adapter)
   └─ generate_with_harbor ─ harbor Trial ─ E2BEnvironment ─┐      │
                                                            ▼      │
                       ACK sandbox-manager (E2B API) ─ claims ─ Sandbox pod (agent runs here)
```

Five things to prepare: **K8s**, **code (image)**, **model**, **data**, then **run**.

---

## 1. Kubernetes preparation

### 1.1 Sandbox stack
Install the ACK sandbox stack (OpenKruise `agents.kruise.io` CRDs + `ack-sandbox-manager` +
`ack-sandbox-gateway`) in namespace `sandbox-system`. This exposes the two E2B endpoints slime uses:

| Plane | Endpoint | Purpose |
|-------|----------|---------|
| Control | `http://sandbox-manager.sandbox-system:8080` | create/claim sandbox, templates |
| Data | `http://sandbox-gateway.sandbox-system:7788` | envd exec/filesystem routing |

Get the E2B admin key from the manager (`--e2b-admin-key`); export it as `E2B_API_KEY` at run time.

### 1.2 Image pull secret (for private task images)
If your task images live in a private registry, create a `kubernetes.io/dockerconfigjson`
secret in the namespace where sandboxes run, e.g. `acr-pro-registry`.

### 1.3 SandboxSet pools (one per task image)

> **Important:** per-claim image override (swapping a warm pod's image at claim time) is **not
> reliable** on the current ACK controller — an in-place image update leaves the sandbox
> `state=dead / InplaceUpdating` and the claim times out. So **bake the task image into the
> SandboxSet** and use `override_claim_image=false`, i.e. **one SandboxSet per task image**, and
> route each sample to its pool via prompt metadata (`sandbox_set_name`).

Each SandboxSet MUST:
- set `spec.runtimes: [{name: agent-runtime}]` — this injects the E2B `envd` daemon (without it,
  `AsyncSandbox.create` fails with `connection refused` on envd port 49983);
- carry the image pull secret if the image is private;
- run privileged (swe-bench images need it).

```yaml
apiVersion: agents.kruise.io/v1alpha1
kind: SandboxSet
metadata:
  name: slime-sbx-pallets-flask-5014   # DNS-safe; referenced from prompt metadata
  namespace: default
spec:
  replicas: 2                          # >= n_samples_per_prompt for this task
  runtimes:
  - name: agent-runtime                # injects envd
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
        image: <registry>/swebench-verified/pallets-flask-5014:<tag>   # the task's image, baked in
        imagePullPolicy: IfNotPresent
        securityContext: { privileged: true, runAsUser: 0 }
        resources:
          requests: { cpu: "1", memory: 4Gi, ephemeral-storage: 10Gi }
```

```bash
kubectl apply -f sandboxset-<instance>.yaml
kubectl get sandboxset -n default          # AVAILABLE should reach REPLICAS
```

### 1.4 Ray cluster
Run slime on a Ray cluster (e.g. a KubeRay `RayCluster`) whose head pod has the GPUs and the
workspace image from step 2. The head pod's IP must be reachable from the sandbox pods (same
cluster network) — it is passed to the agent as `--harbor-adapter-public-host` so the in-sandbox agent
can reach the OpenAI adapter.

---

## 2. Code preparation (build the workspace image)

Build the workspace image from `docker/Dockerfile.workspace`. It bakes in: the slime code
(this branch), Megatron-LM on `sys.path`, and harbor with ACK/E2B support:

```dockerfile
# harbor with ACK sandbox + E2B support (sandbox_set_name / override_claim_image, skips the
# ACK-unsupported template build, per-claim image key). Pulls e2b SDK, litellm, etc.
RUN git clone --depth 1 -b feat/ack-sandbox-image-override \
        https://github.com/alibaba/harbor.git /root/harbor && \
    pip install -e /root/harbor && \
    pip install kubernetes_asyncio
```

```bash
docker build -f docker/Dockerfile.workspace -t <registry>/dev/slime:<tag> .
docker push <registry>/dev/slime:<tag>
# point the RayCluster head at this image and (re)create the head pod
```

This branch also contains the slime fixes required for the newer base image (SGLang 0.5.15 /
recent Megatron) and for correct token capture — no runtime patching is needed once the image
is built:
- `model_provider` / freeze wrapper accept Megatron's `config` / `pg_collection`;
- HF/SGLang arg-validation tolerant of renamed args;
- `torch_memory_saver` preload lib located by glob;
- E2B/HARBOR env vars forwarded into the RolloutManager actor;
- `apply_chat_template` outputs coerced to `list[int]` in the adapter's per-turn token rendering.

---

## 3. Model preparation

Download the HF checkpoint (used for the tokenizer, SGLang, and as the conversion source), then
convert to a Megatron `torch_dist` checkpoint for `--ref-load`:

```bash
# 1. HF checkpoint (e.g. via hf / modelscope)
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# 2. HF -> torch_dist (put the output on a durable volume so it survives pod restarts)
cd /root/slime
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
    --save /var/model/Qwen2.5-0.5B_torch_dist
```

Use a matching `scripts/models/*.sh` preset for other sizes (e.g. `qwen2.5-7B.sh`). Note TP must
divide `num_query_groups` (2 for Qwen2.5), so `--tensor-model-parallel-size` ≤ 2 for these models.

---

## 4. Data preparation

### 4.1 Task directories
slime resolves each sample to a harbor task dir via `--harbor-task-path-template`. swe-bench
tasks (each with `task.toml`, `environment/Dockerfile`, `solution/`, `tests/`) live at e.g.
`/var/model-dataset/swe-bench-verified/{instance_id}`.

### 4.2 Prompts (`prompts.jsonl`)
One JSON object per line. `task_name` selects the task dir; `metadata.sandbox_set_name` routes
the sample to its baked SandboxSet:

```json
{"prompt": "<problem statement>", "task_name": "pallets__flask-5014", "metadata": {"sandbox_set_name": "slime-sbx-pallets-flask-5014"}}
{"prompt": "<problem statement>", "task_name": "astropy__astropy-14309", "metadata": {"sandbox_set_name": "slime-sbx-astropy-14309"}}
```

Run with `--input-key prompt`. slime merges top-level fields (`task_name`, ...) into
`sample.metadata`, and `generate_with_harbor` reads `metadata.sandbox_set_name` first when
resolving the pool.

---

## 5. Run

```bash
export E2B_API_KEY=<ACK sandbox admin key>
bash examples/remote_agent/run_swebench_e2b.sh
```

`examples/remote_agent/run_swebench_e2b.sh` sets the required env and flags. The essential ones:

| Flag / env | Value | Why |
|------------|-------|-----|
| `E2B_API_URL` | `http://sandbox-manager.sandbox-system:8080` | E2B control plane |
| `E2B_SANDBOX_URL` | `http://sandbox-gateway.sandbox-system:7788` | E2B data plane (router) |
| `E2B_VALIDATE_API_KEY` | `false` | skip client-side key format check |
| `--harbor-use-local-trial` | — | run the harbor Trial in-process |
| `--harbor-env-import-path` | `harbor.environments.e2b:E2BEnvironment` | use the E2B env |
| `--harbor-env-kwargs` | `{"override_claim_image": false}` | use the baked pool image (no in-place override) |
| `--harbor-adapter-public-host` | **head pod IP** (not `0.0.0.0`) | so the in-sandbox agent can reach the OpenAI adapter |
| `--harbor-agent-name` | `swe-agent` | built-in SWE-agent |
| `--sglang-disable-cuda-graph` | — | SGLang 0.5.15 prefill CUDA graph is incompatible with slime's memory-saver mode |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | required for tensor parallelism |

---

## 6. What success looks like

In the run log you should see, per sample, real token capture and a GRPO step:

```
[Harbor][...] Reconstructed sample: response_len=976, num_tokens=1026
rollout.py: perf 0: {'rollout/response_len/mean': ~1000, ...}
model.py:  step 0: {'train/loss': ..., 'train/entropy_loss': 2.10,
                    'train/train_rollout_logprob_abs_diff': 0.80, ...}
```

Nonzero `entropy_loss` / `train_rollout_logprob_abs_diff` mean the actor ran a real forward pass
over the captured tokens. With a tiny model that solves no task, all rewards are `0` so the
GRPO advantage (and `grad_norm`) is `0` — use a stronger model, solvable tasks, or a shaped
reward to get a learning signal.

---

## 7. Notes & sizing

- **One pool per image, `override_claim_image=false`.** Per-claim image override is broken on the
  current ACK controller (tracked upstream in `openkruise/agents`); revisit once fixed.
- **Memory.** Colocated Megatron + SGLang on one node is memory-hungry; a full 3B/7B run can OOM a
  200 GiB pod during startup. Start small (0.5B), lower `--sglang-mem-fraction-static`, and keep
  the `--ref-load` checkpoint on a durable volume so a restart doesn't force reconversion.
- **Agent cost limits.** `--harbor-agent-kwargs '{"total_cost_limit":0,...}'` is fine — SWE-agent
  treats `0` as *unlimited*, not "no budget".

---

## 8. FAQ

### `ModuleNotFoundError: No module named 'megatron.training'` (in the Megatron train actors)

Symptom: the run starts, but the `MegatronTrainRayActor` fails to import with
`No module named 'megatron.training'` (or the checkpoint conversion fails the same way).

Cause: `import megatron` resolves to the installed **`megatron-core`** namespace package under
`dist-packages`, which does **not** contain `megatron.training`. `pip install -e /root/Megatron-LM`
does not register Megatron-LM into that namespace, and — crucially — **Ray worker actors do not
inherit the driver process's `PYTHONPATH`**, so exporting `PYTHONPATH=/root/Megatron-LM` before
launching only fixes the driver, not the actors.

Fix (baked into `docker/Dockerfile.workspace`): add `/root/Megatron-LM` to `sys.path` for **every**
Python process via a `.pth` file in site-packages, so `megatron.training` is importable in all Ray
workers without relying on `PYTHONPATH`:

```dockerfile
RUN echo /root/Megatron-LM > "$(python -c 'import site; print(site.getsitepackages()[0])')/zzz_megatron_lm.pth"
```

If you hit this on a running pod (e.g. testing without rebuilding the image), apply the same line
at runtime:

```bash
echo /root/Megatron-LM > "$(python -c 'import site; print(site.getsitepackages()[0])')/zzz_megatron_lm.pth"
python -c "from megatron.training.arguments import parse_args; print('ok')"   # verify
```
