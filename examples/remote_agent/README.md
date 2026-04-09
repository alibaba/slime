# Slime RemoteAgent with Harbor

This example demonstrates how to run Slime training with agents executed
on a remote [Harbor](https://github.com/agent-arena/harbor) server, or
locally via in-process Trial execution.  In both modes the TokenProxy
captures token-level data (token_ids, logprobs) for RL training.

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
│  │  TokenProxy          │   │   OpenAI SDK            │
│  │  (Ray Named Actor)   │◀──┼───(base_url)────────────┘
│  │  FastAPI + LiteLLM   │   │
│  │  → SGLang Ray RPC    │   │
│  │  → SessionRecorder   │   │
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
│  │  TokenProxy          │   │
│  │  (Ray Named Actor)   │   │
│  │  FastAPI + LiteLLM   │   │
│  │  → SGLang Ray RPC    │   │
│  │  → SessionRecorder   │   │
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

### Remote Mode

```bash
export LOCAL_IP=10.0.30.11   # IP reachable from Harbor containers
bash examples/remote_agent/harbor_qwen.sh
```

### Local Trial Mode

No `LOCAL_IP` needed since the agent runs in the same process:

```bash
bash examples/remote_agent/harbor_local_trial.sh
```

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
| `--harbor-proxy-host` | `0.0.0.0` | Bind host for the LLM proxy (remote mode only) |
| `--harbor-proxy-port` | `0` | Proxy port, 0 = auto-select (remote mode only) |
| `--harbor-max-retries` | `3` | Max retry attempts on failure (remote mode only) |
| `--harbor-retry-base-delay` | `2.0` | Base delay (seconds) for exponential backoff (remote mode only) |
| `--harbor-disable-reconstruct` | `False` | Disable token reconstruction from proxy |
| `--harbor-use-local-trial` | `False` | **Run Trial locally instead of remote Harbor** |

### Remote Mode Only Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--harbor-server-url` | `http://localhost:8080` | Harbor Agent Run server URL |
| `LOCAL_IP` env var | `0.0.0.0` | IP the Harbor containers can reach |

## How It Works

1. **Proxy startup**: `train_remote_agent.py` starts a `TokenProxy` as a Ray Named Actor after
   creating the rollout manager. The proxy holds SGLang engine handles and runs
   a FastAPI + LiteLLM server.

2. **Generate function**: `generate_with_harbor` replaces the default generate
   function. For each sample:
   - Creates a `trial_id` and registers a session with the proxy
   - Builds the agent's `base_url` pointing to the proxy
   - **Remote mode**: Submits the task to Harbor HTTP server and waits for completion (with retry)
   - **Local mode**: Runs `harbor.trial.trial.Trial` directly in the current process
   - Reconstructs `Sample.tokens`, `Sample.rollout_log_probs`, and
     `Sample.loss_mask` from the proxy's session recording

3. **Token reconstruction**: LLM-generated tokens get `mask=1` (participate in
   loss), while tool/user replies get `mask=0` (masked out). This is the key
   to RL training with multi-turn agents.

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
