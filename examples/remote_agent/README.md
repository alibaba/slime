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
