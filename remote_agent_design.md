# Slime RemoteAgent 设计方案 (Harbor 对接 / custom-generate-function)

> **⚠️ 已部分过时（历史设计文档）**
>
> 本文档描述的是早期基于独立 `TokenProxy`（head-node Ray 命名 actor + FastAPI/LiteLLM
> + REST `/sessions`、`/engines` 自注册 + 事后 `_reconstruct_output`）的实现。该实现已被
> **进程内 `OpenAIAdapter` + `TrajectoryManager`** 方案取代：adapter 运行在
> `RolloutManager` actor 内，token 在生成时即被捕获，`finish_session` 直接产出训练
> `Sample`，不再有跨进程 REST 往返、引擎自注册或 token 重建。sglang 访问复用 slime 自带的
> sglang router（`--sglang-router-ip/-port`）。
>
> 最新实现见 `slime/rollout/remote_agent/adapter_service.py` 与 `generate.py`，以及
> `examples/remote_agent/README.md`。下文的 `TokenProxy`/`--harbor-proxy-*`/
> `--harbor-disable-reconstruct` 等已不存在。

## 一、核心思路

利用 slime 已有的 `--custom-generate-function-path` 扩展点，在 **单个 sample 的 generate 函数** 内完成 Harbor 任务提交和 token 重建，无需修改 slime 核心代码，也无需引入新的 Loop 类型。

### 1.1 与 Verl 实现的关系

Verl 的 `patch.diff` 实现了完整的 `RemoteAgentLoop`，它继承 `AgentLoopBase` 并注册为 `"remote_agent"`，在 AgentLoop 级别工作。Slime 的架构不同——slime 的 rollout 流程是 `RolloutManager → generate_rollout → generate(args, sample, sampling_params)`。我们只需要替换最内层的 `generate` 函数即可。

### 1.2 数据流

```
train.py → RolloutManager.generate()
  → sglang_rollout.generate_rollout_async()       # 现有 rollout 循环不变
    → generate(args, sample, sampling_params)      # ← 被替换的函数
      ├── 1. 确保 TokenProxy 运行 (Ray Actor / 独立进程)
      ├── 2. 生成 trial_id，注册 session
      ├── 3. 提交任务到 Harbor (HTTP POST /api/v1/runs)
      ├── 4. 等待 Harbor 完成 (长轮询)
      ├── 5. 从 Proxy 获取 session 记录
      ├── 6. _reconstruct_output → 重建 tokens / logprobs
      └── 7. 填充 Sample 返回
    → batched_async_rm()                           # 现有 RM 流程不变
```

---

## 二、架构设计

### 2.1 整体架构

```
Slime Ray Cluster                                Harbor Server (远程)
┌───────────────────────────────┐                ┌──────────────────────┐
│  train.py                     │                │                      │
│    └── RolloutManager         │   HTTP POST    │  /api/v1/runs        │
│         └── generate_rollout  │───────────────▶│    ├── 打包 task     │
│              └── generate()   │                │    ├── 启动 Docker   │
│                   │           │                │    └── 运行 Agent    │
│                   │           │                └──────────┬───────────┘
│                   ▼           │                           │
│  ┌──────────────────────┐    │                           │
│  │  TokenProxy          │    │                           │
│  │  (Ray Named Actor)   │◀───┼────── OpenAI SDK ─────────┘
│  │                      │    │   (base_url = proxy/{trial_id}/v1)
│  │  FastAPI + LiteLLM   │    │
│  │  → VLLM/SGLang RPC   │    │
│  │  → SessionRecorder   │    │
│  └──────────────────────┘    │
│                               │
│  SGLang Engines (可选本地)    │
└───────────────────────────────┘
```

### 2.2 组件关系

| 组件 | 位置 | 职责 |
|------|------|------|
| `generate_with_harbor` | `slime/rollout/remote_agent/generate.py` | custom-generate-function 入口，处理单个 Sample |
| `TokenProxy` | `slime/rollout/remote_agent/proxy.py` | FastAPI + LiteLLM 代理，捕获 token 数据 |
| `ProxyActor` | `slime/rollout/remote_agent/proxy.py` | Ray Named Actor 包装，保证单例 |
| `HarborClient` | `slime/rollout/remote_agent/harbor_client.py` | 与 Harbor HTTP API 交互 |
| `_reconstruct_output` | `slime/rollout/remote_agent/generate.py` | 从 session 记录重建 Sample 字段 |

---

## 三、代码设计

### 3.1 文件结构

```
slime/
├── rollout/
│   └── remote_agent/
│       ├── __init__.py
│       ├── generate.py              # custom-generate-function 实现
│       ├── proxy.py                 # TokenProxy + ProxyActor
│       └── harbor_client.py         # Harbor SDK 客户端
├── utils/
│   └── arguments.py                 # 新增 --harbor-* 参数
```

### 3.2 参数设计

在 `slime/utils/arguments.py` 的 `add_rollout_arguments` 中新增：

```python
def add_harbor_arguments(parser):
    # Harbor 服务器
    parser.add_argument(
        "--harbor-server-url",
        type=str,
        default="http://localhost:8080",
        help="Harbor Agent Run server URL.",
    )
    parser.add_argument(
        "--harbor-timeout",
        type=float,
        default=1800.0,
        help="Timeout in seconds for Harbor task execution.",
    )

    # Agent 配置
    parser.add_argument(
        "--harbor-agent-name",
        type=str,
        default=None,
        help="Name of a built-in Harbor agent (e.g. 'swe-agent').",
    )
    parser.add_argument(
        "--harbor-agent-import-path",
        type=str,
        default=None,
        help="Python import path for custom agent (e.g. 'my_agents:MyAgent').",
    )
    parser.add_argument(
        "--harbor-model-name",
        type=str,
        default=None,
        help="LLM model name passed to the remote agent.",
    )
    parser.add_argument(
        "--harbor-agent-kwargs",
        type=json.loads,
        default="{}",
        help="JSON-encoded dict of extra kwargs for the agent.",
    )

    # 环境配置
    parser.add_argument(
        "--harbor-env-overrides",
        type=json.loads,
        default="{}",
        help="JSON-encoded dict of env vars forwarded to the remote agent.",
    )
    parser.add_argument(
        "--harbor-env-import-path",
        type=str,
        default="harbor.environments.local_docker:LocalDockerEnvironment",
        help="Python import path for the environment class.",
    )
    parser.add_argument(
        "--harbor-env-kwargs",
        type=json.loads,
        default="{}",
        help="JSON-encoded dict of kwargs for the environment constructor.",
    )

    # 任务配置
    parser.add_argument(
        "--harbor-task-path-template",
        type=str,
        default="/home/slime/dataset-tasks/{instance_id}",
        help="Task path template with {instance_id} placeholder.",
    )
    parser.add_argument(
        "--harbor-use-local-trial",
        action="store_true",
        default=False,
        help="Run Trial locally via harbor.trial.trial.Trial instead of Harbor HTTP.",
    )

    # Proxy 配置
    parser.add_argument(
        "--harbor-proxy-host",
        type=str,
        default="0.0.0.0",
        help="Bind host for the LLM proxy server.",
    )
    parser.add_argument(
        "--harbor-proxy-port",
        type=int,
        default=0,
        help="Port for the LLM proxy server. 0 = auto-select.",
    )

    # 重试配置
    parser.add_argument(
        "--harbor-max-retries",
        type=int,
        default=3,
        help="Max retry attempts on Harbor task failure.",
    )
    parser.add_argument(
        "--harbor-retry-base-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds for exponential backoff.",
    )

    # 输出重建
    parser.add_argument(
        "--harbor-disable-reconstruct",
        action="store_true",
        default=False,
        help="Disable token reconstruction from proxy session data.",
    )

    return parser
```

### 3.3 TokenProxy (`proxy.py`)

参考 Verl 的 `LLMProxyServer`，但适配 SGLang：

```python
# slime/rollout/remote_agent/proxy.py
"""LLM Proxy Server — 通过 LiteLLM + SGLang Ray RPC 捕获 token 数据。

架构::

    Harbor 中的 Agent
         |  (OpenAI /v1/chat/completions)
         v
    +----------------------------------+
    |  TokenProxy  (Ray Named Actor)   |
    |  - FastAPI HTTP server           |
    |  - LiteLLM custom provider       |
    |  - SGLangRayProvider             |
    |  - SessionRecorder per trial_id  |
    +---------------+------------------+
                    |  server.generate.remote(prompt_ids, ...)
                    v
             SGLang rollout servers
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import litellm
import ray
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------- 数据模型 ----------

@dataclass
class CompletionRecord:
    """Proxy 捕获的单次 LLM 调用记录"""
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    completion_text: str = ""
    completion_token_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class SessionRecord:
    """单个 trial 的完整 session 记录"""
    session_id: str
    turns: list[CompletionRecord] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: __import__("time").time())
    completed: bool = False


class SessionRecorder:
    """线程安全的 session 记录管理器"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = SessionRecord(session_id=session_id)

    def record_completion(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        completion_text: str,
        token_ids: list[int],
        logprobs: list[float],
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        record = CompletionRecord(
            request_messages=messages,
            completion_text=completion_text,
            completion_token_ids=token_ids,
            completion_logprobs=logprobs,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionRecord(session_id=session_id)
            self._sessions[session_id].turns.append(record)

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._sessions.get(session_id)

    def mark_completed(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].completed = True

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# ---------- SGLang Ray Provider (LiteLLM CustomLLM) ----------

class SGLangRayProvider:
    """LiteLLM custom provider，通过 Ray RPC 调用 SGLang。

    替代 Verl 的 VLLMRayProvider，适配 slime 的 SGLangEngine 接口。
    """

    def __init__(self, engine_handles: list, tokenizer: AutoTokenizer):
        self.engine_handles = list(engine_handles)
        self.tokenizer = tokenizer
        self._sticky: dict[str, int] = {}
        self._request_counts: list[int] = [0] * len(engine_handles)

    def _choose_server(self, session_id: str | None):
        """sticky-session 负载均衡"""
        key = session_id or "default"
        if key in self._sticky:
            return self.engine_handles[self._sticky[key]]
        idx = min(range(len(self._request_counts)), key=lambda i: self._request_counts[i])
        self._sticky[key] = idx
        self._request_counts[idx] += 1
        return self.engine_handles[idx]

    async def acompletion(self, model, messages, api_base, custom_prompt_dict,
                          model_response, print_verbose, encoding, api_key,
                          logging_obj, optional_params, **kwargs):
        """LiteLLM acompletion 接口"""
        session_id = (optional_params.get("extra_body") or {}).get("session_id")
        temperature = optional_params.get("temperature", 1.0)
        top_p = optional_params.get("top_p", 1.0)
        max_tokens = optional_params.get("max_tokens", 2048)
        stop = optional_params.get("stop")

        # Tokenize
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        # 通过 Ray RPC 调用 SGLang
        server = self._choose_server(session_id)
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        if stop:
            sampling_params["stop"] = stop

        # SGLangEngine.generate 的签名: generate(prompt_ids, sampling_params, request_id)
        ref = server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=uuid4().hex,
        )
        output = await asyncio.to_thread(ray.get, ref)

        token_ids = list(output.token_ids)
        log_probs = list(output.log_probs) if output.log_probs else []
        stop_reason = getattr(output, "stop_reason", None)
        completion_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        # 构建 LiteLLM ModelResponse
        from litellm.types.utils import ChatCompletionTokenLogprob, ChoiceLogprobs, Choices, Message, Usage

        logprobs_obj = None
        if log_probs:
            content = []
            for tid, lp in zip(token_ids, log_probs):
                tok_str = self.tokenizer.decode([tid])
                content.append(ChatCompletionTokenLogprob(
                    token=tok_str, logprob=lp,
                    bytes=list(tok_str.encode("utf-8", errors="replace")),
                    top_logprobs=[],
                ))
            logprobs_obj = ChoiceLogprobs(content=content)

        model_response.choices = [Choices(
            finish_reason=stop_reason or "stop",
            index=0,
            message=Message(role="assistant", content=completion_text),
            logprobs=logprobs_obj,
            provider_specific_fields={"token_ids": token_ids},
        )]
        model_response.model = model
        model_response.usage = Usage(
            prompt_tokens=0, completion_tokens=len(token_ids), total_tokens=len(token_ids)
        )
        return model_response


# ---------- FastAPI 应用 ----------

def _build_app(proxy: "TokenProxy") -> FastAPI:
    app = FastAPI(title="Slime LLM Proxy", version="0.1.0")

    # Session 管理
    @app.post("/sessions/{session_id}")
    async def create_session(session_id: str):
        proxy.recorder.create_session(session_id)
        return {"session_id": session_id, "status": "created"}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = proxy.recorder.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "turns": [
                {
                    "request_messages": t.request_messages,
                    "completion_text": t.completion_text,
                    "completion_token_ids": t.completion_token_ids,
                    "completion_logprobs": t.completion_logprobs,
                    "finish_reason": t.finish_reason,
                    "tool_calls": t.tool_calls,
                }
                for t in session.turns
            ],
            "completed": session.completed,
        }

    @app.post("/sessions/{session_id}/complete")
    async def complete_session(session_id: str):
        proxy.recorder.mark_completed(session_id)
        return {"session_id": session_id, "status": "completed"}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        proxy.recorder.delete_session(session_id)
        return {"session_id": session_id, "status": "deleted"}

    # OpenAI 兼容 chat completions
    @app.post("/{trial_id}/v1/chat/completions")
    @app.post("/v1/chat/completions/{trial_id}")
    async def chat_completions(trial_id: str, request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        is_streaming = body.get("stream", False)

        completion_kwargs = {
            "model": "slime-sglang/default",
            "messages": messages,
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "max_tokens": body.get("max_tokens", 2048),
            "stream": is_streaming,
            "extra_body": {"session_id": trial_id},
        }
        if body.get("tools"):
            completion_kwargs["tools"] = body["tools"]
        if "stop" in body:
            completion_kwargs["stop"] = body["stop"]

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as e:
            logger.error("Generate failed for trial %s: %s", trial_id, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

        # 记录到 session
        _record_from_response(proxy, trial_id, messages, response)

        if is_streaming:
            return StreamingResponse(
                _stream_and_record(proxy, trial_id, messages, response),
                media_type="text/event-stream",
            )
        return JSONResponse(content=response.model_dump())

    return app


def _record_from_response(proxy: "TokenProxy", trial_id: str,
                          messages: list[dict], response: Any) -> None:
    """从 LiteLLM ModelResponse 提取 token_ids/logprobs 并记录"""
    choice = response.choices[0]
    psf = getattr(choice, "provider_specific_fields", None) or {}
    token_ids = psf.get("token_ids", [])

    logprobs_list = []
    choice_logprobs = getattr(choice, "logprobs", None)
    if choice_logprobs and hasattr(choice_logprobs, "content") and choice_logprobs.content:
        logprobs_list = [t.logprob for t in choice_logprobs.content]

    tool_calls = None
    msg = choice.message
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in msg.tool_calls]

    content = getattr(msg, "content", "") or ""

    proxy.recorder.record_completion(
        session_id=trial_id,
        messages=messages,
        completion_text=content,
        token_ids=token_ids,
        logprobs=logprobs_list,
        finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
        tool_calls=tool_calls,
    )


async def _stream_and_record(proxy, trial_id, messages, stream):
    """流式响应并记录"""
    collected_text_parts = []
    collected_token_ids = []
    collected_logprobs = []
    finish_reason = "stop"

    try:
        async for chunk in stream:
            chunk_data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            yield f"data: {json.dumps(chunk_data)}\n\n"

            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0]
                if hasattr(delta, "delta") and hasattr(delta.delta, "content"):
                    if delta.delta.content:
                        collected_text_parts.append(delta.delta.content)
                psf = getattr(delta, "provider_specific_fields", None) or {}
                if "token_ids" in psf:
                    collected_token_ids = psf["token_ids"]
                if hasattr(delta, "finish_reason") and delta.finish_reason:
                    finish_reason = delta.finish_reason
                chunk_logprobs = getattr(delta, "logprobs", None)
                if chunk_logprobs and hasattr(chunk_logprobs, "content") and chunk_logprobs.content:
                    for t in chunk_logprobs.content:
                        collected_logprobs.append(t.logprob)

        yield "data: [DONE]\n\n"
    finally:
        proxy.recorder.record_completion(
            session_id=trial_id,
            messages=messages,
            completion_text="".join(collected_text_parts),
            token_ids=collected_token_ids,
            logprobs=collected_logprobs,
            finish_reason=finish_reason,
        )


# ---------- TokenProxy 主类 ----------

class TokenProxy:
    """LLM Proxy 服务器，单例运行，桥接 OpenAI HTTP 到 SGLang Ray RPC"""

    def __init__(
        self,
        engine_handles: list,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 0,
    ):
        self.host = host
        self.port = port
        self.engine_handles = engine_handles
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.recorder = SessionRecorder()

        # 初始化 LiteLLM custom provider
        self._provider = SGLangRayProvider(engine_handles, self.tokenizer)
        litellm.custom_provider_map = [
            {"provider": "slime-sglang", "custom_handler": self._provider}
        ]

        self.app = _build_app(self)
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None
        self._actual_port: int | None = None

    @property
    def url(self) -> str | None:
        if self._actual_port is None:
            return None
        return f"http://{self.host}:{self._actual_port}"

    async def start(self) -> str:
        """启动 HTTP 服务器，返回 base URL"""
        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                self._actual_port = s.getsockname()[1]
        else:
            self._actual_port = self.port

        config = uvicorn.Config(app=self.app, host=self.host, port=self._actual_port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None
        self._serve_task = asyncio.create_task(self._server.serve())

        for _ in range(100):
            if self._server.started:
                break
            await asyncio.sleep(0.1)

        logger.info("LLM Proxy started at %s", self.url)
        return self.url

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
            if self._serve_task:
                try:
                    await self._serve_task
                except asyncio.CancelledError:
                    pass
                self._serve_task = None
            self._server = None


# ---------- Ray Named Actor包装 ----------

PROXY_ACTOR_NAME = "slime_llm_proxy"
_proxy_actor_handle = None


@ray.remote
class ProxyActor:
    """Ray Named Actor，固定在 head node，管理 TokenProxy 生命周期"""

    def __init__(self, engine_handles, model_path, host, port):
        self.proxy = TokenProxy(engine_handles, model_path, host, port)
        self._url = None

    async def start(self) -> str:
        self._url = await self.proxy.start()
        return self._url

    def get_url(self) -> str | None:
        return self._url

    async def stop(self):
        await self.proxy.stop()


def start_proxy_server(engine_handles, model_path, host="0.0.0.0", port=0) -> str:
    """在 head node 启动 Proxy Ray Named Actor"""
    global _proxy_actor_handle

    # 复用已有
    try:
        existing = ray.get_actor(PROXY_ACTOR_NAME)
        url = ray.get(existing.get_url.remote())
        if url:
            logger.info("Reusing existing proxy actor at %s", url)
            return url
    except ValueError:
        pass

    head_node_id = ray.get_runtime_context().get_node_id()
    actor = ProxyActor.options(
        name=PROXY_ACTOR_NAME,
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=head_node_id, soft=False,
        ),
    ).remote(engine_handles, model_path, host, port)

    _proxy_actor_handle = actor
    url = ray.get(actor.start.remote())
    logger.info("Proxy server actor created at %s", url)
    return url


def get_proxy_url() -> str | None:
    """获取已运行 Proxy 的 URL"""
    try:
        actor = ray.get_actor(PROXY_ACTOR_NAME)
        return ray.get(actor.get_url.remote())
    except ValueError:
        return None
```

### 3.4 Harbor Client (`harbor_client.py`)

参考 Verl 的 `AgentRunClient`，适配 slime 的 async 场景：

```python
# slime/rollout/remote_agent/harbor_client.py
"""Harbor SDK 客户端 — 提交 Agent 任务到远程 Harbor 服务器。

Harbor 服务器 API:
    POST /api/v1/runs    提交任务 (multipart/form-data)
    GET  /api/v1/runs/{run_id}  查询状态
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class HarborAgentConfig:
    name: str | None = None
    import_path: str | None = None
    model_name: str | None = None
    llm_proxy_url: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarborVerifierConfig:
    disable: bool = False


@dataclass
class HarborRunResult:
    run_id: str
    status: str  # completed | failed | timeout | error
    rewards: dict[str, float] | None = None
    error_message: str | None = None
    result_uri: str | None = None


def _create_task_archive(task_path: str) -> tuple[bytes, str]:
    """将本地 task 目录打包为 tar.gz"""
    task_dir = Path(task_path).resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(task_dir), arcname=task_dir.name)
    buf.seek(0)
    return buf.getvalue(), task_dir.name


class HarborClient:
    """与 Harbor Agent Run 服务器通信的客户端"""

    def __init__(self, server_url: str, timeout: float = 1800.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    async def submit_async(
        self,
        task_path: str,
        agent: HarborAgentConfig,
        verifier: HarborVerifierConfig | None = None,
        environment_overrides: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        timeout_multiplier: float = 1.0,
    ) -> HarborRunResult:
        """提交任务到 Harbor 并等待完成"""
        verifier = verifier or HarborVerifierConfig()

        payload = {
            "task_path": task_path,
            "agent": {k: v for k, v in {
                "name": agent.name,
                "import_path": agent.import_path,
                "model_name": agent.model_name,
                "llm_proxy_url": agent.llm_proxy_url,
                "kwargs": agent.kwargs,
            }.items() if v is not None and v != {} and v != []},
            "timeout_multiplier": timeout_multiplier,
            "verifier": {"disable": verifier.disable},
        }
        if environment_overrides:
            payload["environment_overrides"] = environment_overrides
        if environment_kwargs:
            payload["environment_kwargs"] = environment_kwargs

        # 打包 task 目录
        try:
            archive_bytes, _dir_name = await asyncio.to_thread(_create_task_archive, task_path)
        except FileNotFoundError as e:
            return HarborRunResult(
                run_id="", status="error", rewards={"reward": 0.0},
                error_message=str(e),
            )

        url = f"{self.server_url}/api/v1/runs"
        logger.info("Submitting Harbor run to %s task=%s", url, task_path)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    data={"metadata": json.dumps(payload)},
                    files={"task_archive": ("task.tar.gz", archive_bytes, "application/gzip")},
                )
        except Exception as e:
            return HarborRunResult(
                run_id="", status="error", rewards={"reward": 0.0},
                error_message=str(e),
            )

        if response.status_code != 200:
            return HarborRunResult(
                run_id="", status="error", rewards={"reward": 0.0},
                error_message=f"Server returned HTTP {response.status_code}: {response.text}",
            )

        data = response.json()
        return HarborRunResult(
            run_id=data.get("run_id", ""),
            status=data.get("status", "error"),
            rewards=data.get("rewards"),
            error_message=data.get("error"),
            result_uri=data.get("result_uri"),
        )
```

### 3.5 Generate 函数 (`generate.py`)

这是核心——`custom-generate-function` 的实现：

```python
# slime/rollout/remote_agent/generate.py
"""custom-generate-function 实现：通过 Harbor 执行 Remote Agent 并捕获 token 数据。

使用方式::

    python train.py \\
        --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \\
        --harbor-server-url http://harbor:8080 \\
        --harbor-agent-name swe-agent \\
        --harbor-model-name openai/qwen-max \\
        --hf-checkpoint /path/to/checkpoint \\
        ...

此函数替代默认的 ``generate(args, sample, sampling_params)``，在每个 sample
级别完成 Harbor 任务提交和 token 重建。
"""

from __future__ import annotations

import asyncio
import logging
import os
import shortuuid
from argparse import Namespace
from typing import Any

from slime.rollout.remote_agent.harbor_client import (
    HarborAgentConfig,
    HarborClient,
    HarborRunResult,
    HarborVerifierConfig,
)
from slime.rollout.remote_agent.proxy import get_proxy_url
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class _HarborGenerateState(metaclass=SingletonMeta):
    """generate 函数的全局状态（进程级别单例）"""
    harbor_client: HarborClient | None = None
    proxy_url: str | None = None
    tokenizer = None

    def ensure_client(self, args: Namespace) -> HarborClient:
        if self.harbor_client is None:
            self.harbor_client = HarborClient(
                server_url=args.harbor_server_url,
                timeout=args.harbor_timeout,
            )
        return self.harbor_client

    async def ensure_proxy_started(self, args: Namespace) -> str:
        """确保 Proxy 已启动并返回 URL"""
        if self.proxy_url is None:
            from slime.rollout.remote_agent.proxy import start_proxy_server

            # 需要 SGLang engine handles —— 从 rollout_manager 获取
            # 在 generate 函数中无法直接访问 rollout_manager，
            # 所以这里用 Ray named actor 发现模式
            self.proxy_url = get_proxy_url()
            if self.proxy_url is None:
                raise RuntimeError(
                    "Proxy server actor not found. "
                    "Make sure the training script starts the proxy before rollout. "
                    "See examples/remote_agent/ for a reference setup."
                )
        return self.proxy_url


def generate_with_harbor(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """custom-generate-function 入口。

    此函数会被 ``sglang_rollout.generate_and_rm`` 调用，替代默认的 SGLang HTTP 请求。

    Args:
        args: 全局配置
        sample: 当前 sample，包含 prompt
        sampling_params: 采样参数 (temperature, top_p, max_new_tokens)
        evaluation: 是否为评估模式

    Returns:
        填充了 tokens、response、logprobs 的 Sample
    """
    state = _HarborGenerateState()
    return asyncio.get_event_loop().run_until_complete(
        _generate_with_harbor_async(args, sample, sampling_params, evaluation, state)
    )


async def _generate_with_harbor_async(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool,
    state: _HarborGenerateState,
) -> Sample:
    """异步实现"""
    client = state.ensure_client(args)
    proxy_url = await state.ensure_proxy_started(args)

    # 1. 生成 trial_id
    instance_id = sample.metadata.get("instance_id", str(sample.index))
    trial_id = f"{instance_id}-{shortuuid.uuid()}"

    # 2. 注册 session
    import httpx
    async with httpx.AsyncClient() as http:
        await http.post(f"{proxy_url}/sessions/{trial_id}")

    try:
        # 3. 构造 Agent 的 LLM proxy URL
        #    LOCAL_IP 是外部可达的 IP（Harbor 服务器能访问到）
        local_ip = os.getenv("LOCAL_IP", "0.0.0.0")
        if local_ip == "0.0.0.0":
            logger.warning(
                "LOCAL_IP is not set, falling back to 0.0.0.0. "
                "The Harbor agent may not be able to reach the proxy."
            )

        # 解析 proxy port
        from urllib.parse import urlparse
        parsed = urlparse(proxy_url)
        proxy_port = parsed.port
        agent_base_url = f"http://{local_ip}:{proxy_port}/{trial_id}/v1"

        # 4. 构造 task 路径
        task_path = args.harbor_task_path_template.format(instance_id=instance_id)

        # 5. 构建 agent 配置
        agent_kwargs = dict(args.harbor_agent_kwargs)
        agent_kwargs["model_base_url"] = agent_base_url
        agent_kwargs["session_id"] = trial_id
        agent_kwargs["temperature"] = sampling_params.get("temperature", 1.0)
        agent_kwargs["top_p"] = sampling_params.get("top_p", 1.0)

        agent_config = HarborAgentConfig(
            name=args.harbor_agent_name,
            import_path=args.harbor_agent_import_path,
            model_name=args.harbor_model_name,
            llm_proxy_url=agent_base_url,
            kwargs=agent_kwargs,
        )

        env_overrides = dict(args.harbor_env_overrides)
        env_overrides.setdefault("OPENAI_API_KEY", "slime-proxy")
        env_overrides.setdefault("OPENAI_BASE_URL", agent_base_url)

        # 6. 提交到 Harbor（带重试）
        result = await _submit_with_retry(
            args, client, trial_id, task_path, agent_config, env_overrides, proxy_url,
        )

        # 7. 标记 session 完成
        async with httpx.AsyncClient() as http:
            await http.post(f"{proxy_url}/sessions/{trial_id}/complete")

        # 8. 获取 session 记录
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{proxy_url}/sessions/{trial_id}")
            session_data = resp.json() if resp.status_code == 200 else None

        # 9. 重建 Sample
        if session_data and session_data.get("turns") and not args.harbor_disable_reconstruct:
            _reconstruct_output(sample, session_data, args)
        else:
            # Fallback: 无 token 记录
            logger.warning("Session %s has no recorded turns — token reconstruction skipped.", trial_id)
            sample.response = ""
            sample.response_length = 0
            sample.status = Sample.Status.FAILED

        # 10. 填充奖励（来自 Harbor）
        if result.rewards:
            sample.reward = result.rewards.get("reward", 0.0)

        sample.metadata["trial_id"] = trial_id
        sample.metadata["harbor_run_id"] = result.run_id
        sample.metadata["harbor_status"] = result.status

    finally:
        # 11. 清理 session
        async with httpx.AsyncClient() as http:
            try:
                await http.delete(f"{proxy_url}/sessions/{trial_id}")
            except Exception:
                pass

    return sample


async def _submit_with_retry(
    args, client, trial_id, task_path, agent_config, env_overrides, proxy_url,
) -> HarborRunResult:
    """带指数退避重试的 Harbor 提交"""
    last_error = None
    for attempt in range(args.harbor_max_retries):
        if attempt > 0:
            delay = args.harbor_retry_base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Retrying trial %s (attempt %d/%d) after %.1fs delay",
                trial_id, attempt + 1, args.harbor_max_retries, delay,
            )
            await asyncio.sleep(delay)

            # 重置 session
            import httpx
            async with httpx.AsyncClient() as http:
                await http.delete(f"{proxy_url}/sessions/{trial_id}")
                await http.post(f"{proxy_url}/sessions/{trial_id}")

        try:
            result = await client.submit_async(
                task_path=task_path,
                agent=agent_config,
                verifier=HarborVerifierConfig(),
                environment_overrides=env_overrides,
                environment_kwargs=dict(args.harbor_env_kwargs),
            )

            if result.status == "completed":
                return result

            last_error = RuntimeError(f"Harbor run {result.status}: {result.error_message or 'unknown'}")
            logger.warning(
                "Trial %s attempt %d/%d %s: %s",
                trial_id, attempt + 1, args.harbor_max_retries,
                result.status, result.error_message,
            )
        except Exception as e:
            last_error = e
            logger.warning(
                "Trial %s attempt %d/%d raised %s: %s",
                trial_id, attempt + 1, args.harbor_max_retries,
                type(e).__name__, e,
            )

    return HarborRunResult(
        run_id="", status="error", rewards={"reward": 0.0},
        error_message=f"All {args.harbor_max_retries} retries exhausted: {last_error}",
    )


def _reconstruct_output(sample: Sample, session_data: dict, args: Namespace) -> None:
    """从 proxy session 记录重建 Sample 的 tokens / logprobs / mask。

    参考 Verl 的 ``RemoteAgentLoop._reconstruct_output``，
    将多轮对话的 LLM 生成 (mask=1) 和 tool/user 回复 (mask=0)
    拼接为完整的 token 序列。
    """
    turns = session_data.get("turns", [])
    if not turns:
        sample.status = Sample.Status.FAILED
        return

    response_ids = []
    response_mask = []
    response_logprobs = []
    num_turns = 0

    # 需要 tokenizer —— 从 state 获取或延迟初始化
    state = _HarborGenerateState()
    if state.tokenizer is None:
        from transformers import AutoTokenizer
        state.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    tokenizer = state.tokenizer

    for i, turn in enumerate(turns):
        # LLM 生成 tokens → mask=1
        completion_ids = turn.get("completion_token_ids", [])
        completion_logprobs = turn.get("completion_logprobs", [])

        response_ids.extend(completion_ids)
        response_mask.extend([1] * len(completion_ids))
        response_logprobs.extend(completion_logprobs)
        num_turns += 1

        # 多轮对话：下一轮的 request_messages 中新增的部分是 tool/user 回复 → mask=0
        if i + 1 < len(turns):
            next_turn = turns[i + 1]
            prev_count = len(turn.get("request_messages", []))
            next_count = len(next_turn.get("request_messages", []))

            if next_count > prev_count:
                new_messages = next_turn["request_messages"][prev_count:]
                tool_messages = [
                    m for m in new_messages
                    if m.get("role") in ("tool", "user", "system")
                ]
                if tool_messages:
                    tool_ids = _tokenize_messages(tool_messages, tokenizer)
                    response_ids.extend(tool_ids)
                    response_mask.extend([0] * len(tool_ids))
                    response_logprobs.extend([0.0] * len(tool_ids))
                    num_turns += len(tool_messages)

    # 截断到最大响应长度
    max_len = args.rollout_max_response_len or 4096
    response_ids = response_ids[:max_len]
    response_mask = response_mask[:max_len]
    response_logprobs = response_logprobs[:max_len]

    # 填充 Sample
    sample.tokens = response_ids  # 只包含 response tokens
    sample.response_length = len(response_ids)
    sample.rollout_log_probs = response_logprobs if response_logprobs else None
    sample.response = "".join(
        turn.get("completion_text", "") for turn in turns
    )
    sample.status = Sample.Status.COMPLETED
    sample.loss_mask = response_mask  # 用于训练时 mask 掉非 LLM 生成部分
    sample.metadata["num_agent_turns"] = num_turns


def _tokenize_messages(messages: list[dict], tokenizer) -> list[int]:
    """将 messages  tokenize 为 token IDs"""
    # 处理 multi-part content (OpenAI vision format)
    normalized = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            msg["content"] = "\n".join(text_parts) if text_parts else ""
        normalized.append(msg)

    return tokenizer.apply_chat_template(normalized, add_generation_prompt=False, tokenize=True)
```

---

## 四、使用方式

### 4.1 启动 Proxy

在 `train.py` 中、rollout 开始前启动 Proxy（类似 Verl 的 `agentic_main.py`）：

```python
# 在 train.py 的 train() 函数中，create_rollout_manager 之后添加：
from slime.rollout.remote_agent.proxy import start_proxy_server

if args.harbor_agent_name or args.harbor_agent_import_path:
    # 获取 SGLang engine handles
    engine_handles = ray.get(rollout_manager.rollout_engines)
    proxy_url = start_proxy_server(
        engine_handles=engine_handles,
        model_path=args.hf_checkpoint,
        host=args.harbor_proxy_host,
        port=args.harbor_proxy_port,
    )
    os.environ["LOCAL_IP"] = args.harbor_proxy_host
    logger.info(f"Harbor LLM Proxy started at {proxy_url}")
```

### 4.2 命令行

```bash
# 设置外部可达的 IP
export LOCAL_IP=10.0.30.11

python train.py \
  --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
  --harbor-server-url http://harbor-server:8080 \
  --harbor-agent-name swe-agent \
  --harbor-model-name openai/qwen-max \
  --harbor-task-path-template '/home/slime/dataset-tasks/{instance_id}' \
  --harbor-agent-kwargs '{"total_cost_limit":0,"per_instance_cost_limit":0}' \
  --harbor-proxy-host 10.0.30.11 \
  --harbor-max-retries 3 \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \
  --rollout-max-response-len 4096 \
  ...其他训练参数...
```

### 4.3 与现有扩展点的关系

| 扩展点 | 用途 | 是否冲突 |
|--------|------|---------|
| `--custom-generate-function-path` | 替换单个 sample 的生成逻辑 | **本方案使用** |
| `--rollout-function-path` | 替换整个 rollout 循环 | 不冲突（可同时用） |
| `--rollout-external` | 使用外部 SGLang 实例 | 正交（Proxy 上游可用外部 SGLang） |
| `--custom-rm-path` | 自定义 reward model | 不冲突 |

### 4.4 不需要本地 SGLang 的模式

如果 Agent 完全在远程 Harbor 中执行（不依赖本地模型），可以设置 `--debug-train-only` 跳过 SGLang 启动，但需要修改 Proxy 使其上游指向 Harbor 中的模型（而非本地 SGLang）。这种情况下 Proxy 只是一个 token 记录器：

```python
# 纯记录模式的 Proxy (不转发到 SGLang)
# 此时 Agent 使用自己的 LLM (如 GPT-4、Claude)，Proxy 仅记录
# 需要在 Harbor 的 Agent 配置中指定 llm_proxy_url 指向一个纯记录 Proxy
```

---

## 五、关键设计决策

### 5.1 为什么用 custom-generate-function 而不是替换 rollout

slime 的 `generate(args, sample, sampling_params)` 签名天然适配 Harbor 任务提交：
- 一个 sample = 一个 Harbor task
- 返回值是填充后的 Sample，与现有 pipeline 无缝对接
- RM、训练数据转换等下游逻辑完全不需要改动

而 Verl 必须引入 `RemoteAgentLoop` 是因为 verl 的 `AgentLoop` 抽象粒度不同——它返回的是 `AgentLoopOutput(prompt_ids, response_ids, response_mask, ...)`，与 slime 的 `Sample` 数据结构不同。

### 5.2 Proxy 通过 LiteLLM + SGLang Ray RPC

参考 Verl 的 `VLLMRayProvider` 设计，而不是简单的 HTTP 转发。好处：
- 直接获取 token_ids 和 logprobs，无需从 HTTP 响应中解析
- LiteLLM 自动处理 OpenAI 兼容的 JSON 格式化、SSE 流式
- Sticky-session 负载均衡

### 5.3 输出重建 (_reconstruct_output)

这是 RL 训练的关键。Agent 的多轮对话中：
- LLM 生成的 tokens → `mask=1`（参与 loss 计算）
- Tool execution 结果 / 用户回复 → `mask=0`（不参与 loss）

`_reconstruct_output` 遍历 session 的所有 turns，对比相邻 turn 的 `request_messages` 长度差，提取新增的 tool/user messages 并 tokenize。

### 5.4 与 Verl 的差异

| 方面 | Verl 实现 | Slime 方案 |
|------|----------|-----------|
| 集成点 | `RemoteAgentLoop` (AgentLoop 级别) | `custom-generate-function` (Sample 级别) |
| Proxy 部署 | Ray Named Actor (固定 head node) | 相同 |
| vLLM Provider | LiteLLM CustomLLM → vLLM Ray RPC | LiteLLM CustomLLM → SGLang Ray RPC |
| 任务提交 | `AgentRunClient` → Harbor HTTP | 相同的 HarborClient |
| 重试机制 | 指数退避 + session 重置 | 相同 |
| 输出格式 | `AgentLoopOutput` | `Sample` (填充 tokens/logprobs/mask) |

---

## 六、实施步骤

```
Phase 1: 核心组件
  ├── 1.1 slime/rollout/remote_agent/__init__.py
  ├── 1.2 slime/rollout/remote_agent/proxy.py
  │       └── SessionRecorder + TokenProxy + ProxyActor + SGLangRayProvider
  ├── 1.3 slime/rollout/remote_agent/harbor_client.py
  └── 1.4 slime/rollout/remote_agent/generate.py
          └── generate_with_harbor (custom-generate-function)

Phase 2: 参数集成
  ├── 2.1 slime/utils/arguments.py 添加 --harbor-* 参数
  └── 2.2 train.py 中集成 Proxy 启动逻辑

Phase 3: 示例
  ├── 3.1 examples/remote_agent/README.md
  ├── 3.2 examples/remote_agent/harbor_qwen.sh (完整启动脚本)
  └── 3.3 提供 Harbor Agent 配置示例 (swe-agent 等)

Phase 4: 测试
  ├── 4.1 单元测试: SessionRecorder, HarborClient, _reconstruct_output
  └── 4.2 集成测试: 使用 --harbor-use-local-trial 本地验证
```
