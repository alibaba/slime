"""LLM Proxy Server — capture token-level data from Harbor agent calls.

Architecture::

    Harbor Agent (in Docker/K8s)
         |  (OpenAI /v1/chat/completions)
         v
    +----------------------------------+
    |  TokenProxy  (Ray Named Actor)   |
    |  - FastAPI HTTP server           |
    |  - LiteLLM custom provider       |
    |  - SGLangRayProvider             |
    |  - SessionRecorder per trial_id  |
    +---------------+------------------+
                    |  engine.generate.remote(prompt_ids, ...)
                    v
             SGLang rollout servers

Usage from train.py::

    from slime.rollout.remote_agent.proxy import start_proxy_server

    proxy_url = start_proxy_server(
        engine_handles=engine_handles,
        model_path=args.hf_checkpoint,
        host=args.harbor_proxy_host,
        port=args.harbor_proxy_port,
    )

Usage from generate function::

    from slime.rollout.remote_agent.proxy import get_proxy_url

    url = get_proxy_url()  # discover the proxy URL
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import ray
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------- Data Models ----------


@dataclass
class CompletionRecord:
    """Record of a single LLM completion call captured by the proxy."""
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    completion_text: str = ""
    completion_token_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class SessionRecord:
    """All recorded data for a single agent session (one rollout)."""
    session_id: str
    turns: list[CompletionRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed: bool = False


# ---------- Session Recorder ----------


class SessionRecorder:
    """Thread-safe session data recorder for the LLM proxy.

    Multiple proxy request handlers may record completions concurrently
    for different sessions.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> None:
        """Create a new session for recording."""
        with self._lock:
            if session_id in self._sessions:
                logger.warning("Session %s already exists, resetting", session_id)
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
        """Record a single LLM completion for a session."""
        record = CompletionRecord(
            request_messages=messages,
            completion_text=completion_text,
            completion_token_ids=token_ids,
            completion_logprobs=logprobs,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(
                    "Session %s not found, auto-creating for recording", session_id
                )
                session = SessionRecord(session_id=session_id)
                self._sessions[session_id] = session
            session.turns.append(record)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve session data. Returns None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def mark_completed(self, session_id: str) -> None:
        """Mark a session as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.completed = True

    def delete_session(self, session_id: str) -> None:
        """Remove a session and free its memory."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())


# ---------- SGLang Ray Provider (LiteLLM CustomLLM) ----------


class SGLangRayProvider:
    """LiteLLM custom provider that routes requests to SGLang via Ray RPC.

    Encapsulates the core generate logic (tokenize -> Ray RPC -> decode)
    and returns standard ``litellm.ModelResponse`` objects.  LiteLLM then
    takes care of OpenAI-compatible JSON formatting, SSE streaming, etc.
    """

    def __init__(self, engine_handles: list, tokenizer: AutoTokenizer):
        self.engine_handles = list(engine_handles)
        self.tokenizer = tokenizer

        # Sticky-session load balancing: session_id -> server index
        self._sticky: dict[str, int] = {}
        self._request_counts: list[int] = [0] * len(engine_handles)

    # -- Load balancing --

    def _choose_server(self, session_id: str | None):
        """Pick a server handle using sticky sessions."""
        key = session_id or "default"
        if key in self._sticky:
            return self.engine_handles[self._sticky[key]]

        idx = min(
            range(len(self._request_counts)),
            key=lambda i: self._request_counts[i],
        )
        self._sticky[key] = idx
        self._request_counts[idx] += 1
        return self.engine_handles[idx]

    def release_session(self, session_id: str) -> None:
        """Remove sticky-session binding for the given session."""
        self._sticky.pop(session_id, None)

    # -- LiteLLM CustomLLM interface --

    async def acompletion(
        self,
        model,
        messages,
        api_base,
        custom_prompt_dict,
        model_response,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params,
        **kwargs,
    ):
        """Async chat completion via SGLang Ray RPC."""
        optional_params = optional_params or {}
        session_id = (optional_params.get("extra_body") or {}).get("session_id")
        temperature = optional_params.get("temperature", 1.0)
        top_p = optional_params.get("top_p", 1.0)
        max_tokens = optional_params.get("max_tokens", 2048)
        stop = optional_params.get("stop")

        # Tokenize
        normalized = _normalize_messages(messages)
        prompt_ids = self.tokenizer.apply_chat_template(
            normalized, add_generation_prompt=True, tokenize=True
        )

        # Call SGLang via Ray RPC
        server = self._choose_server(session_id)
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        if stop:
            sampling_params["stop"] = stop

        request_id = uuid4().hex
        ref = server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        output = await asyncio.to_thread(ray.get, ref)

        token_ids = list(output.token_ids) if hasattr(output, "token_ids") else []
        log_probs = (
            list(output.log_probs)
            if hasattr(output, "log_probs") and output.log_probs
            else []
        )
        stop_reason = getattr(output, "stop_reason", None)
        completion_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        # Build LiteLLM ModelResponse
        from litellm.types.utils import (
            ChatCompletionTokenLogprob,
            ChoiceLogprobs,
            Choices,
            Message,
            Usage,
        )

        logprobs_obj = None
        if log_probs:
            content = []
            for tid, lp in zip(token_ids, log_probs):
                tok_str = self.tokenizer.decode([tid])
                content.append(
                    ChatCompletionTokenLogprob(
                        token=tok_str,
                        logprob=lp,
                        bytes=list(tok_str.encode("utf-8", errors="replace")),
                        top_logprobs=[],
                    )
                )
            logprobs_obj = ChoiceLogprobs(content=content)

        model_response.choices = [
            Choices(
                finish_reason=stop_reason or "stop",
                index=0,
                message=Message(role="assistant", content=completion_text),
                logprobs=logprobs_obj,
                provider_specific_fields={"token_ids": token_ids},
            )
        ]
        model_response.model = model
        model_response.usage = Usage(
            prompt_tokens=0,
            completion_tokens=len(token_ids),
            total_tokens=len(token_ids),
        )
        return model_response


# ---------- Message Normalization ----------


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize message content for ``apply_chat_template``.

    Handles multi-part content lists (OpenAI vision format) -> plain strings.
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif "text" in part:
                        text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            msg["content"] = "\n".join(text_parts) if text_parts else ""
        normalized.append(msg)
    return normalized


# ---------- Helpers for session recording ----------


def _record_from_response(
    proxy: "TokenProxy",
    trial_id: str,
    messages: list[dict[str, Any]],
    response: Any,
) -> None:
    """Extract token_ids/logprobs from a LiteLLM ModelResponse and record."""
    choice = response.choices[0]

    # token_ids via provider_specific_fields
    psf = getattr(choice, "provider_specific_fields", None) or {}
    token_ids = psf.get("token_ids", [])

    # logprobs from standard ChoiceLogprobs
    logprobs_list: list[float] = []
    choice_logprobs = getattr(choice, "logprobs", None)
    if choice_logprobs and hasattr(choice_logprobs, "content") and choice_logprobs.content:
        logprobs_list = [t.logprob for t in choice_logprobs.content]

    # tool_calls
    tool_calls = None
    msg = choice.message
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = [
            tc.model_dump() if hasattr(tc, "model_dump") else tc
            for tc in msg.tool_calls
        ]

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


async def _stream_and_record(
    proxy: "TokenProxy",
    trial_id: str,
    messages: list[dict[str, Any]],
    stream,
) -> Any:
    """Iterate a LiteLLM streaming response, yield SSE chunks, and record."""
    collected_text_parts: list[str] = []
    collected_token_ids: list[int] = []
    collected_logprobs: list[float] = []
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


# ---------- FastAPI application builder ----------


def _build_app(proxy: "TokenProxy") -> FastAPI:
    """Build the FastAPI application that serves as the OpenAI proxy."""

    app = FastAPI(title="Slime LLM Proxy", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # -- Session management --

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

    # -- OpenAI-compatible chat completions --
    # Two URL patterns so the proxy works with both:
    #   1. Agents that call a fully custom endpoint
    #   2. Standard OpenAI SDK where base_url = http://host:port/{trial_id}/v1

    @app.post("/v1/chat/completions/{trial_id}")
    @app.post("/{trial_id}/v1/chat/completions")
    async def chat_completions(trial_id: str, request: Request):
        """Generate a chat completion via LiteLLM+SGLang and record data."""
        body = await request.json()
        messages: list[dict[str, Any]] = body.get("messages", [])
        is_streaming: bool = body.get("stream", False)

        completion_kwargs: dict[str, Any] = {
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
            import litellm
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as e:
            logger.error("Generate failed for trial %s: %s", trial_id, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

        if is_streaming:
            return StreamingResponse(
                _stream_and_record(proxy, trial_id, messages, response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: record and return
        _record_from_response(proxy, trial_id, messages, response)
        return JSONResponse(content=response.model_dump())

    return app


# ---------- TokenProxy ----------


class TokenProxy:
    """LLM Proxy server that holds SGLang server handles and serves
    OpenAI-compatible HTTP requests via LiteLLM.

    Designed to be a **singleton** in the Ray cluster.
    Created once when training starts and shared across all
    ``generate_with_harbor`` calls.
    """

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

        # Initialize LiteLLM custom provider
        self._provider = SGLangRayProvider(engine_handles, self.tokenizer)
        import litellm
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
        """Start the HTTP server.  Returns the base URL."""
        if self.port == 0:
            self._actual_port = _find_free_port()
        else:
            self._actual_port = self.port

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self._actual_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None

        async def _safe_serve():
            try:
                await self._server.serve()
            except asyncio.CancelledError:
                logger.warning("Proxy server task cancelled")
                raise

        self._serve_task = asyncio.create_task(_safe_serve())

        for _ in range(100):
            if self._server.started:
                break
            await asyncio.sleep(0.1)

        logger.info("LLM Proxy started at %s", self.url)
        return self.url

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._server is not None:
            self._server.should_exit = True
            if self._serve_task is not None:
                try:
                    await self._serve_task
                except asyncio.CancelledError:
                    pass
                self._serve_task = None
            self._server = None
            logger.info("LLM Proxy stopped")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------- Ray Named Actor ----------


PROXY_ACTOR_NAME = "slime_llm_proxy"
_proxy_actor_handle = None


@ray.remote
class ProxyActor:
    """Ray actor wrapping ``TokenProxy``, pinned to head node."""

    def __init__(
        self,
        engine_handles: list,
        model_path: str,
        host: str,
        port: int,
    ):
        self.proxy = TokenProxy(
            engine_handles=engine_handles,
            model_path=model_path,
            host=host,
            port=port,
        )
        self._url: str | None = None

    async def start(self) -> str:
        """Start the proxy HTTP server and return a cluster-routable URL."""
        self._url = await self.proxy.start()
        logger.info("ProxyActor started at %s", self._url)
        return self._url

    def get_url(self) -> str | None:
        """Return the cluster-routable proxy URL, or ``None`` if not started."""
        return self._url

    async def stop(self) -> None:
        """Gracefully shut down the proxy HTTP server."""
        if self.proxy is not None:
            await self.proxy.stop()
            logger.info("ProxyActor stopped")


def start_proxy_server(
    engine_handles: list,
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 0,
) -> str:
    """Start the proxy server as a Ray named actor on the head node.

    If an actor with the same name already exists, its URL is returned
    immediately without creating a new one.

    This function is intended to be called from ``train.py`` **after**
    the rollout manager has been created and engine handles are available.

    Args:
        engine_handles: Ray actor handles of SGLang rollout engines.
        model_path: HuggingFace model name/path for the tokenizer.
        host: Bind address for the HTTP server.
        port: Port number. ``0`` means auto-select a free port.

    Returns:
        The cluster-routable proxy URL, e.g. ``http://10.0.1.5:9123``.
    """
    global _proxy_actor_handle

    # Fast path: reuse existing actor
    try:
        existing = ray.get_actor(PROXY_ACTOR_NAME)
        url = ray.get(existing.get_url.remote())
        if url is not None:
            logger.info("Reusing existing proxy server actor at %s", url)
            return url
    except ValueError:
        pass

    # Create new actor, pinned to the current (head) node
    head_node_id = ray.get_runtime_context().get_node_id()
    actor = ProxyActor.options(
        name=PROXY_ACTOR_NAME,
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(engine_handles, model_path, host, port)

    _proxy_actor_handle = actor
    url = ray.get(actor.start.remote())
    logger.info("Proxy server actor created at %s (head node %s)", url, head_node_id)
    return url


def get_proxy_url() -> str | None:
    """Get the URL of an already-running proxy server.

    Performs a lightweight Ray RPC to the named actor.  Returns ``None``
    if the actor does not exist.

    This is the primary entry point for ``generate_with_harbor`` on any node
    to discover the proxy address.
    """
    try:
        actor = ray.get_actor(PROXY_ACTOR_NAME)
        return ray.get(actor.get_url.remote())
    except ValueError:
        return None


def stop_proxy_server() -> None:
    """Stop the proxy server actor and remove it from the cluster."""
    global _proxy_actor_handle
    try:
        actor = ray.get_actor(PROXY_ACTOR_NAME)
        ray.get(actor.stop.remote())
        ray.kill(actor)
        logger.info("Proxy server actor stopped and killed")
    except ValueError:
        pass
    _proxy_actor_handle = None
