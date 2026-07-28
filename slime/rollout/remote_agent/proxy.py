"""LLM Proxy Server — capture token-level data from Harbor agent calls.

Architecture (Ray mode)::

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

Architecture (Standalone mode)::

    Engine 1 ── POST /engines/register ──→  ┌──────────────────────┐
    Engine 2 ── POST /engines/register ──→  │     TokenProxy       │
                                             │                      │
    Harbor Agent ── /{trial_id}/v1/chat/ ──→ │  - EngineRegistry    │
                    completions              │  - SGlangHTTPProvider│
                                             │  - SessionRecorder   │
                                             └──────────────────────┘

Usage from train.py::

    from slime.rollout.remote_agent.proxy import start_proxy_server

    proxy_url = start_proxy_server(
        engine_handles=engine_handles,
        model_path=args.***,
        host=args.harbor_proxy_host,
        port=args.harbor_proxy_port,
    )

Usage from generate function::

    from slime.rollout.remote_agent.proxy import get_proxy_url

    url = get_proxy_url()  # discover the proxy URL

Standalone usage::

    from slime.rollout.remote_agent.proxy import start_standalone_proxy

    proxy_url = start_standalone_proxy(
        model_path="/path/to/model",
        host="0.0.0.0",
        port=8888,
    )

    # Engines self-register:
    # POST /engines/register {"url": "http://localhost:30000", ...}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal
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


# ---------- Engine Registry (Standalone Mode) ----------


@dataclass
class EngineInfo:
    """Metadata for a registered SGLang engine."""
    engine_id: str
    url: str
    model: str
    max_concurrent: int
    heartbeat_interval: int
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    active_requests: int = 0
    total_requests: int = 0
    healthy: bool = True


class EngineRegistry:
    """Thread-safe registry for standalone-mode engines that self-register."""

    def __init__(self) -> None:
        self._engines: dict[str, EngineInfo] = {}
        self._lock = threading.Lock()
        # Sticky-session load balancing: session_id -> engine_id
        self._sticky: dict[str, str] = {}

    def register(
        self,
        url: str,
        model: str,
        max_concurrent: int = 8,
        heartbeat_interval: int = 30,
    ) -> EngineInfo:
        """Register a new engine. Returns the EngineInfo with assigned ID."""
        engine_id = f"eng_{uuid4().hex[:12]}"
        info = EngineInfo(
            engine_id=engine_id,
            url=url,
            model=model,
            max_concurrent=max_concurrent,
            heartbeat_interval=heartbeat_interval,
        )
        with self._lock:
            self._engines[engine_id] = info
        logger.info("Engine registered: %s (url=%s, max_concurrent=%d)", engine_id, url, max_concurrent)
        return info

    def unregister(self, engine_id: str) -> bool:
        """Deregister an engine. Returns True if found and removed."""
        with self._lock:
            removed = self._engines.pop(engine_id, None)
            if removed:
                # Clean up sticky bindings for this engine
                self._sticky = {k: v for k, v in self._sticky.items() if v != engine_id}
                logger.info("Engine unregistered: %s", engine_id)
                return True
        return False

    def heartbeat(self, engine_id: str) -> bool:
        """Update heartbeat timestamp. Returns True if engine exists."""
        with self._lock:
            engine = self._engines.get(engine_id)
            if engine:
                engine.last_heartbeat = time.time()
                engine.healthy = True
                return True
        return False

    def get_healthy_engines(self) -> list[EngineInfo]:
        """Return list of currently healthy engines."""
        now = time.time()
        with self._lock:
            healthy = []
            stale_ids = []
            for eid, eng in self._engines.items():
                ttl = eng.heartbeat_interval * 3
                if now - eng.last_heartbeat > ttl:
                    eng.healthy = False
                    stale_ids.append(eid)
                else:
                    healthy.append(eng)
            # Auto-remove stale engines
            for eid in stale_ids:
                del self._engines[eid]
                logger.warning("Engine %s removed (stale, no heartbeat for %ds)", eid, ttl)
            return healthy

    def choose_engine(self, session_id: str | None = None) -> str | None:
        """Pick an engine URL using sticky sessions + least-load."""
        engines = self.get_healthy_engines()
        if not engines:
            return None

        # Sticky session
        if session_id and session_id in self._sticky:
            target_id = self._sticky[session_id]
            for eng in engines:
                if eng.engine_id == target_id:
                    return eng.url
            # Sticky engine gone, fall through

        # Least-load: lowest (active_requests / max_concurrent) ratio
        best = min(engines, key=lambda e: e.active_requests / max(e.max_concurrent, 1))
        if session_id:
            self._sticky[session_id] = best.engine_id
        return best.url

    def release_session(self, session_id: str) -> None:
        """Remove sticky binding for a session."""
        with self._lock:
            self._sticky.pop(session_id, None)

    def increment_active(self, engine_id: str) -> None:
        with self._lock:
            eng = self._engines.get(engine_id)
            if eng:
                eng.active_requests += 1
                eng.total_requests += 1

    def decrement_active(self, engine_id: str) -> None:
        with self._lock:
            eng = self._engines.get(engine_id)
            if eng:
                eng.active_requests = max(0, eng.active_requests - 1)

    def list_engines(self) -> list[dict]:
        """Return engine list for debug endpoint."""
        engines = self.get_healthy_engines()
        return [
            {
                "engine_id": e.engine_id,
                "url": e.url,
                "model": e.model,
                "max_concurrent": e.max_concurrent,
                "active_requests": e.active_requests,
                "total_requests": e.total_requests,
                "healthy": e.healthy,
                "last_heartbeat": e.last_heartbeat,
            }
            for e in engines
        ]


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
        # Newer transformers may return a BatchEncoding / tensor instead of a
        # flat list[int]; coerce so it is JSON-serializable for the engine RPC.
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()

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


# ---------- SGLang HTTP Provider (Standalone Mode) ----------


class SGlangHTTPProvider:
    """LiteLLM custom provider that routes requests to SGLang via HTTP.

    Used in standalone mode where engines self-register with the proxy
    and the proxy load-balances across them.
    """

    def __init__(self, registry: EngineRegistry, tokenizer: AutoTokenizer):
        self.registry = registry
        self.tokenizer = tokenizer
        self._http_client = None

    async def _get_client(self):
        """Lazy-init the async HTTP client."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=300.0)
        return self._http_client

    def release_session(self, session_id: str) -> None:
        """Remove sticky-session binding."""
        self.registry.release_session(session_id)

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
        """Async chat completion via SGLang HTTP."""
        optional_params = optional_params or {}
        session_id = (optional_params.get("extra_body") or {}).get("session_id")
        temperature = optional_params.get("temperature", 1.0)
        top_p = optional_params.get("top_p", 1.0)
        max_tokens = optional_params.get("max_tokens", 2048)
        stop = optional_params.get("stop")

        # Pick an engine from the registry
        engine_url = self.registry.choose_engine(session_id)
        if engine_url is None:
            raise RuntimeError(
                "No healthy engines registered. "
                "Engines must POST /engines/register before generating."
            )

        engine_info = self.registry.get_engine_by_url(engine_url)
        if engine_info:
            self.registry.increment_active(engine_info.engine_id)

        try:
            # Tokenize
            normalized = _normalize_messages(messages)
            prompt_ids = self.tokenizer.apply_chat_template(
                normalized, add_generation_prompt=True, tokenize=True
            )
            # Newer transformers may return a BatchEncoding / tensor instead of a
            # flat list[int]; coerce so it is JSON-serializable for the engine RPC.
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids["input_ids"]
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()

            # Call SGLang via HTTP /generate endpoint
            sampling_params = {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_tokens,
            }
            if stop:
                sampling_params["stop"] = stop

            client = await self._get_client()
            resp = await client.post(
                f"{engine_url}/generate",
                json={
                    "input_ids": prompt_ids,
                    "sampling_params": sampling_params,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Parse response — SGLang /generate returns:
            # {"output_ids": [...], "meta_info": {...}}
            token_ids = data.get("output_ids", [])
            meta_info = data.get("meta_info", {})
            log_probs = meta_info.get("logprobs", [])
            stop_reason = meta_info.get("finish_reason", {}).get("type", "stop")

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
                    if isinstance(lp, dict):
                        # SGLang logprobs are dicts with token prob info
                        lp_val = lp.get("logprob", lp.get("token_logprob", 0))
                    else:
                        lp_val = float(lp)
                    tok_str = self.tokenizer.decode([tid])
                    content.append(
                        ChatCompletionTokenLogprob(
                            token=tok_str,
                            logprob=lp_val,
                            bytes=list(tok_str.encode("utf-8", errors="replace")),
                            top_logprobs=[],
                        )
                    )
                logprobs_obj = ChoiceLogprobs(content=content)

            model_response.choices = [
                Choices(
                    finish_reason=stop_reason if isinstance(stop_reason, str) else "stop",
                    index=0,
                    message=Message(role="assistant", content=completion_text),
                    logprobs=logprobs_obj,
                    provider_specific_fields={"token_ids": token_ids},
                )
            ]
            model_response.model = model
            model_response.usage = Usage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(token_ids),
                total_tokens=len(prompt_ids) + len(token_ids),
            )
            return model_response

        finally:
            if engine_info:
                self.registry.decrement_active(engine_info.engine_id)

    def get_engine_by_url(self, url: str) -> EngineInfo | None:
        """Find engine by URL."""
        return self.registry.get_engine_by_url(url)


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

    # -- Engine management (Standalone mode) --

    @app.post("/engines/register")
    async def register_engine(request: Request):
        """Register a new SGLang engine with the proxy."""
        body = await request.json()
        url = body.get("url")
        model = body.get("model", "unknown")
        max_concurrent = body.get("max_concurrent", 8)
        heartbeat_interval = body.get("heartbeat_interval", 30)

        if not url:
            raise HTTPException(status_code=400, detail="url is required")

        info = proxy.engine_registry.register(
            url=url,
            model=model,
            max_concurrent=max_concurrent,
            heartbeat_interval=heartbeat_interval,
        )
        return {
            "engine_id": info.engine_id,
            "heartbeat_url": f"{proxy.url}/engines/{info.engine_id}/heartbeat",
        }

    @app.post("/engines/{engine_id}/heartbeat")
    async def engine_heartbeat(engine_id: str):
        """Engine heartbeat — keeps the engine in the routing pool."""
        if proxy.engine_registry.heartbeat(engine_id):
            return {"status": "ok"}
        raise HTTPException(status_code=404, detail="Engine not found")

    @app.post("/engines/{engine_id}/unregister")
    async def unregister_engine(engine_id: str):
        """Deregister an engine from the proxy."""
        if proxy.engine_registry.unregister(engine_id):
            return {"status": "ok"}
        raise HTTPException(status_code=404, detail="Engine not found")

    @app.get("/engines")
    async def list_engines():
        """List all registered engines (debug endpoint)."""
        return {"engines": proxy.engine_registry.list_engines()}

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

    In Ray mode, designed to be a **singleton** in the Ray cluster.
    Created once when training starts and shared across all
    ``generate_with_harbor`` calls.

    In standalone mode, runs as a standalone process. Engines
    self-register via HTTP API.
    """

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 0,
        # Ray mode parameters
        engine_handles: list | None = None,
        # Standalone mode parameters
        mode: Literal["ray", "standalone"] = "ray",
    ):
        self.host = host
        self.port = port
        self.mode = mode
        self.engine_handles = engine_handles or []
        self.engine_registry = EngineRegistry()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.recorder = SessionRecorder()

        # Initialize the appropriate provider based on mode
        import litellm
        if mode == "standalone":
            self._provider = SGlangHTTPProvider(self.engine_registry, self.tokenizer)
            litellm.custom_provider_map = [
                {"provider": "slime-sglang", "custom_handler": self._provider}
            ]
        else:
            self._provider = SGLangRayProvider(self.engine_handles, self.tokenizer)
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

        logger.info("LLM Proxy started at %s (mode=%s)", self.url, self.mode)
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
            mode="ray",
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


# ---------- Standalone Proxy (No Ray Required) ----------


def start_standalone_proxy(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8888,
) -> TokenProxy:
    """Start the proxy server as a standalone process (no Ray required).

    Engines self-register via HTTP API:
        POST /engines/register {"url": "http://...", "model": "..."}

    Args:
        model_path: HuggingFace model name/path for the tokenizer.
        host: Bind address for the HTTP server.
        port: Port number.

    Returns:
        The TokenProxy instance with the running server.
        Use ``proxy.url`` to get the base URL.
    """
    proxy = TokenProxy(
        model_path=model_path,
        host=host,
        port=port,
        mode="standalone",
    )

    # Run in a new thread since this is a sync function
    loop = asyncio.new_event_loop()

    def _run_server():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(proxy.start())

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(100):
        if proxy.url is not None:
            break
        time.sleep(0.1)

    if proxy.url is None:
        raise RuntimeError("Failed to start standalone proxy server")

    logger.info("Standalone proxy started at %s", proxy.url)
    return proxy


# ---------- CLI Entry Point ----------


def main():
    """CLI entry point for running the proxy in standalone mode."""
    parser = argparse.ArgumentParser(
        description="Slime LLM Proxy Server (standalone mode)"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="HuggingFace model path for the tokenizer",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8888,
        help="Port number (default: 8888)",
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print(f"Starting Slime LLM Proxy (standalone mode)")
    print(f"  Model: {args.model_path}")
    print(f"  Listen: {args.host}:{args.port}")
    print()
    print("Engines can register with:")
    print(f"  POST http://{args.host}:{args.port}/engines/register")
    print(f'  {{"url": "http://<engine-host>:<port>", "model": "..."}}')
    print()

    proxy = start_standalone_proxy(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
    )

    print(f"Proxy running at {proxy.url}")
    print("Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
            # Print engine status periodically
            engines = proxy.engine_registry.list_engines()
            if engines:
                active = sum(e["active_requests"] for e in engines)
                print(f"  [{len(engines)} engine(s), {active} active requests]", end="\r")
    except KeyboardInterrupt:
        print("\nShutting down...")
        import threading
        loop = asyncio.new_event_loop()
        loop.run_until_complete(proxy.stop())


if __name__ == "__main__":
    main()
