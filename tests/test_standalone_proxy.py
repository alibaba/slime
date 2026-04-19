"""Mock tests for the standalone proxy server.

Tests the proxy's core functionality without requiring:
- Ray cluster
- Real SGLang engines
- Real Harbor agents

The pure-Python classes (EngineRegistry, SessionRecorder) are tested in
isolation. HTTP endpoint tests require the full slime environment.

Run with:
    cd ~/projects/slime
    pip install -e .  # if not already installed
    pytest tests/test_standalone_proxy.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Inline copies of the classes under test (no slime import needed).
# These MUST stay in sync with proxy.py — the goal is to test the logic,
# not the import machinery.
# ---------------------------------------------------------------------------


@dataclass
class CompletionRecord:
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    completion_text: str = ""
    completion_token_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class SessionRecord:
    session_id: str
    turns: list[CompletionRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed: bool = False


class SessionRecorder:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                pass  # silently allow re-create
            self._sessions[session_id] = SessionRecord(session_id=session_id)

    def record_completion(
        self, session_id: str, messages: list[dict[str, Any]],
        completion_text: str, token_ids: list[int], logprobs: list[float],
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        record = CompletionRecord(
            request_messages=messages, completion_text=completion_text,
            completion_token_ids=token_ids, completion_logprobs=logprobs,
            finish_reason=finish_reason, tool_calls=tool_calls,
        )
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionRecord(session_id=session_id)
                self._sessions[session_id] = session
            session.turns.append(record)

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._sessions.get(session_id)

    def mark_completed(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.completed = True

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


@dataclass
class EngineInfo:
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
    def __init__(self) -> None:
        self._engines: dict[str, EngineInfo] = {}
        self._lock = threading.Lock()
        self._sticky: dict[str, str] = {}

    def register(self, url: str, model: str, max_concurrent: int = 8,
                 heartbeat_interval: int = 30) -> EngineInfo:
        engine_id = f"eng_{uuid4().hex[:12]}"
        info = EngineInfo(
            engine_id=engine_id, url=url, model=model,
            max_concurrent=max_concurrent, heartbeat_interval=heartbeat_interval,
        )
        with self._lock:
            self._engines[engine_id] = info
        return info

    def unregister(self, engine_id: str) -> bool:
        with self._lock:
            removed = self._engines.pop(engine_id, None)
            if removed:
                self._sticky = {k: v for k, v in self._sticky.items() if v != engine_id}
                return True
        return False

    def heartbeat(self, engine_id: str) -> bool:
        with self._lock:
            engine = self._engines.get(engine_id)
            if engine:
                engine.last_heartbeat = time.time()
                engine.healthy = True
                return True
        return False

    def get_healthy_engines(self) -> list[EngineInfo]:
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
            for eid in stale_ids:
                del self._engines[eid]
            return healthy

    def choose_engine(self, session_id: str | None = None) -> str | None:
        engines = self.get_healthy_engines()
        if not engines:
            return None
        if session_id and session_id in self._sticky:
            target_id = self._sticky[session_id]
            for eng in engines:
                if eng.engine_id == target_id:
                    return eng.url
        best = min(engines, key=lambda e: e.active_requests / max(e.max_concurrent, 1))
        if session_id:
            self._sticky[session_id] = best.engine_id
        return best.url

    def release_session(self, session_id: str) -> None:
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
        engines = self.get_healthy_engines()
        return [
            {
                "engine_id": e.engine_id, "url": e.url, "model": e.model,
                "max_concurrent": e.max_concurrent,
                "active_requests": e.active_requests,
                "total_requests": e.total_requests,
                "healthy": e.healthy, "last_heartbeat": e.last_heartbeat,
            }
            for e in engines
        ]


# ---------------------------------------------------------------------------
# Engine Registry Tests (pure Python, no dependencies)
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    def test_register_engine(self):
        registry = EngineRegistry()
        info = registry.register(url="http://localhost:30000", model="Qwen2.5-7B", max_concurrent=8)
        assert info.engine_id.startswith("eng_")
        assert info.url == "http://localhost:30000"
        assert info.model == "Qwen2.5-7B"
        assert info.max_concurrent == 8
        assert info.healthy is True
        assert info.active_requests == 0

    def test_unregister_engine(self):
        registry = EngineRegistry()
        info = registry.register(url="http://localhost:30000", model="test")
        assert registry.unregister(info.engine_id) is True
        assert len(registry.get_healthy_engines()) == 0
        assert registry.unregister(info.engine_id) is False

    def test_heartbeat(self):
        registry = EngineRegistry()
        info = registry.register(url="http://localhost:30000", model="test")
        assert registry.heartbeat(info.engine_id) is True
        assert registry.heartbeat("nonexistent") is False

    def test_choose_engine_sticky(self):
        registry = EngineRegistry()
        registry.register(url="http://localhost:30000", model="test")
        registry.register(url="http://localhost:30001", model="test")

        url1 = registry.choose_engine("session-1")
        url2 = registry.choose_engine("session-1")
        assert url1 == url2, "Same session should always get the same engine"

        url3 = registry.choose_engine("session-2")
        assert url3 is not None

    def test_choose_engine_no_engines(self):
        registry = EngineRegistry()
        assert registry.choose_engine() is None

    def test_stale_engine_auto_remove(self):
        registry = EngineRegistry()
        info = registry.register(url="http://localhost:30000", model="test", heartbeat_interval=1)

        assert len(registry.get_healthy_engines()) == 1

        # Simulate time passing beyond TTL (3 * heartbeat_interval = 3s)
        info.last_heartbeat = time.time() - 10

        healthy = registry.get_healthy_engines()
        assert len(healthy) == 0, "Stale engine should be auto-removed"

    def test_load_balancing_least_load(self):
        registry = EngineRegistry()
        info1 = registry.register(url="http://localhost:30000", model="test", max_concurrent=10)
        info2 = registry.register(url="http://localhost:30001", model="test", max_concurrent=10)

        # Simulate engine1 being heavily loaded
        for _ in range(5):
            registry.increment_active(info1.engine_id)

        # New session should pick engine2 (less loaded)
        url = registry.choose_engine("new-session")
        assert url == "http://localhost:30001"

    def test_increment_decrement_active(self):
        registry = EngineRegistry()
        info = registry.register(url="http://localhost:30000", model="test")

        assert info.active_requests == 0
        registry.increment_active(info.engine_id)
        assert info.active_requests == 1
        registry.increment_active(info.engine_id)
        assert info.active_requests == 2
        registry.decrement_active(info.engine_id)
        assert info.active_requests == 1
        registry.decrement_active(info.engine_id)
        assert info.active_requests == 0
        # Should not go negative
        registry.decrement_active(info.engine_id)
        assert info.active_requests == 0

    def test_release_session(self):
        registry = EngineRegistry()
        registry.register(url="http://localhost:30000", model="test")
        registry.choose_engine("session-1")
        assert "session-1" in registry._sticky
        registry.release_session("session-1")
        assert "session-1" not in registry._sticky

    def test_list_engines(self):
        registry = EngineRegistry()
        registry.register(url="http://localhost:30000", model="model-a")
        registry.register(url="http://localhost:30001", model="model-b")

        engines = registry.list_engines()
        assert len(engines) == 2
        assert engines[0]["url"] == "http://localhost:30000"
        assert engines[0]["model"] == "model-a"

    def test_total_requests_counter(self):
        registry = EngineRegistry()
        info = registry.register(url="http://localhost:30000", model="test")

        assert info.total_requests == 0
        for _ in range(10):
            registry.increment_active(info.engine_id)
            registry.decrement_active(info.engine_id)
        assert info.total_requests == 10
        assert info.active_requests == 0


# ---------------------------------------------------------------------------
# SessionRecorder Tests (pure Python, no dependencies)
# ---------------------------------------------------------------------------


class TestSessionRecorder:
    def test_create_and_record(self):
        recorder = SessionRecorder()
        recorder.create_session("s1")
        recorder.record_completion(
            session_id="s1", messages=[{"role": "user", "content": "hi"}],
            completion_text="hello", token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3], finish_reason="stop",
        )
        session = recorder.get_session("s1")
        assert session is not None
        assert len(session.turns) == 1
        assert session.turns[0].completion_text == "hello"

    def test_auto_create_on_record(self):
        recorder = SessionRecorder()
        recorder.record_completion(
            session_id="s2", messages=[], completion_text="auto",
            token_ids=[1], logprobs=[0.0],
        )
        session = recorder.get_session("s2")
        assert session is not None
        assert len(session.turns) == 1

    def test_list_sessions(self):
        recorder = SessionRecorder()
        recorder.create_session("a")
        recorder.create_session("b")
        recorder.create_session("c")
        sessions = recorder.list_sessions()
        assert set(sessions) == {"a", "b", "c"}

    def test_delete_session(self):
        recorder = SessionRecorder()
        recorder.create_session("x")
        recorder.delete_session("x")
        assert recorder.get_session("x") is None

    def test_mark_completed(self):
        recorder = SessionRecorder()
        recorder.create_session("s1")
        assert recorder.get_session("s1").completed is False
        recorder.mark_completed("s1")
        assert recorder.get_session("s1").completed is True

    def test_multiple_turns(self):
        recorder = SessionRecorder()
        recorder.create_session("multi")
        for i in range(5):
            recorder.record_completion(
                session_id="multi", messages=[],
                completion_text=f"turn-{i}", token_ids=[i], logprobs=[0.0],
            )
        session = recorder.get_session("multi")
        assert len(session.turns) == 5
        assert session.turns[0].completion_text == "turn-0"
        assert session.turns[4].completion_text == "turn-4"

    def test_thread_safety(self):
        recorder = SessionRecorder()
        errors = []

        def record_many(prefix):
            try:
                for i in range(50):
                    recorder.create_session(f"{prefix}-{i}")
                    recorder.record_completion(
                        session_id=f"{prefix}-{i}", messages=[],
                        completion_text=f"response-{i}", token_ids=[i],
                        logprobs=[0.0],
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many, args=(f"t{j}",)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(recorder.list_sessions()) == 200


# ---------------------------------------------------------------------------
# HTTP Endpoint Tests (requires full slime environment)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Health endpoint — requires slime to be installed."""

    async def test_health(self):
        proxy = self._make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    def _make_proxy(self):
        return _import_and_make_proxy()


@pytest.mark.asyncio
class TestSessionEndpoints:
    async def test_create_session(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            resp = await client.post("/sessions/test-session-1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "test-session-1"

    async def test_get_session(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            await client.post("/sessions/test-session-2")
            resp = await client.get("/sessions/test-session-2")
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "test-session-2"
            assert data["turns"] == []

    async def test_get_nonexistent_session(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            resp = await client.get("/sessions/nonexistent")
            assert resp.status_code == 404

    async def test_complete_session(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            await client.post("/sessions/test-session-3")
            await client.post("/sessions/test-session-3/complete")
            resp = await client.get("/sessions/test-session-3")
            assert resp.json()["completed"] is True

    async def test_delete_session(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            await client.post("/sessions/test-session-4")
            await client.delete("/sessions/test-session-4")
            resp = await client.get("/sessions/test-session-4")
            assert resp.status_code == 404


@pytest.mark.asyncio
class TestEngineEndpoints:
    async def test_register_engine(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            resp = await client.post("/engines/register", json={
                "url": "http://localhost:30000",
                "model": "Qwen2.5-7B-Instruct",
                "max_concurrent": 8,
                "heartbeat_interval": 30,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["engine_id"].startswith("eng_")
            assert "heartbeat_url" in data

    async def test_register_engine_missing_url(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            resp = await client.post("/engines/register", json={"model": "test-model"})
            assert resp.status_code == 400
            assert "url is required" in resp.json()["detail"]

    async def test_heartbeat(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            reg = await client.post("/engines/register", json={"url": "http://localhost:30000", "model": "test"})
            engine_id = reg.json()["engine_id"]
            resp = await client.post(f"/engines/{engine_id}/heartbeat")
            assert resp.status_code == 200

    async def test_heartbeat_nonexistent(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            resp = await client.post("/engines/nonexistent/heartbeat")
            assert resp.status_code == 404

    async def test_unregister_engine(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            reg = await client.post("/engines/register", json={"url": "http://localhost:30000", "model": "test"})
            engine_id = reg.json()["engine_id"]
            resp = await client.post(f"/engines/{engine_id}/unregister")
            assert resp.status_code == 200

    async def test_list_engines(self):
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            await client.post("/engines/register", json={"url": "http://localhost:30000", "model": "a"})
            await client.post("/engines/register", json={"url": "http://localhost:30001", "model": "b"})
            resp = await client.get("/engines")
            assert resp.status_code == 200
            assert len(resp.json()["engines"]) == 2


@pytest.mark.asyncio
class TestChatCompletions:
    async def test_chat_completion_no_engines(self):
        """Chat should fail gracefully when no engines are registered."""
        proxy = _import_and_make_proxy()
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.side_effect = RuntimeError("No healthy engines registered.")
                resp = await client.post("/v1/chat/completions/test-trial", json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7, "max_tokens": 100,
                })
                assert resp.status_code == 500

    async def test_chat_completion_with_engine(self):
        """Chat completion with a registered engine (mocked litellm)."""
        proxy = _import_and_make_proxy()
        import httpx
        mock_resp = _make_mock_litellm_response()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            await client.post("/engines/register", json={
                "url": "http://localhost:30000", "model": "Qwen2.5-7B-Instruct",
            })
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = mock_resp
                resp = await client.post("/v1/chat/completions/test-trial-1", json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7, "max_tokens": 100,
                })
                assert resp.status_code == 200
                data = resp.json()
                assert "choices" in data

                # Verify session was recorded
                session_resp = await client.get("/sessions/test-trial-1")
                assert session_resp.status_code == 200
                session_data = session_resp.json()
                assert len(session_data["turns"]) == 1

    async def test_chat_completion_alternate_url_pattern(self):
        """Test the /{trial_id}/v1/chat/completions URL pattern."""
        proxy = _import_and_make_proxy()
        import httpx
        mock_resp = _make_mock_litellm_response()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            await client.post("/engines/register", json={"url": "http://localhost:30000", "model": "test"})
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = mock_resp
                resp = await client.post("/test-trial-2/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "Test"}],
                })
                assert resp.status_code == 200

    async def test_chat_completion_with_tools(self):
        """Chat completion with tool definitions."""
        proxy = _import_and_make_proxy()
        import httpx
        mock_resp = _make_mock_litellm_response()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            await client.post("/engines/register", json={"url": "http://localhost:30000", "model": "test"})
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = mock_resp
                resp = await client.post("/v1/chat/completions/test-trial-tools", json={
                    "messages": [{"role": "user", "content": "What's the weather?"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                })
                assert resp.status_code == 200


@pytest.mark.asyncio
class TestEndToEndFlow:
    """Test the full agent workflow: register → session → chat → complete → fetch."""

    async def test_full_workflow(self):
        proxy = _import_and_make_proxy()
        import httpx
        mock_resp = _make_mock_litellm_response()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            # 1. Register engine
            reg = await client.post("/engines/register", json={
                "url": "http://localhost:30000", "model": "Qwen2.5-7B-Instruct",
                "max_concurrent": 4, "heartbeat_interval": 10,
            })
            assert reg.status_code == 200

            # 2. Create session
            await client.post("/sessions/trial-001")

            # 3. Chat completion
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = mock_resp
                resp = await client.post("/v1/chat/completions/trial-001", json={
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                    "temperature": 0.7, "max_tokens": 50,
                })
                assert resp.status_code == 200

            # 4. Mark session complete
            resp = await client.post("/sessions/trial-001/complete")
            assert resp.status_code == 200

            # 5. Fetch session recording
            resp = await client.get("/sessions/trial-001")
            assert resp.status_code == 200
            data = resp.json()
            assert data["completed"] is True
            assert len(data["turns"]) == 1
            assert data["turns"][0]["completion_text"] == "Hello, I am the model response."
            assert data["turns"][0]["completion_token_ids"] == [101, 102, 103, 104, 105]

            # 6. Cleanup
            await client.delete("/sessions/trial-001")
            resp = await client.get("/sessions/trial-001")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Helper functions for HTTP tests
# ---------------------------------------------------------------------------


def _make_mock_litellm_response():
    response = MagicMock()
    response.model = "slime-sglang/default"
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.index = 0
    message = MagicMock()
    message.content = "Hello, I am the model response."
    message.tool_calls = None
    choice.message = message
    psf = {"token_ids": [101, 102, 103, 104, 105]}
    choice.provider_specific_fields = psf
    choice.logprobs = None
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 5
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 10
    response.model_dump = lambda: {
        "model": response.model,
        "choices": [{"finish_reason": "stop", "index": 0,
                      "message": {"role": "assistant", "content": message.content},
                      "provider_specific_fields": psf}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    return response


def _import_and_make_proxy():
    """Import proxy module with mocked deps and create a proxy instance."""
    with patch.dict("sys.modules", {"ray": MagicMock()}):
        with patch("transformers.AutoTokenizer") as mock_at:
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template = lambda messages, add_generation_prompt=True, tokenize=True: [1, 2, 3, 4, 5]
            mock_tokenizer.decode = lambda token_ids, skip_special_tokens=False: "Hello, I am the model response."
            mock_at.from_pretrained.return_value = mock_tokenizer

            import importlib
            import slime.rollout.remote_agent.proxy as proxy_mod
            try:
                importlib.reload(proxy_mod)
            except Exception:
                pass

            proxy = proxy_mod.TokenProxy(
                model_path="fake/model", host="127.0.0.1", port=0, mode="standalone",
            )
            return proxy
