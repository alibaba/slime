#!/usr/bin/env python3
"""Integration test for the standalone proxy server.

Starts the proxy with mocked dependencies, registers engines,
tests chat completions, and verifies session recording.

Usage:
    python test_proxy_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import httpx

# Ensure slime is importable
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Mock setup
# ---------------------------------------------------------------------------

def setup_mocks():
    """Create and return mock objects for heavy dependencies."""
    import importlib.util
    _spec = importlib.util.find_spec("importlib")
    
    def make_mock_module(name):
        m = MagicMock()
        m.__spec__ = _spec
        m.__name__ = name
        return m
    
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.0.0"
    mock_torch.__spec__ = _spec
    mock_torch.cuda.is_available.return_value = False
    
    mock_ray = MagicMock()
    mock_ray.__spec__ = _spec
    mock_ray.get_actor = MagicMock(side_effect=ValueError("No actor"))
    mock_ray.get = MagicMock(return_value=None)
    mock_ray.remote = MagicMock(return_value=lambda cls: cls)
    mock_ray.get_runtime_context.return_value.get_node_id.return_value = "head-node"
    mock_ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy = MagicMock()
    
    return {
        "torch": mock_torch,
        "torch.nn": make_mock_module("torch.nn"),
        "torch.nn.functional": make_mock_module("torch.nn.functional"),
        "torch.distributed": make_mock_module("torch.distributed"),
        "ray": mock_ray,
    }


class MockTokenizer:
    """A tokenizer that works without any model files."""
    
    def __init__(self):
        self.vocab_size = 1000
    
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        prompt = " ".join(str(m.get("content", "")) for m in messages)
        token_ids = []
        for i, char in enumerate(prompt[:100]):
            token_ids.append(ord(char) % self.vocab_size + 100)
        if not token_ids:
            token_ids = [100, 101, 102]
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=False):
        if not token_ids:
            return ""
        return f"Generated response with {len(token_ids)} tokens."
    
    def __getattr__(self, name):
        return MagicMock()


def start_proxy():
    """Start the proxy server with mocked dependencies."""
    print("=" * 60)
    print("STEP 1: Starting standalone proxy server")
    print("=" * 60)
    
    mock_modules = setup_mocks()
    
    with patch.dict("sys.modules", mock_modules):
        with patch("transformers.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = MockTokenizer()
            
            import slime.rollout.remote_agent.proxy as proxy_mod
            
            proxy = proxy_mod.TokenProxy(
                model_path="mock/model",
                host="127.0.0.1",
                port=18888,
                mode="standalone",
            )
            
            loop = asyncio.new_event_loop()
            
            def run_server():
                asyncio.set_event_loop(loop)
                loop.run_until_complete(proxy.start())
                loop.run_forever()
            
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            
            for i in range(50):
                if proxy.url is not None:
                    print(f"  Proxy started at {proxy.url}")
                    return proxy
                time.sleep(0.1)
            
            raise RuntimeError("Proxy failed to start")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_mock_litellm_response(content="Test response", token_ids=None, logprobs=None):
    """Create a mock LiteLLM ModelResponse."""
    if token_ids is None:
        token_ids = [201, 202, 203]
    if logprobs is None:
        logprobs = [-0.5, -0.3, -0.7]
    
    response = MagicMock()
    response.model = "slime-sglang/default"
    
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.index = 0
    
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice.message = message
    
    psf = {"token_ids": token_ids}
    choice.provider_specific_fields = psf
    
    choice.logprobs = MagicMock()
    choice.logprobs.content = [MagicMock(logprob=lp) for lp in logprobs]
    
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = len(token_ids)
    response.usage.total_tokens = 10 + len(token_ids)
    
    response.model_dump = lambda: {
        "model": response.model,
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "provider_specific_fields": psf,
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(token_ids),
            "total_tokens": 10 + len(token_ids),
        },
    }
    return response


async def run_test(name, coro):
    """Run a single test with output."""
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    print(f"{'─' * 60}")
    
    try:
        result = await coro
        print(f"  ✓ PASSED")
        if isinstance(result, dict):
            for k, v in result.items():
                val = str(v)
                if len(val) > 80:
                    val = val[:77] + "..."
                print(f"    {k}: {val}")
        return True
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def run_tests(proxy):
    """Run the full integration test suite."""
    base_url = proxy.url
    passed = 0
    total = 0
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
        
        # Test 1: Health check
        async def t1():
            resp = await client.get("/health")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            assert data["status"] == "ok"
            return data
        
        total += 1
        if await run_test("Health Check", t1()):
            passed += 1
        
        # Test 2: Register engine
        async def t2():
            resp = await client.post("/engines/register", json={
                "url": "http://127.0.0.1:30000",
                "model": "Qwen2.5-7B-Instruct",
                "max_concurrent": 8,
                "heartbeat_interval": 10,
            })
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            data = resp.json()
            assert data["engine_id"].startswith("eng_")
            assert "heartbeat_url" in data
            return data
        
        total += 1
        if await run_test("Register Engine #1", t2()):
            passed += 1
        
        # Test 3: Register second engine
        async def t3():
            resp = await client.post("/engines/register", json={
                "url": "http://127.0.0.1:30001",
                "model": "Qwen2.5-7B-Instruct",
                "max_concurrent": 4,
            })
            assert resp.status_code == 200
            return resp.json()
        
        total += 1
        if await run_test("Register Engine #2", t3()):
            passed += 1
        
        # Test 4: List engines
        async def t4():
            resp = await client.get("/engines")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["engines"]) == 2, f"Expected 2 engines, got {len(data['engines'])}"
            return {"count": len(data["engines"]), "urls": [e["url"] for e in data["engines"]]}
        
        total += 1
        if await run_test("List Engines", t4()):
            passed += 1
        
        # Test 5: Heartbeat
        async def t5():
            reg = await client.post("/engines/register", json={
                "url": "http://127.0.0.1:30002", "model": "test",
            })
            engine_id = reg.json()["engine_id"]
            resp = await client.post(f"/engines/{engine_id}/heartbeat")
            assert resp.status_code == 200
            return {"engine_id": engine_id}
        
        total += 1
        if await run_test("Engine Heartbeat", t5()):
            passed += 1
        
        # Test 6: Create session
        async def t6():
            resp = await client.post("/sessions/trial-001")
            assert resp.status_code == 200
            return resp.json()
        
        total += 1
        if await run_test("Create Session", t6()):
            passed += 1
        
        # Test 7: Chat completion
        async def t7():
            mock_resp = make_mock_litellm_response(
                content="This is the AI response to your query.",
                token_ids=[201, 202, 203, 204, 205, 206],
                logprobs=[-0.5, -0.3, -0.7, -0.2, -0.4, -0.6],
            )
            
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = mock_resp
                
                resp = await client.post("/v1/chat/completions/trial-001", json={
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100,
                })
                
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
                data = resp.json()
                assert "choices" in data
                return {
                    "status": resp.status_code,
                    "model": data.get("model"),
                    "finish_reason": data["choices"][0]["finish_reason"],
                }
        
        total += 1
        if await run_test("Chat Completion (mocked litellm)", t7()):
            passed += 1
        
        # Test 8: Verify session recording
        async def t8():
            resp = await client.get("/sessions/trial-001")
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "trial-001"
            assert len(data["turns"]) == 1, f"Expected 1 turn, got {len(data['turns'])}"
            
            turn = data["turns"][0]
            assert turn["completion_text"] == "This is the AI response to your query."
            assert turn["completion_token_ids"] == [201, 202, 203, 204, 205, 206]
            assert turn["finish_reason"] == "stop"
            assert len(turn["completion_logprobs"]) == 6
            
            return {
                "turns": len(data["turns"]),
                "text": turn["completion_text"][:50] + "...",
                "tokens": len(turn["completion_token_ids"]),
            }
        
        total += 1
        if await run_test("Session Recording", t8()):
            passed += 1
        
        # Test 9: Mark session complete
        async def t9():
            resp = await client.post("/sessions/trial-001/complete")
            assert resp.status_code == 200
            
            resp = await client.get("/sessions/trial-001")
            data = resp.json()
            assert data["completed"] is True
            return {"completed": True}
        
        total += 1
        if await run_test("Mark Session Complete", t9()):
            passed += 1
        
        # Test 10: Multi-turn conversation
        async def t10():
            for i in range(3):
                mock_resp = make_mock_litellm_response(
                    content=f"Response turn {i+2}",
                    token_ids=[300+i],
                )
                with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                    mock_ac.return_value = mock_resp
                    resp = await client.post("/v1/chat/completions/trial-002", json={
                        "messages": [{"role": "user", "content": f"Question {i+2}"}],
                    })
                    assert resp.status_code == 200
            
            resp = await client.get("/sessions/trial-002")
            data = resp.json()
            assert len(data["turns"]) == 3, f"Expected 3 turns, got {len(data['turns'])}"
            return {"turns": len(data["turns"])}
        
        total += 1
        if await run_test("Multi-Turn Conversation (3 turns)", t10()):
            passed += 1
        
        # Test 11: Alternate URL pattern
        async def t11():
            mock_resp = make_mock_litellm_response(content="Alternate URL response")
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = mock_resp
                resp = await client.post("/trial-003/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "Test alternate URL"}],
                })
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            
            resp = await client.get("/sessions/trial-003")
            data = resp.json()
            assert len(data["turns"]) == 1
            return {"status": 200}
        
        total += 1
        if await run_test("Alternate URL Pattern", t11()):
            passed += 1
        
        # Test 12: Error handling
        async def t12():
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.side_effect = RuntimeError("No healthy engines registered.")
                resp = await client.post("/v1/chat/completions/trial-error", json={
                    "messages": [{"role": "user", "content": "Test"}],
                })
                assert resp.status_code == 500, f"Expected 500, got {resp.status_code}"
                data = resp.json()
                assert "error" in data
                return {"status": 500, "error": data["error"][:60]}
        
        total += 1
        if await run_test("Error Handling (no engines)", t12()):
            passed += 1
        
        # Test 13: Session cleanup
        async def t13():
            resp = await client.delete("/sessions/trial-001")
            assert resp.status_code == 200
            
            resp = await client.get("/sessions/trial-001")
            assert resp.status_code == 404
            return {"deleted": True}
        
        total += 1
        if await run_test("Session Cleanup (delete)", t13()):
            passed += 1
        
        # Test 14: Unregister engine
        async def t14():
            reg = await client.post("/engines/register", json={
                "url": "http://127.0.0.1:30003", "model": "temp",
            })
            engine_id = reg.json()["engine_id"]
            
            # Verify it exists
            resp = await client.get("/engines")
            assert len(resp.json()["engines"]) >= 1
            
            # Unregister
            resp = await client.post(f"/engines/{engine_id}/unregister")
            assert resp.status_code == 200
            
            # Verify it's gone (check it's not in the list)
            resp = await client.get("/engines")
            engines = resp.json()["engines"]
            assert not any(e["engine_id"] == engine_id for e in engines)
            return {"unregistered": engine_id}
        
        total += 1
        if await run_test("Engine Unregister", t14()):
            passed += 1
        
        # Test 15: Register engine missing URL
        async def t15():
            resp = await client.post("/engines/register", json={"model": "test"})
            assert resp.status_code == 400
            assert "url is required" in resp.json()["detail"]
            return {"status": 400}
        
        total += 1
        if await run_test("Validation (missing URL)", t15()):
            passed += 1
    
    return passed, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Standalone Proxy Pipeline Test" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    proxy = start_proxy()
    print()
    
    passed, total = asyncio.run(run_tests(proxy))
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
