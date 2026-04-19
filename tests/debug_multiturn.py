#!/usr/bin/env python3
"""Debug script to test multi-turn conversation with full logging."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import httpx

_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Enable debug logging for the proxy
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("slime.rollout.remote_agent.proxy").setLevel(logging.DEBUG)


def setup_mocks():
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
    def __init__(self):
        self.vocab_size = 1000
    
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        prompt = " ".join(str(m.get("content", "")) for m in messages)
        token_ids = [ord(c) % self.vocab_size + 100 for c in prompt[:50]]
        if not token_ids:
            token_ids = [100, 101, 102]
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=False):
        return f"Response ({len(token_ids)} tokens)"
    
    def __getattr__(self, name):
        return MagicMock()


def make_mock_response(content, token_ids):
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
    choice.logprobs.content = [MagicMock(logprob=-0.5)]
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = len(token_ids)
    response.usage.total_tokens = 10 + len(token_ids)
    response.model_dump = lambda: {
        "model": response.model,
        "choices": [{
            "finish_reason": "stop", "index": 0,
            "message": {"role": "assistant", "content": content},
            "provider_specific_fields": psf,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(token_ids), "total_tokens": 10 + len(token_ids)},
    }
    return response


def main():
    mock_modules = setup_mocks()
    
    with patch.dict("sys.modules", mock_modules):
        with patch("transformers.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = MockTokenizer()
            
            import slime.rollout.remote_agent.proxy as proxy_mod
            
            proxy = proxy_mod.TokenProxy(
                model_path="mock/model",
                host="127.0.0.1",
                port=18890,
                mode="standalone",
            )
            
            loop = asyncio.new_event_loop()
            def run_server():
                asyncio.set_event_loop(loop)
                loop.run_until_complete(proxy.start())
                loop.run_forever()
            
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            
            for _ in range(50):
                if proxy.url:
                    break
                time.sleep(0.1)
            
            print("\n" + "=" * 70)
            print("  MULTI-TURN CONVERSATION TEST — FULL LOG")
            print("=" * 70)
            
            async def run_test():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=proxy.app),
                    base_url="http://test"
                ) as client:
                    
                    # Step 1: Register engine
                    print("\n>>> Registering engine...")
                    resp = await client.post("/engines/register", json={
                        "url": "http://127.0.0.1:30000",
                        "model": "Qwen2.5-7B",
                    })
                    print(f"    → {resp.json()}")
                    
                    # Step 2: Create session
                    print("\n>>> Creating session trial-multi...")
                    resp = await client.post("/sessions/trial-multi")
                    print(f"    → {resp.json()}")
                    
                    # Step 3: Multi-turn conversation
                    questions = [
                        "What is 2+2?",
                        "Now multiply that by 3",
                        "Add 10 to the result",
                    ]
                    answers = [
                        "The answer is 4.",
                        "4 multiplied by 3 is 12.",
                        "12 plus 10 is 22.",
                    ]
                    
                    for i, (q, a) in enumerate(zip(questions, answers), 1):
                        print(f"\n{'─' * 70}")
                        print(f"  TURN {i}")
                        print(f"  User:  {q}")
                        print(f"{'─' * 70}")
                        
                        mock_resp = make_mock_response(a, [500+i*100, 501+i*100, 502+i*100])
                        
                        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                            mock_ac.return_value = mock_resp
                            
                            print(f"  → POST /v1/chat/completions/trial-multi")
                            print(f"     messages: [user: '{q}']")
                            
                            resp = await client.post("/v1/chat/completions/trial-multi", json={
                                "messages": [{"role": "user", "content": q}],
                                "temperature": 0.7,
                                "max_tokens": 100,
                            })
                            
                            data = resp.json()
                            print(f"  ← Status: {resp.status_code}")
                            print(f"  ← Model:  {data.get('model')}")
                            print(f"  ← Finish: {data['choices'][0]['finish_reason']}")
                            print(f"  ← Response: '{a}'")
                            print(f"  ← Tokens:   {[500+i*100, 501+i*100, 502+i*100]}")
                    
                    # Step 4: Fetch session recording
                    print(f"\n{'=' * 70}")
                    print("  SESSION RECORDING")
                    print(f"{'=' * 70}")
                    
                    resp = await client.get("/sessions/trial-multi")
                    data = resp.json()
                    print(f"  Session ID: {data['session_id']}")
                    print(f"  Completed:  {data['completed']}")
                    print(f"  Total turns: {len(data['turns'])}")
                    
                    for i, turn in enumerate(data["turns"], 1):
                        print(f"\n  Turn {i}:")
                        print(f"    Request:  {turn['request_messages']}")
                        print(f"    Response: {turn['completion_text']}")
                        print(f"    Tokens:   {turn['completion_token_ids']}")
                        print(f"    Logprobs: {turn['completion_logprobs']}")
                        print(f"    Reason:   {turn['finish_reason']}")
                    
                    # Step 5: Complete and cleanup
                    print(f"\n{'=' * 70}")
                    print("  CLEANUP")
                    print(f"{'=' * 70}")
                    
                    resp = await client.post("/sessions/trial-multi/complete")
                    print(f"  Complete: {resp.json()}")
                    
                    resp = await client.delete("/sessions/trial-multi")
                    print(f"  Delete:   {resp.json()}")
                    
                    resp = await client.get("/sessions/trial-multi")
                    print(f"  Verify:   Status {resp.status_code} (expected 404)")
            
            asyncio.run(run_test())
            
            print(f"\n{'=' * 70}")
            print("  ENGINE REGISTRY STATUS")
            print(f"{'=' * 70}")
            engines = proxy.engine_registry.list_engines()
            for e in engines:
                print(f"  Engine: {e['engine_id']} | {e['url']} | total_requests: {e['total_requests']} | active: {e['active_requests']}")


if __name__ == "__main__":
    main()
