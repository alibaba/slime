"""Smoke test for in-process token capture via the OpenAI adapter.

Drives one ``/v1/chat/completions`` turn (session carried as a Bearer token,
exactly as ``generate_with_harbor`` wires it) with sglang's ``/generate``
monkeypatched, then asserts ``finish_session`` yields a self-consistent training
``Sample``.

Requires the full runtime stack (torch via ``slime.utils.types``, aiohttp); the
module is skipped where those are absent so minimal envs don't error.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from slime.agent.adapters import OpenAIAdapter  # noqa: E402
from slime.agent.adapters import common as adapter_common  # noqa: E402
from slime.agent.trajectory import TurnRecord  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

_OUT_IDS = [7, 8, 9]
_OUT_LOGPROBS = [-0.1, -0.2, -0.3]


class _FakeTokenizer:
    """Deterministic, torch-free stand-in for the served tokenizer."""

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True, **kwargs):
        n = len(messages) + (1 if add_generation_prompt else 0)
        return list(range(100, 100 + n))

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def test_openai_adapter_captures_single_turn(monkeypatch):
    async def _fake_generate(prompt_ids, session, body, *, adapter, session_id=None):
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=list(_OUT_IDS),
            finish_reason="stop",
            output_log_probs=list(_OUT_LOGPROBS),
        )

    # _run_turn resolves call_sglang_generate from the module globals at call
    # time, so patching the module attribute intercepts the sglang round-trip.
    monkeypatch.setattr(adapter_common, "call_sglang_generate", _fake_generate)

    adapter = OpenAIAdapter(tokenizer=_FakeTokenizer(), sglang_url="http://unused")
    sid = "sess-1"
    base = Sample(index=0, group_index=0, prompt=[{"role": "user", "content": "hi"}])

    async def _run():
        adapter.open_session(sid, sampling_defaults={}, max_context_tokens=0)
        async with TestClient(TestServer(adapter.app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {sid}"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 200, await resp.text()
        return await adapter.finish_session(sid, base_sample=base, reward=1.0)

    samples = asyncio.run(_run())

    assert len(samples) == 1  # linear conversation -> one training sample
    s = samples[0]
    assert s.response_length == len(_OUT_IDS)
    assert s.loss_mask is not None and len(s.loss_mask) == s.response_length
    assert sum(s.loss_mask) == len(_OUT_IDS)  # every generated token is trained
    assert s.rollout_log_probs is not None and len(s.rollout_log_probs) == s.response_length
    assert s.reward == 1.0
    assert s.status == Sample.Status.COMPLETED


def test_finish_session_idempotent_and_empty(monkeypatch):
    """A session with no turns yields no samples; a second finish is a no-op."""
    adapter = OpenAIAdapter(tokenizer=_FakeTokenizer(), sglang_url="http://unused")
    base = Sample(index=0, group_index=0, prompt=[{"role": "user", "content": "hi"}])

    async def _run():
        adapter.open_session("empty", sampling_defaults={}, max_context_tokens=0)
        first = await adapter.finish_session("empty", base_sample=base)
        second = await adapter.finish_session("empty", base_sample=base)
        return first, second

    first, second = asyncio.run(_run())
    assert first == []
    assert second == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
