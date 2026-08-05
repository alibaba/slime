"""In-process OpenAI adapter service for the Harbor remote-agent rollout.

This is the replacement for the old head-node ``TokenProxy`` Ray actor. It runs
an :class:`~slime.agent.adapters.OpenAIAdapter` (aiohttp) *inside* the single
``RolloutManager`` actor where ``generate_with_harbor`` executes, so every
concurrent trial of a rollout step shares one adapter endpoint and token capture
happens in-process (no REST / no post-hoc reconstruction).

Two endpoints are involved:

* **Endpoint A** — the adapter's own OpenAI-compatible HTTP server. The remote
  Harbor sandbox (outside the Ray cluster) points ``OPENAI_BASE_URL`` at it and
  carries the session id via ``OPENAI_API_KEY`` (Bearer). Bound to a fixed port
  on the head node (``--harbor-adapter-*``) so it has a stable, reachable
  address.
* **Endpoint B** — the sglang router. The adapter reaches it via
  ``args.sglang_router_ip/port``, which ``start_rollout_servers`` populates in
  this same process (``slime/ray/rollout.py``). No engine self-registration.
"""

from __future__ import annotations

import logging
import os

from slime.agent.adapters import OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def _resolve_public_host(args) -> str:
    """Host that the remote sandbox uses to reach the adapter (endpoint A).

    Precedence: ``--harbor-adapter-public-host`` -> ``LOCAL_IP`` env. Raises if
    neither is set, since a sandbox outside the cluster cannot dial back to
    ``0.0.0.0``.
    """
    host = getattr(args, "harbor_adapter_public_host", None) or os.getenv("LOCAL_IP")
    if not host or host == "0.0.0.0":
        raise RuntimeError(
            "Cannot determine a reachable adapter host. Set "
            "--harbor-adapter-public-host to the head-node address that the "
            "Harbor sandbox can reach (or export LOCAL_IP). Without it the "
            "remote agent cannot dial back to the adapter."
        )
    return host


class HarborAdapterService(metaclass=SingletonMeta):
    """Process-level singleton owning the OpenAIAdapter + its HTTP server.

    Constructed lazily on the first ``generate_with_harbor`` call. The adapter,
    its TrajectoryManager and the aiohttp server live for the process lifetime;
    per-trial state is created/destroyed via ``open_session``/``finish_session``/
    ``drop_session`` on ``self.adapter``.
    """

    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None

        # Endpoint B: sglang router, populated in this same process by
        # start_rollout_servers before any rollout runs.
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

        # Keep one linear conversation -> one training Sample: pick a fork
        # threshold large enough that a single trajectory is never split on
        # length. (The manager's own default of 1024 would fork long agent
        # responses.) Truncation to the context budget still happens via
        # max_sample_tokens in finish_session.
        fork_threshold = (
            max(
                int(getattr(args, "rollout_max_response_len", 0) or 0),
                self.max_context_len,
            )
            or 1_000_000
        )

        self.adapter = OpenAIAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=fork_threshold,
        )

        bind_host = getattr(args, "harbor_adapter_bind_host", "0.0.0.0")
        bind_port = int(getattr(args, "harbor_adapter_port", 0) or 0)
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the fire-and-forget /abort_request in the adapter;
        # otherwise a cancelled client leaves an inflight sglang /generate that
        # races the next release_memory_occupation and trips its idle assertion.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=bind_host,
            port=bind_port,
            thread_name="harbor-openai-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )

        public_host = _resolve_public_host(args)
        self.adapter_url = f"http://{public_host}:{self.app_handle.port}"

        self._wait_for_router(sglang_url)

        logger.info(
            "[Harbor] adapter service ready: adapter_url=%s sglang_url=%s "
            "max_context_len=%s fork_threshold=%s tool_parser=%s reasoning_parser=%s",
            self.adapter_url,
            sglang_url,
            self.max_context_len,
            fork_threshold,
            self.tool_parser,
            self.reasoning_parser,
        )

    def _wait_for_router(self, sglang_url: str, *, tries: int = 5, delay: float = 1.0) -> None:
        """Best-effort readiness probe for the sglang router (endpoint B).

        Runs once at startup (first generate call). By construction the router
        and workers are up before rollout, so this is defensive: it logs a
        warning rather than failing if the probe does not succeed.
        """
        import time

        import requests

        for attempt in range(tries):
            try:
                resp = requests.get(f"{sglang_url}/health", timeout=2)
                if resp.status_code < 400:
                    return
            except Exception:
                pass
            if attempt < tries - 1:
                time.sleep(delay)
        logger.warning(
            "[Harbor] sglang router health probe did not succeed at %s; "
            "proceeding anyway (first trials may fail if it is not ready).",
            sglang_url,
        )
