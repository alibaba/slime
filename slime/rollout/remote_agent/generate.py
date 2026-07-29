"""custom-generate-function: execute Remote Agent via Harbor and capture token data.

Usage::

    python train_remote_agent.py \\
        --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \\
        --harbor-server-url http://harbor:8080 \\
        --harbor-agent-name swe-agent \\
        --harbor-model-name openai/qwen-max \\
        --harbor-task-path-template '/home/slime/dataset-tasks/{instance_id}' \\
        --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \\
        ...

This function replaces the default ``generate(args, sample, sampling_params)``
and is called once per sample by ``sglang_rollout.generate_rollout_async``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from argparse import Namespace
from typing import Any
from urllib.parse import urlparse

import shortuuid

from slime.rollout.remote_agent.harbor_client import (
    HarborAgentConfig,
    HarborClient,
    HarborRunResult,
    HarborVerifierConfig,
    run_local_trial,
)
from slime.rollout.remote_agent.proxy import get_proxy_url
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Generate state (process-level singleton)
# ---------------------------------------------------------------------------


class _HarborGenerateState(metaclass=SingletonMeta):
    """Process-level singleton holding shared generate resources."""

    harbor_client: HarborClient | None = None
    proxy_url: str | None = None
    tokenizer = None

    def ensure_client(self, args: Namespace) -> HarborClient:
        """Lazily create HarborClient for remote mode."""
        if self.harbor_client is None:
            self.harbor_client = HarborClient(
                server_url=args.harbor_server_url,
                timeout=args.harbor_timeout,
            )
        return self.harbor_client

    def ensure_proxy_url(self) -> str:
        """Discover the running proxy URL via Ray named actor."""
        if self.proxy_url is None:
            url = get_proxy_url()
            if url is None:
                raise RuntimeError(
                    "Proxy server actor not found. "
                    "Make sure the training script starts the proxy before rollout. "
                    "See examples/remote_agent/ for a reference setup."
                )
            self.proxy_url = url
            logger.info("Discovered proxy URL: %s", url)
        return self.proxy_url


# ---------------------------------------------------------------------------
# SandboxSet routing
# ---------------------------------------------------------------------------


def _read_task_sandbox_class(task_path: str, class_key: str) -> str | None:
    """Read the pod-size class from a task's ``task.toml``.

    Looks up ``class_key`` under ``[metadata]``, ``[environment]``, ``[task]``
    and the top level, in that order. Returns None if the file is missing /
    unreadable or the key is absent.
    """
    toml_path = os.path.join(task_path, "task.toml")
    try:
        import tomllib

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:  # malformed toml, permission error, ...
        logger.warning("Failed to read sandbox class from %s: %s", toml_path, e)
        return None

    for section in ("metadata", "environment", "task"):
        sec = data.get(section)
        if isinstance(sec, dict) and sec.get(class_key):
            return str(sec[class_key])
    if data.get(class_key):
        return str(data[class_key])
    return None


def _resolve_sandbox_set_name(
    args: Namespace, sample: Sample, task_path: str
) -> str | None:
    """Resolve the target SandboxSet / pool name for this task.

    Precedence:
      1. an explicit ``sandbox_set_name`` in the sample metadata;
      2. the pod-size class (``--harbor-sandbox-class-key``) from the sample
         metadata, converted via ``--harbor-sandbox-set-name-template``;
      3. the same class read from the task's ``task.toml``, converted likewise.

    Returns None when nothing is found, so the environment falls back to its
    default pool.
    """
    md = sample.metadata or {}
    explicit = md.get("sandbox_set_name")
    if explicit:
        return str(explicit)

    class_key = getattr(args, "harbor_sandbox_class_key", "sandbox_class")
    sandbox_class = md.get(class_key) or _read_task_sandbox_class(task_path, class_key)
    if not sandbox_class:
        return None

    template = getattr(args, "harbor_sandbox_set_name_template", "{sandbox_class}")
    return template.format(sandbox_class=sandbox_class)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def generate_with_harbor(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Custom generate function — submit one sample to Harbor, rebuild Sample.

    This is the entry point specified via ``--custom-generate-function-path``.
    It replaces the default SGLang HTTP generate for each sample.

    Args:
        args: Parsed CLI arguments (includes --harbor-* params).
        sample: Current sample containing prompt and metadata.
        sampling_params: Sampling params (temperature, top_p, max_new_tokens).
        evaluation: Whether this is an evaluation rollout.

    Returns:
        The sample populated with tokens, response, logprobs, and mask.
    """
    state = _HarborGenerateState()
    return await _generate_with_harbor_async(args, sample, sampling_params, evaluation, state)


async def _generate_with_harbor_async(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool,
    state: _HarborGenerateState,
) -> Sample:
    """Async core implementation."""
    import httpx

    use_local = getattr(args, "harbor_use_local_trial", False)
    proxy_url = state.ensure_proxy_url()

    # 1. Generate trial_id
    # Support reading instance_id from metadata or directly from sample (for swe-bench format)
    instance_id = sample.metadata.get("instance_id") or sample.metadata.get("task_name")
    if instance_id is None:
        instance_id = str(sample.index)
    trial_id = f"{instance_id}-{shortuuid.uuid()}"
    logger.info("[Harbor][%s] Starting generate, use_local=%s, proxy_url=%s, prompt_len=%d", trial_id, use_local, proxy_url, len(sample.prompt) if sample.prompt else 0)

    # 2. Register session with proxy
    logger.debug("[Harbor][%s] Registering session with proxy", trial_id)
    async with httpx.AsyncClient() as http:
        await http.post(f"{proxy_url}/sessions/{trial_id}")

    try:
        # 3. Build agent_base_url that the agent can reach
        # Use harbor-proxy-host arg if set, otherwise fall back to LOCAL_IP env var
        proxy_host = args.harbor_proxy_host
        if proxy_host == "0.0.0.0":
            proxy_host = os.getenv("LOCAL_IP", "0.0.0.0")
        if proxy_host == "0.0.0.0":
            logger.warning(
                "[Harbor][%s] LOCAL_IP is not set and --harbor-proxy-host is default. "
                "The Harbor agent may not be able to reach the proxy.",
                trial_id,
            )

        parsed = urlparse(proxy_url)
        proxy_port = parsed.port
        agent_base_url = f"http://{proxy_host}:{proxy_port}/{trial_id}/v1"
        logger.info("[Harbor][%s] Agent base_url=%s (proxy_host=%s)", trial_id, agent_base_url, proxy_host)

        # 4. Resolve task path
        task_path = args.harbor_task_path_template.format(instance_id=instance_id)
        logger.info("[Harbor][%s] Resolved task_path=%s", trial_id, task_path)

        # 4b. Resolve the per-task SandboxSet name from the task's pod-size class
        # and merge it into the (otherwise global) environment kwargs so it
        # reaches the Harbor environment / e2b SDK for this trial.
        env_kwargs = dict(getattr(args, "harbor_env_kwargs", {}) or {})
        sandbox_set_name = _resolve_sandbox_set_name(args, sample, task_path)
        if sandbox_set_name:
            key = getattr(args, "harbor_sandbox_set_key", "sandbox_set_name")
            env_kwargs[key] = sandbox_set_name
            logger.info(
                "[Harbor][%s] Routing to SandboxSet %s=%s", trial_id, key, sandbox_set_name
            )

        # 5. Build agent configuration
        agent_kwargs = dict(args.harbor_agent_kwargs)
        agent_kwargs["api_base"] = agent_base_url
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
        logger.debug("[Harbor][%s] Built agent_config: name=%s, model=%s", trial_id, agent_config.name, agent_config.model_name)

        # Environment overrides — inject OpenAI SDK vars pointing to our proxy
        env_overrides = dict(args.harbor_env_overrides)
        env_overrides.setdefault("OPENAI_API_KEY", "slime-proxy")
        env_overrides.setdefault("OPENAI_BASE_URL", agent_base_url)

        # 6. Submit to agent (remote or local)
        logger.info("[Harbor][%s] Submitting trial (%s mode)...", trial_id, "local" if use_local else "remote")
        if use_local:
            result = await run_local_trial(
                task_path=task_path,
                agent=agent_config,
                verifier=HarborVerifierConfig(),
                environment_overrides=env_overrides,
                environment_kwargs=env_kwargs,
                timeout=args.harbor_timeout,
                environment_import_path=args.harbor_env_import_path,
            )
        else:
            client = state.ensure_client(args)
            result = await _submit_with_retry(
                args=args,
                client=client,
                trial_id=trial_id,
                task_path=task_path,
                agent_config=agent_config,
                env_overrides=env_overrides,
                environment_kwargs=env_kwargs,
                proxy_url=proxy_url,
            )
        logger.info("[Harbor][%s] Trial completed with status=%s, reward=%s", trial_id, result.status, result.rewards)

        # 7. Mark session complete
        logger.debug("[Harbor][%s] Marking session as complete", trial_id)
        async with httpx.AsyncClient() as http:
            await http.post(f"{proxy_url}/sessions/{trial_id}/complete")

        # 8. Fetch session recording
        logger.debug("[Harbor][%s] Fetching session recording", trial_id)
        session_data = None
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{proxy_url}/sessions/{trial_id}")
            if resp.status_code == 200:
                session_data = resp.json()
                num_turns = len(session_data.get("turns", []))
                logger.debug("[Harbor][%s] Session has %d turns", trial_id, num_turns)

        # 9. Reconstruct Sample fields from session data
        if (
            session_data
            and session_data.get("turns")
            and not getattr(args, "harbor_disable_reconstruct", False)
        ):
            _reconstruct_output(sample, session_data, args)
            logger.info(
                "[Harbor][%s] Reconstructed sample: response_len=%d, num_tokens=%d",
                trial_id,
                sample.response_length,
                len(sample.tokens) if sample.tokens else 0,
            )
        else:
            logger.warning(
                "[Harbor][%s] Session has no recorded turns — token reconstruction skipped.",
                trial_id,
            )
            # Even on failure, we must set prompt tokens to avoid training crash
            # (total_length = 0 causes RuntimeError: narrow() length must be non-negative)
            _init_sample_with_prompt_only(sample, args)
            sample.response = ""
            sample.response_length = 0
            sample.status = Sample.Status.FAILED

        # 10. Fill reward from Harbor result
        if result.rewards:
            sample.reward = result.rewards.get("reward", 0.0)

        # 11. Record metadata
        sample.metadata["trial_id"] = trial_id
        sample.metadata["harbor_run_id"] = result.run_id
        sample.metadata["harbor_status"] = result.status
        sample.metadata["harbor_mode"] = "local" if use_local else "remote"
        logger.info("[Harbor][%s] Trial finished: status=%s, reward=%.4f", trial_id, sample.status, sample.reward or 0.0)

    finally:
        # 12. Clean up session
        logger.debug("[Harbor][%s] Cleaning up session", trial_id)
        async with httpx.AsyncClient() as http:
            try:
                await http.delete(f"{proxy_url}/sessions/{trial_id}")
            except Exception:
                pass

    return sample


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


async def _submit_with_retry(
    args: Namespace,
    client: HarborClient,
    trial_id: str,
    task_path: str,
    agent_config: HarborAgentConfig,
    env_overrides: dict[str, Any],
    proxy_url: str,
    environment_kwargs: dict[str, Any] | None = None,
) -> HarborRunResult:
    """Submit to remote Harbor with exponential backoff retry.

    Only used in remote mode (not local trial).  Local mode does not
    retry since the agent runs synchronously in the current process.
    """
    import httpx

    max_retries = getattr(args, "harbor_max_retries", 3)
    base_delay = getattr(args, "harbor_retry_base_delay", 2.0)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Retrying trial %s (attempt %d/%d) after %.1fs delay",
                trial_id,
                attempt + 1,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)

            # Reset session for clean retry
            async with httpx.AsyncClient() as http:
                try:
                    await http.delete(f"{proxy_url}/sessions/{trial_id}")
                except Exception:
                    pass
                await http.post(f"{proxy_url}/sessions/{trial_id}")

        try:
            result = await client.submit_async(
                task_path=task_path,
                agent=agent_config,
                verifier=HarborVerifierConfig(),
                environment_overrides=env_overrides,
                environment_kwargs=(
                    environment_kwargs
                    if environment_kwargs is not None
                    else dict(getattr(args, "harbor_env_kwargs", {}) or {})
                ),
                job_id=getattr(args, "wandb_group", None) or "slime-rl",
                task_id=trial_id,
            )

            if result.status == "completed":
                return result

            last_error = RuntimeError(
                f"Harbor run {result.status}: {result.error_message or 'unknown'}"
            )
            logger.warning(
                "Trial %s attempt %d/%d %s: %s",
                trial_id,
                attempt + 1,
                max_retries,
                result.status,
                result.error_message,
            )

        except Exception as e:
            last_error = e
            logger.warning(
                "Trial %s attempt %d/%d raised %s: %s",
                trial_id,
                attempt + 1,
                max_retries,
                type(e).__name__,
                e,
            )

    return HarborRunResult(
        run_id="",
        status="error",
        rewards={"reward": 0.0},
        error_message=f"All {max_retries} retries exhausted: {last_error}",
    )


# ---------------------------------------------------------------------------
# Output reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_output(
    sample: Sample, session_data: dict, args: Namespace
) -> None:
    """Rebuild Sample tokens / logprobs / mask from proxy session data.

    Traverses all recorded turns and:
    - LLM-generated tokens → mask=1 (participate in loss)
    - Tool/user replies → mask=0 (masked out of loss)

    Inspired by ``RemoteAgentLoop._reconstruct_output`` in Verl.
    """
    turns = session_data.get("turns", [])
    if not turns:
        sample.status = Sample.Status.FAILED
        return

    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    num_turns = 0

    # Lazy tokenizer init
    state = _HarborGenerateState()
    if state.tokenizer is None:
        from transformers import AutoTokenizer

        ckpt = getattr(args, "hf_checkpoint", None)
        if ckpt is None:
            raise ValueError(
                "--hf-checkpoint is required for token reconstruction. "
                "Pass the HuggingFace model path used by the rollout engines."
            )
        state.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True
        )
    tokenizer = state.tokenizer

    for i, turn in enumerate(turns):
        # LLM generation → mask=1
        completion_ids = turn.get("completion_token_ids", [])
        completion_logprobs = turn.get("completion_logprobs", [])

        response_ids.extend(completion_ids)
        response_mask.extend([1] * len(completion_ids))
        response_logprobs.extend(completion_logprobs)
        num_turns += 1

        # Multi-turn: new tool/user messages in next turn → mask=0
        if i + 1 < len(turns):
            next_turn = turns[i + 1]
            prev_count = len(turn.get("request_messages", []))
            next_count = len(next_turn.get("request_messages", []))

            if next_count > prev_count:
                new_messages = next_turn["request_messages"][prev_count:]
                tool_messages = [
                    m
                    for m in new_messages
                    if m.get("role") in ("tool", "user", "system")
                ]
                if tool_messages:
                    tool_ids = _tokenize_messages(tool_messages, tokenizer)
                    response_ids.extend(tool_ids)
                    response_mask.extend([0] * len(tool_ids))
                    response_logprobs.extend([0.0] * len(tool_ids))
                    num_turns += len(tool_messages)

    # Truncate to max response length
    max_len = getattr(args, "rollout_max_response_len", 4096) or 4096
    response_ids = response_ids[:max_len]
    response_mask = response_mask[:max_len]
    response_logprobs = response_logprobs[:max_len]

    # Tokenize prompt to get prompt_ids
    prompt_ids = _tokenize_prompt(sample, tokenizer)

    # Populate Sample: tokens = prompt_ids + response_ids (MUST include prompt tokens)
    sample.tokens = prompt_ids + response_ids
    sample.response_length = len(response_ids)  # Only response length, not total
    sample.rollout_log_probs = response_logprobs if response_logprobs else None
    sample.response = "".join(
        turn.get("completion_text", "") for turn in turns
    )
    sample.status = Sample.Status.COMPLETED
    sample.loss_mask = response_mask
    sample.metadata["num_agent_turns"] = num_turns
    sample.metadata["prompt_length"] = len(prompt_ids)


def _as_token_ids(tokenized) -> list[int]:
    """Coerce ``apply_chat_template`` output to a flat ``list[int]``.

    Newer transformers may return a ``BatchEncoding`` (dict-like) or a tensor instead of a
    plain list. Iterating a ``BatchEncoding`` yields its string keys ("input_ids", ...),
    which would corrupt the token stream, so normalize here.
    """
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    return tokenized


def _tokenize_messages(
    messages: list[dict], tokenizer
) -> list[int]:
    """Tokenize a list of chat messages into token IDs."""
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

    return _as_token_ids(
        tokenizer.apply_chat_template(
            normalized, add_generation_prompt=False, tokenize=True
        )
    )


def _tokenize_prompt(sample: Sample, tokenizer) -> list[int]:
    """Tokenize the prompt part of a sample.

    Args:
        sample: The sample containing prompt (str or list of messages).
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of token IDs for the prompt.
    """
    prompt = sample.prompt
    if isinstance(prompt, str):
        # Plain text prompt
        return tokenizer.encode(prompt, add_special_tokens=False)
    elif isinstance(prompt, list):
        # Chat format (list of messages)
        return _as_token_ids(
            tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=True
            )
        )
    return []


def _init_sample_with_prompt_only(sample: Sample, args: Namespace) -> None:
    """Initialize sample.tokens with prompt tokens when there's no response.

    This is called when the Harbor trial fails or has no recorded turns.
    Without prompt tokens, total_length would be 0, causing training to crash
    with: RuntimeError: narrow(): length must be non-negative.

    Args:
        sample: The sample to initialize.
        args: CLI args containing hf_checkpoint for tokenizer.
    """
    state = _HarborGenerateState()
    if state.tokenizer is None:
        from transformers import AutoTokenizer

        ckpt = getattr(args, "hf_checkpoint", None)
        if ckpt is None:
            logger.warning(
                "--hf-checkpoint not set, cannot tokenize prompt for failed sample"
            )
            sample.tokens = []
            return
        state.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True
        )

    sample.tokens = _tokenize_prompt(sample, state.tokenizer)
    sample.metadata["prompt_length"] = len(sample.tokens)
