"""custom-generate-function: execute Remote Agent via Harbor and capture token data.

Usage::

    python train_remote_agent.py \\
        --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \\
        --harbor-agent-name swe-agent \\
        --harbor-model-name openai/qwen-max \\
        --harbor-env-import-path harbor.environments.local_docker:LocalDockerEnvironment \\
        --harbor-task-path-template '/home/slime/dataset-tasks/{instance_id}' \\
        --harbor-adapter-public-host <head-node-ip> \\
        --harbor-adapter-port 18001 \\
        --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \\
        ...

This function replaces the default ``generate(args, sample, sampling_params)``
and is called once per sample by ``sglang_rollout.generate_rollout_async``.

Token capture is done by an in-process ``OpenAIAdapter`` (see
``adapter_service.HarborAdapterService``): the remote agent's OpenAI calls hit
the adapter, which renders each turn to token ids, calls sglang ``/generate``
and folds the result into a per-session ``TrajectoryManager``. ``finish_session``
then linearises the session into training ``Sample`` objects — no cross-process
REST round-trip and no post-hoc token reconstruction.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import Any

import shortuuid

from slime.rollout.remote_agent.adapter_service import HarborAdapterService
from slime.rollout.remote_agent.harbor_client import (
    HarborAgentConfig,
    HarborVerifierConfig,
    run_local_trial,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
) -> list[Sample]:
    """Custom generate function — submit one sample to Harbor, capture tokens.

    This is the entry point specified via ``--custom-generate-function-path``.
    It replaces the default SGLang HTTP generate for each sample.

    Returns a ``list[Sample]``: normally one training sample per trial (the
    adapter keeps a linear conversation as a single trajectory), an eval
    placeholder on the eval path, or a single dropped sample on failure.
    """
    return await _generate_with_harbor_async(args, sample, sampling_params, evaluation)


async def _generate_with_harbor_async(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool,
) -> list[Sample]:
    """Async core implementation."""
    service = HarborAdapterService(args)

    # 1. Resolve identifiers. The group layer assigns a unique session_id per
    #    sample; fall back to a readable one built from instance_id.
    instance_id = sample.metadata.get("instance_id") or sample.metadata.get("task_name")
    if instance_id is None:
        instance_id = str(sample.index)
    sid = sample.session_id or f"{instance_id}-{shortuuid.uuid()}"
    sample.session_id = sid
    logger.info(
        "[Harbor][%s] Starting generate, evaluation=%s, adapter_url=%s",
        sid,
        evaluation,
        service.adapter_url,
    )

    # 2. Open the adapter session that the agent's OpenAI calls fold into.
    service.adapter.open_session(
        sid,
        sampling_defaults=sampling_params,
        max_context_tokens=service.max_context_len,
    )

    try:
        # 3. Agent-facing endpoint (endpoint A). Session travels via the Bearer
        #    token (OPENAI_API_KEY = sid), resolved by the adapter's sid_from_bearer.
        agent_base_url = f"{service.adapter_url}/v1"

        # 4. Resolve task path.
        task_path = args.harbor_task_path_template.format(instance_id=instance_id)

        # 4b. Resolve the per-task SandboxSet name and merge it into the
        #     (otherwise global) environment kwargs so it reaches the Harbor
        #     environment for this trial.
        env_kwargs = dict(getattr(args, "harbor_env_kwargs", {}) or {})
        sandbox_set_name = _resolve_sandbox_set_name(args, sample, task_path)
        if sandbox_set_name:
            key = getattr(args, "harbor_sandbox_set_key", "sandbox_set_name")
            env_kwargs[key] = sandbox_set_name
            logger.info("[Harbor][%s] Routing to SandboxSet %s=%s", sid, key, sandbox_set_name)

        # 5. Build agent configuration.
        agent_kwargs = dict(args.harbor_agent_kwargs)
        agent_kwargs["api_base"] = agent_base_url
        agent_kwargs["session_id"] = sid
        agent_kwargs["temperature"] = sampling_params.get("temperature", 1.0)
        agent_kwargs["top_p"] = sampling_params.get("top_p", 1.0)

        agent_config = HarborAgentConfig(
            name=args.harbor_agent_name,
            import_path=args.harbor_agent_import_path,
            model_name=args.harbor_model_name,
            llm_proxy_url=agent_base_url,
            kwargs=agent_kwargs,
        )

        # Environment overrides — point the OpenAI SDK at the adapter and carry
        # the session id as the API key (Bearer). These MUST match the adapter
        # session, so set them explicitly rather than via setdefault.
        env_overrides = dict(args.harbor_env_overrides)
        env_overrides["OPENAI_API_KEY"] = sid
        env_overrides["OPENAI_BASE_URL"] = agent_base_url

        # 6. Run the agent trial in-process; its OpenAI calls fold into the adapter.
        logger.info("[Harbor][%s] Running local trial...", sid)
        result = await run_local_trial(
            task_path=task_path,
            agent=agent_config,
            verifier=HarborVerifierConfig(),
            environment_overrides=env_overrides,
            environment_kwargs=env_kwargs,
            timeout=args.harbor_timeout,
            environment_import_path=args.harbor_env_import_path,
        )
        reward = float(result.rewards.get("reward", 0.0)) if result.rewards else 0.0
        logger.info("[Harbor][%s] Trial completed status=%s reward=%.4f", sid, result.status, reward)

        trial_metadata = {
            "trial_id": sid,
            "harbor_run_id": result.run_id,
            "harbor_status": result.status,
            "instance_id": instance_id,
        }

        # 7. Eval path only needs the reward — skip token capture.
        if evaluation:
            return [_make_eval_sample(sample, reward=reward, metadata=trial_metadata)]

        # 8. Drain the captured trajectory into training samples.
        samples = await service.adapter.finish_session(
            sid,
            base_sample=sample,
            reward=reward,
            extra_metadata=trial_metadata,
        )
        if not samples:
            logger.warning("[Harbor][%s] No turns recorded — dropping sample.", sid)
            return [_make_aborted_sample(sample, reason="adapter_session_empty", metadata=trial_metadata)]

        logger.info("[Harbor][%s] Finished: status=%s reward=%.4f samples=%d", sid, result.status, reward, len(samples))
        return samples

    except Exception as e:
        logger.warning("[Harbor][%s] Rollout failed: %s: %s", sid, type(e).__name__, e)
        return [_make_aborted_sample(sample, reason=f"exception:{type(e).__name__}", metadata={"instance_id": instance_id})]
    finally:
        # Cleanup only, idempotent (also drops any leftover trajectory state).
        await service.adapter.drop_session(sid)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _make_aborted_sample(sample: Sample, *, reason: str, metadata: dict) -> Sample:
    """Mark ``sample`` dropped in place (2-token dummy avoids training narrow())."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), **metadata, "abort_reason": reason}
    return sample


def _make_eval_sample(sample: Sample, *, reward: float, metadata: dict) -> Sample:
    """Eval-path placeholder: only ``reward`` matters."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = reward
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {**(sample.metadata or {}), **metadata}
    return sample
