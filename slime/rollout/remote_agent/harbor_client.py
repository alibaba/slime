"""Harbor trial runner — run an agent task in-process via the Harbor Python API.

The agent (e.g. a SWE-agent) executes locally through ``harbor.trial.Trial``;
its LLM calls go through the in-process OpenAI adapter (see
``adapter_service.HarborAdapterService``) so tokens are captured for training.
No external Harbor server is required.

Usage::

    from slime.rollout.remote_agent.harbor_client import (
        HarborAgentConfig, run_local_trial,
    )

    result = await run_local_trial(
        task_path="/data/tasks/my-task",
        agent=HarborAgentConfig(name="swe-agent", llm_proxy_url="http://proxy/..."),
    )

Requires the ``harbor`` package (https://github.com/alibaba/harbor).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HarborAgentConfig:
    """Agent configuration to submit with a run request."""
    name: str | None = None
    import_path: str | None = None
    model_name: str | None = None
    llm_proxy_url: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarborVerifierConfig:
    """Verifier configuration to submit with a run request."""
    disable: bool = False


@dataclass
class HarborRunResult:
    """Result of a Harbor agent run."""
    run_id: str
    status: str  # completed | failed | timeout | error
    rewards: dict[str, float] | None = None
    error_message: str | None = None
    result_uri: str | None = None


# ---------------------------------------------------------------------------
# Local Trial — direct Python function call
# ---------------------------------------------------------------------------


async def run_local_trial(
    task_path: str,
    agent: HarborAgentConfig,
    verifier: HarborVerifierConfig | None = None,
    environment_overrides: dict[str, Any] | None = None,
    environment_kwargs: dict[str, Any] | None = None,
    timeout: float = 1800.0,
    timeout_multiplier: float = 1.0,
    environment_import_path: str = "harbor.environments.local_docker:LocalDockerEnvironment",
) -> HarborRunResult:
    """Run a Harbor Trial directly in the current Python process.

    This bypasses the Harbor HTTP server entirely.  The agent executes
    locally, but its LLM calls still go through the in-process OpenAI
    adapter so tokens are captured for training.

    Args:
        task_path: Path to the local task directory.
        agent: Agent configuration (name/import_path, model, proxy URL).
        verifier: Verifier configuration.
        environment_overrides: Env var overrides for the environment.
        environment_kwargs: Extra kwargs for the environment constructor.
        timeout: Maximum wall-clock seconds for a single run.
        timeout_multiplier: Multiplier for timeout values.
        environment_import_path: Import path for the environment class
            (e.g., "harbor.environments.local_docker:LocalDockerEnvironment").

    Returns:
        A ``HarborRunResult`` with run_id, status, and rewards.
    """
    import shortuuid

    try:
        from harbor.models.trial.config import (
            AgentConfig as TrialAgentConfig,
            EnvironmentConfig as TrialEnvironmentConfig,
            TaskConfig,
            TrialConfig,
            VerifierConfig as TrialVerifierConfig,
        )
        from harbor.trial.trial import Trial
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "run_local_trial requires the 'harbor' package, which drives the "
            "external agent. Install it with:\n"
            "    pip install git+https://github.com/alibaba/harbor.git"
        ) from e

    verifier = verifier or HarborVerifierConfig()
    run_id = f"local-{shortuuid.uuid()}"
    logger.info("[HarborTrial][%s] Starting local trial, task_path=%s", run_id, task_path)

    task_dir = Path(task_path).resolve()
    if not task_dir.is_dir():
        logger.error("[HarborTrial][%s] Task directory not found: %s", run_id, task_dir)
        return HarborRunResult(
            run_id=run_id,
            status="error",
            rewards={"reward": 0.0},
            error_message=f"Task directory not found: {task_dir}",
        )

    logger.info(
        "[HarborTrial][%s] Task directory resolved: %s, agent=%s, model=%s",
        run_id,
        task_path,
        agent.name or agent.import_path,
        agent.model_name,
    )

    try:
        # Build agent kwargs
        agent_kwargs = dict(agent.kwargs)
        if agent.llm_proxy_url:
            agent_kwargs["llm_proxy_url"] = agent.llm_proxy_url

        logger.info(
            "[HarborTrial][%s] Agent kwargs: llm_proxy_url=%s, model_name=%s, kwargs=%s, env=%s",
            run_id,
            agent.llm_proxy_url,
            agent.model_name,
            agent_kwargs,
            environment_overrides,
        )

        # Build TrialConfig using Harbor's native API
        trial_config = TrialConfig(
            trial_name=run_id,
            task=TaskConfig(path=task_dir),
            agent=TrialAgentConfig(
                name=agent.name,
                import_path=agent.import_path,
                model_name=agent.model_name,
                kwargs=agent_kwargs,
                env=environment_overrides or {},
            ),
            environment=TrialEnvironmentConfig(
                import_path=environment_import_path,
                env=environment_overrides or {},
                kwargs=environment_kwargs or {},
            ),
            verifier=TrialVerifierConfig(
                disable=verifier.disable,
            ),
        )

        logger.info(
            "[HarborTrial][%s] Creating Trial with config: env_import=%s",
            run_id,
            environment_import_path,
        )

        # Create and run the trial
        trial = await Trial.create(trial_config)
        logger.info("[HarborTrial][%s] Trial created, starting execution...", run_id)

        trial_result = await trial.run()
        logger.info(
            "[HarborTrial][%s] Trial execution completed, exception_info=%s",
            run_id,
            trial_result.exception_info,
        )

        # Extract results
        if trial_result.exception_info is None:
            rewards = {}
            if trial_result.verifier_result is not None:
                rewards = trial_result.verifier_result.rewards or {}
            if not rewards:
                # Try to read reward from file
                reward_file = trial._trial_paths.reward_text_path
                if reward_file.exists():
                    try:
                        rewards = {"reward": float(reward_file.read_text().strip())}
                    except ValueError:
                        rewards = {"reward": 0.0}
                else:
                    rewards = {"reward": 0.0}

            logger.info(
                "[HarborTrial][%s] Trial finished successfully: rewards=%s",
                run_id,
                rewards,
            )
            return HarborRunResult(
                run_id=run_id,
                status="completed",
                rewards=rewards,
                result_uri=str(task_dir),
            )

        # Trial had an exception
        exc = trial_result.exception_info
        logger.error(
            "[HarborTrial][%s] Trial failed with exception: %s: %s",
            run_id,
            exc.exception_type,
            exc.exception_message,
        )
        return HarborRunResult(
            run_id=run_id,
            status="error",
            rewards={"reward": 0.0},
            error_message=f"{exc.exception_type}: {exc.exception_message}",
        )

    except asyncio.TimeoutError:
        logger.error("[HarborTrial][%s] Trial timed out after %ds", run_id, timeout)
        return HarborRunResult(
            run_id=run_id,
            status="timeout",
            rewards={"reward": 0.0},
            error_message=f"Local trial timed out after {timeout}s",
        )

    except Exception as e:
        logger.exception("[HarborTrial][%s] Trial failed: %s", run_id, e)
        return HarborRunResult(
            run_id=run_id,
            status="error",
            rewards={"reward": 0.0},
            error_message=f"{type(e).__name__}: {e}",
        )
