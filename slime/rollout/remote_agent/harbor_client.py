"""Harbor SDK client — submit agent tasks to a remote Harbor server or run locally.

Remote Harbor server API::

    POST /api/v1/runs            Submit task (multipart/form-data)
    GET  /api/v1/runs/{run_id}   Query status

Local trial mode::

    Directly instantiates ``harbor.trial.trial.Trial`` and runs it in-process
    via ``run_local_trial()``.  No Harbor server required.

Usage (remote)::

    from slime.rollout.remote_agent.harbor_client import (
        HarborAgentConfig, HarborClient, HarborVerifierConfig,
    )

    client = HarborClient(server_url="http://harbor:8080")
    result = await client.submit_async(
        task_path="/data/tasks/my-task",
        agent=HarborAgentConfig(name="swe-agent", llm_proxy_url="http://proxy/..."),
    )

Usage (local)::

    from slime.rollout.remote_agent.harbor_client import (
        HarborAgentConfig, run_local_trial,
    )

    result = await run_local_trial(
        task_path="/data/tasks/my-task",
        agent=HarborAgentConfig(name="swe-agent", llm_proxy_url="http://proxy/..."),
    )
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

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
# Helpers
# ---------------------------------------------------------------------------


def _create_task_archive(task_path: str) -> tuple[bytes, str]:
    """Create a gzipped tar archive of the local task directory.

    Returns:
        (archive_bytes, original_dir_name)
    """
    task_dir = Path(task_path).resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Archive the directory under its own name so the server
        # can find it as <tmp>/<dir_name>/
        tar.add(str(task_dir), arcname=task_dir.name)
    buf.seek(0)
    return buf.getvalue(), task_dir.name


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class HarborClient:
    """Async client for submitting agent runs to the Harbor Agent Run server.

    The API is designed to feel similar to locally instantiating a Harbor
    Trial, but the actual execution happens on the remote server.
    The local task directory is automatically archived and uploaded to
    the server with each request.
    """

    def __init__(self, server_url: str, timeout: float = 1800.0):
        """Initialize the client.

        Args:
            server_url: Base URL of the Harbor Agent Run server
                (e.g. ``http://localhost:8080``).
            timeout: HTTP request timeout in seconds. Since agent runs can
                take a long time, the default is 30 minutes.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    async def submit_async(
        self,
        task_path: str,
        agent: HarborAgentConfig,
        verifier: HarborVerifierConfig | None = None,
        environment_overrides: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        timeout_multiplier: float = 1.0,
        job_id: str = "slime-rl",
        task_id: str | None = None,
    ) -> HarborRunResult:
        """Submit an agent run and block until the result is available.

        The local ``task_path`` directory is archived and uploaded to the
        server automatically.

        Args:
            task_path: Path to the local task directory.
            agent: Agent configuration (name or import_path, model, proxy URL).
            verifier: Verifier configuration.
            environment_overrides: Env var overrides for the execution environment.
            environment_kwargs: Extra kwargs for the environment constructor.
            timeout_multiplier: Multiplier for all timeout values.

        Returns:
            A ``HarborRunResult`` containing run_id, status, and rewards.
        """
        verifier = verifier or HarborVerifierConfig()

        # Archive the local task directory first so we can send the tar-relative
        # directory name as task_path. The server unpacks the archive and resolves
        # task_path against its own workdir, so an absolute client-side path would
        # not exist there.
        try:
            archive_bytes, dir_name = await asyncio.to_thread(
                _create_task_archive, task_path
            )
        except FileNotFoundError as e:
            return HarborRunResult(
                run_id="",
                status="error",
                rewards={"reward": 0.0},
                error_message=str(e),
            )

        # Build agent dict (strip empty values)
        agent_dict = {}
        if agent.name is not None:
            agent_dict["name"] = agent.name
        if agent.import_path is not None:
            agent_dict["import_path"] = agent.import_path
        if agent.model_name is not None:
            agent_dict["model_name"] = agent.model_name
        if agent.llm_proxy_url is not None:
            agent_dict["llm_proxy_url"] = agent.llm_proxy_url
        if agent.kwargs:
            agent_dict["kwargs"] = agent.kwargs

        # job_id/task_id are required by the server's AgentRunRequest.
        payload = {
            "job_id": job_id,
            "task_id": task_id or dir_name,
            "task_path": dir_name,
            "agent": agent_dict,
            "timeout_multiplier": timeout_multiplier,
            "verifier": {"disable": verifier.disable},
        }
        if environment_overrides:
            payload["environment_overrides"] = environment_overrides
        if environment_kwargs:
            payload["environment_kwargs"] = environment_kwargs

        url = f"{self.server_url}/api/v1/runs"
        logger.info("Submitting Harbor run to %s task=%s", url, task_path)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    data={"metadata": json.dumps(payload)},
                    files={
                        "task_archive": (
                            "task.tar.gz",
                            archive_bytes,
                            "application/gzip",
                        )
                    },
                )
        except Exception as e:
            return HarborRunResult(
                run_id="",
                status="error",
                rewards={"reward": 0.0},
                error_message=str(e),
            )

        if response.status_code != 200:
            return HarborRunResult(
                run_id="",
                status="error",
                rewards={"reward": 0.0},
                error_message=(
                    f"Server returned HTTP {response.status_code}: "
                    f"{response.text}"
                ),
            )

        data = response.json()
        return HarborRunResult(
            run_id=data.get("run_id", ""),
            status=data.get("status", "error"),
            rewards=data.get("rewards"),
            error_message=data.get("error"),
            result_uri=data.get("result_uri"),
        )


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
    locally, but its LLM calls still go through the TokenProxy (if one
    is running) so tokens are captured for training.

    Args:
        task_path: Path to the local task directory.
        agent: Agent configuration (name/import_path, model, proxy URL).
        verifier: Verifier configuration.
        environment_overrides: Env var overrides for the environment.
        environment_kwargs: Extra kwargs for the environment constructor.
        timeout: Maximum wall-clock seconds for a single run.
        timeout_multiplier: Multiplier for timeout values.
        environment_import_path: Import path for the environment class
            (e.g., "harbor.environments.ack:ACKEnvironment").

    Returns:
        A ``HarborRunResult`` with run_id, status, and rewards.
    """
    import shortuuid

    from harbor.models.trial.config import (
        AgentConfig as TrialAgentConfig,
        EnvironmentConfig as TrialEnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig as TrialVerifierConfig,
    )
    from harbor.trial.trial import Trial

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
