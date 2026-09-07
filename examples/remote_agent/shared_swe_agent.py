"""Harbor SWE-agent adapter backed by one read-only shared installation.

The stock Harbor ``SweAgent.install`` clones the repository and builds a Python
venv in every sandbox. At 64-way SWE-bench concurrency that repeats identical
network and filesystem work 64 times. This subclass expects a prepared directory
mounted at the same absolute path in every sandbox and only creates the local
compatibility symlinks/wrapper expected by Harbor's existing SweAgent.run.

Prepare the directory once with ``prepare_shared_swe_agent.py``. The mount is
read-only during trials, so there is no concurrent first-install race.
"""

from __future__ import annotations

import shlex

from harbor.agents.installed.swe_agent import SweAgent
from harbor.environments.base import BaseEnvironment


class SharedSweAgent(SweAgent):
    """Run stock Harbor SWE-agent from a prebuilt shared directory."""

    def __init__(self, *args, shared_install_dir: str = "/opt/sweagent-shared", **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_install_dir = shared_install_dir.rstrip("/")
        if not self.shared_install_dir.startswith("/"):
            raise ValueError("shared_install_dir must be an absolute path")

    async def install(self, environment: BaseEnvironment) -> None:
        # These are image-level dependencies, not part of the shared Python venv.
        # Prewarmed SandboxSets install them before becoming Ready; this check is
        # therefore normally a no-op and remains a safe fallback for cold pods.
        await self.ensure_system_dependencies(
            environment, ("curl", "build_tools", "git", "tmux")
        )

        root = shlex.quote(self.shared_install_dir)
        command = f"""set -euo pipefail
ROOT={root}
for path in \
  "$ROOT/.complete" \
  "$ROOT/venv/bin/python" \
  "$ROOT/repo/config/default.yaml" \
  "$ROOT/configs/default.yaml"; do
  if [ ! -e "$path" ]; then
    echo "shared SWE-agent installation is incomplete: $path" >&2
    exit 1
  fi
done
rm -rf /opt/sweagent-venv /opt/sweagent-repo /opt/sweagent-configs
ln -s "$ROOT/venv" /opt/sweagent-venv
ln -s "$ROOT/repo" /opt/sweagent-repo
ln -s "$ROOT/configs" /opt/sweagent-configs
cat > /usr/local/bin/sweagent <<'WRAPPER'
#!/bin/bash
source /opt/sweagent-venv/bin/activate
exec python -m sweagent.run.run "$@"
WRAPPER
chmod +x /usr/local/bin/sweagent
cat > /etc/profile.d/testbed-conda.sh <<'PROFILE'
if [ -z "${{CONDA_DEFAULT_ENV:-}}" ] && [ -d /opt/miniconda3/envs/testbed ]; then
    if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
        . /opt/miniconda3/etc/profile.d/conda.sh
        conda activate testbed 2>/dev/null || true
    fi
fi
PROFILE
grep -qF '/etc/profile.d/testbed-conda.sh' /root/.bashrc 2>/dev/null || \
  echo '. /etc/profile.d/testbed-conda.sh' >> /root/.bashrc
"""
        result = await environment.exec(command=command, user="root")
        if result.return_code != 0:
            raise RuntimeError(
                "Failed to activate shared SWE-agent installation: "
                f"{result.stderr or result.stdout}"
            )

    def get_version_command(self) -> str | None:
        return f"{shlex.quote(self.shared_install_dir)}/venv/bin/pip show sweagent | grep ^Version:"
