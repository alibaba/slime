#!/usr/bin/env python3
"""Prepare one durable SWE-agent installation for all sandbox pods.

The target directory lives on the RWX dataset PVC. During preparation, the head
maps it to the same absolute path used inside sandboxes (``/opt/sweagent-shared``)
so venv scripts and the managed Python stay valid after the directory is mounted
there read-only.

Only this command writes the installation. Trial pods never install or mutate it,
so 64 concurrent trials have no first-install race.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

DEFAULT_TARGET = Path("/var/model-dataset/shared-agents/swe-agent-v1.1.0")
DEFAULT_MOUNT = Path("/opt/sweagent-shared")


def run(*args: str, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(args), flush=True)
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    return completed.stdout


def validated(root: Path, version: str) -> bool:
    try:
        manifest = json.loads((root / ".complete").read_text())
    except Exception:
        return False
    required = (
        root / "venv/bin/python",
        root / "venv/bin/pip",
        root / "repo/config/default.yaml",
        root / "configs/default.yaml",
    )
    return manifest.get("version") == version and all(path.exists() for path in required)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--mount-path", type=Path, default=DEFAULT_MOUNT)
    ap.add_argument("--version", default="v1.1.0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    target = args.target.resolve()
    mount = args.mount_path
    if validated(target, args.version) and not args.force:
        print(f"already prepared: {target}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    # Build directly at the sandbox's future absolute mount path, but on the
    # head's local overlay. uv creates thousands of small files and writes them
    # randomly; doing that on ossfs takes orders of magnitude longer. After the
    # build, one sequential tree copy persists it to the PVC.
    if mount.is_symlink():
        mount.unlink()
    elif mount.exists():
        shutil.rmtree(mount)
    mount.mkdir(parents=True)
    staging = mount

    env = os.environ.copy()
    env["UV_PYTHON_INSTALL_DIR"] = str(mount / "python")
    try:
        run("uv", "python", "install", "3.12", env=env)
        python = run(
            "uv", "python", "find", "--managed-python", "3.12", env=env
        ).strip().splitlines()[-1]
        run(
            "uv", "venv", "--seed", "--relocatable", "--python", python,
            str(mount / "venv"), env=env,
        )
        run(
            "git", "clone", "--depth", "1", "--branch", args.version,
            "https://github.com/SWE-agent/SWE-agent.git", str(mount / "repo"),
            env=env,
        )
        run(
            "uv", "pip", "install", "--python", str(mount / "venv/bin/python"),
            str(mount / "repo"), env=env,
        )

        site_packages = run(
            str(mount / "venv/bin/python"), "-c",
            "import site; print(site.getsitepackages()[0])",
            env=env,
        ).strip().splitlines()[-1]
        shutil.copytree(mount / "repo/config", Path(site_packages) / "config")
        shutil.copytree(mount / "repo/tools", Path(site_packages) / "tools")
        configs = mount / "configs"
        configs.mkdir()
        shutil.copy2(mount / "repo/config/default.yaml", configs / "default.yaml")
        backticks = mount / "repo/config/default_backticks.yaml"
        if backticks.exists():
            shutil.copy2(backticks, configs / "default_backticks.yaml")
        (Path(site_packages) / "trajectories").mkdir(exist_ok=True)

        version_output = run(
            str(mount / "venv/bin/pip"), "show", "sweagent", env=env
        )
        (mount / ".complete").write_text(json.dumps({
            "version": args.version,
            "created_at": time.time(),
            "pip_show": version_output,
        }, indent=2) + "\n")

        # Image builds set target == mount: the prepared tree is already in its
        # final layer, so no durability copy is needed.
        if target == mount.resolve():
            if not validated(target, args.version):
                raise RuntimeError("image installation failed validation")
            run(str(mount / "venv/bin/python"), "-c", "import sweagent; print(sweagent.__file__)")
            print(f"prepared {target}")
            return 0

        # A head-side preparation copies the complete local tree to a sibling
        # staging directory on the PVC, then atomically renames it into place.
        # Trial pods never observe a partially populated installation.
        pvc_staging = target.with_name(f".{target.name}.copy-{os.getpid()}")
        if pvc_staging.exists():
            shutil.rmtree(pvc_staging)
        print(f"copying prepared installation to {pvc_staging}", flush=True)
        shutil.copytree(staging, pvc_staging, symlinks=True)
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{int(time.time())}")
            target.rename(backup)
            print(f"previous installation moved to {backup}")
        pvc_staging.rename(target)
        if not validated(target, args.version):
            raise RuntimeError("shared installation failed post-move validation")
        run(str(mount / "venv/bin/python"), "-c", "import sweagent; print(sweagent.__file__)")
        print(f"prepared {target}")
        return 0
    except BaseException:
        raise


if __name__ == "__main__":
    raise SystemExit(main())
