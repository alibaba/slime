#!/usr/bin/env python3
"""Retarget terminal-bench task.toml files at the in-region registry and widen phase budgets.

Why this exists
---------------
The dataset ships ``docker_image`` pointing at the cn-hongkong ACR. Pulling that
from the me-east-1 cluster does not survive concurrency: a few hundred
simultaneous pulls fail with TLS handshake timeouts against the Hong Kong
endpoint. The identical images (same manifest digest) are in the me-east-1
instance, so only the registry host changes.

The phase budgets come from the task itself, not from slime's ``--harbor-timeout``:
harbor reads ``[agent].timeout_sec`` and ``[verifier].timeout_sec``. Raising only
the agent budget leaves the verifier to be killed at its own (short) limit and
the trial's reward is discarded, so both are set here.

Usage (in the ray head pod, on the local dataset copy)::

    python examples/remote_agent/retarget_tasks.py /root/terminal-bench \
        --agent-timeout 3600 --verifier-timeout 1800

Idempotent: rewriting an already-retargeted tree is a no-op. The first run keeps
a ``task.toml.orig`` next to each file it touches.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SRC_HOST = "yueming-acr-registry.cn-hongkong.cr.aliyuncs.com"
DST_HOST = "yueming-acr-me-registry.me-east-1.cr.aliyuncs.com"


def retarget(task_toml: Path, dst_host: str, agent: float | None, verifier: float | None) -> list[str]:
    """Rewrite one task.toml in place; return the list of changes made."""
    text = original = task_toml.read_text()
    changes = []

    new_text, n = re.subn(rf'(docker_image\s*=\s*")[^"]*{re.escape(SRC_HOST)}', rf"\g<1>{dst_host}", text)
    if n:
        changes.append("registry")
        text = new_text

    # Phase budgets live in their own tables, so anchor on the table header to
    # avoid rewriting the other one (both keys are literally `timeout_sec`).
    for table, value in (("agent", agent), ("verifier", verifier)):
        if value is None:
            continue
        new_text, n = re.subn(rf"(\[{table}\]\s*\ntimeout_sec\s*=\s*)[0-9.]+", rf"\g<1>{value:.1f}", text)
        if n and new_text != text:
            changes.append(f"{table}_timeout")
            text = new_text

    if text != original:
        backup = task_toml.with_suffix(".toml.orig")
        if not backup.exists():
            shutil.copy2(task_toml, backup)
        task_toml.write_text(text)
    return changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_root", help="directory holding one subdirectory per task")
    ap.add_argument("--dst-host", default=DST_HOST, help="registry host to point docker_image at")
    ap.add_argument("--agent-timeout", type=float, default=None, help="[agent].timeout_sec seconds")
    ap.add_argument("--verifier-timeout", type=float, default=None, help="[verifier].timeout_sec seconds")
    ap.add_argument("--only", nargs="*", help="restrict to these task names")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    tomls = sorted(root.glob("*/task.toml"))
    if args.only:
        wanted = set(args.only)
        tomls = [t for t in tomls if t.parent.name in wanted]
    if not tomls:
        sys.exit(f"no task.toml found under {root}")

    touched = 0
    tally: dict[str, int] = {}
    for toml in tomls:
        changes = retarget(toml, args.dst_host, args.agent_timeout, args.verifier_timeout)
        if changes:
            touched += 1
            for c in changes:
                tally[c] = tally.get(c, 0) + 1

    print(f"scanned {len(tomls)} task.toml, changed {touched}")
    for key, count in sorted(tally.items()):
        print(f"  {key}: {count}")

    stale = [t.parent.name for t in tomls if SRC_HOST in t.read_text()]
    if stale:
        print(f"WARNING: {len(stale)} still reference {SRC_HOST}: {stale[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
