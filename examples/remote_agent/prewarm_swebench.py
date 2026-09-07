#!/usr/bin/env python3
"""Prewarm SWE-bench SandboxSets in controlled batches.

Each selected task has a different image, so it gets its own one-replica
SandboxSet. Images are pulled four at a time to avoid overwhelming the ACR public
endpoint. The main-container startup installs only OS tools; SWE-agent itself is
prepared once in the workspace image, then an init container copies it to a
pod-local emptyDir mounted read-only in the task container.

No SSH transport is involved. The training launcher reaches these sandboxes via
Harbor's persistent pods/exec connection.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import socket
import tomllib
from pathlib import Path

from kubernetes import client, config
from kubernetes.client.rest import ApiException


def quantity(task: dict, key: str, default: str | int) -> str:
    value = str(task.get("environment", {}).get(key, default))
    return value.replace("G", "Gi") if key in ("memory", "storage") else value


def kubernetes_name(value: str, max_length: int = 58) -> str:
    """Match Harbor ACKEnvironment's SandboxSet naming exactly."""
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "harbor"
    if len(sanitized) <= max_length:
        return sanitized
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{sanitized[: max_length - len(digest) - 1].rstrip('-')}-{digest}"


class Prewarmer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api = client.CustomObjectsApi()

    def body(self, task_name: str) -> dict:
        task = tomllib.loads((self.args.dataset / task_name / "task.toml").read_text())
        env_name = task_name.replace("__", "-").replace("/", "-")
        name = kubernetes_name(f"{self.args.prefix}-{env_name}")
        resources = {"requests": {
            "cpu": quantity(task, "cpus", 1),
            "memory": quantity(task, "memory", "4G"),
            "ephemeral-storage": quantity(task, "storage", "10G"),
        }}
        labels = {
            "app": "sandbox",
            "environment": env_name,
            "alibabacloud.com/acs": "true",
            "slime.prewarm/run": self.args.prefix,
        }
        return {
            "apiVersion": "agents.kruise.io/v1alpha1",
            "kind": "SandboxSet",
            "metadata": {
                "name": name,
                "namespace": self.args.namespace,
                "labels": {**labels, "app": "kube-rl"},
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "imagePullSecrets": [{"name": self.args.image_pull_secret}],
                        "volumes": [{"name": "sweagent-shared", "emptyDir": {}}],
                        "initContainers": [{
                            "name": "copy-sweagent",
                            "image": self.args.workspace_image,
                            "command": [
                                "/bin/sh", "-c",
                                "cp -a /opt/sweagent-shared/. /shared/",
                            ],
                            "volumeMounts": [{
                                "name": "sweagent-shared", "mountPath": "/shared",
                            }],
                        }],
                        "containers": [{
                            "name": "main",
                            "image": task["environment"]["docker_image"],
                            "command": [
                                "/bin/sh", "-c",
                                "DEBIAN_FRONTEND=noninteractive apt-get update && "
                                "DEBIAN_FRONTEND=noninteractive apt-get install -y "
                                "curl git build-essential tmux && exec sleep infinity",
                            ],
                            "volumeMounts": [{
                                "name": "sweagent-shared",
                                "mountPath": "/opt/sweagent-shared",
                                "readOnly": True,
                            }],
                            "securityContext": {"privileged": False, "runAsUser": 0},
                            "resources": resources,
                        }],
                    },
                },
            },
        }

    async def call(self, method: str, plural: str, *args, **kwargs):
        fn = getattr(self.api, f"{method}_namespaced_custom_object")
        return await asyncio.to_thread(
            fn,
            "agents.kruise.io", "v1alpha1", self.args.namespace, plural,
            *args, **kwargs,
        )

    async def ensure_ready(self, task_name: str) -> str:
        spec = self.body(task_name)
        name = spec["metadata"]["name"]
        old_sandboxes: set[str] = set()
        try:
            await self.call("create", "sandboxsets", spec)
            print(f"created {name}", flush=True)
        except ApiException as exc:
            if exc.status != 409:
                raise
            current = await self.call(
                "list", "sandboxes",
                label_selector=f"agents.kruise.io/sandbox-pool={name}",
            )
            old_sandboxes = {item["metadata"]["name"] for item in current.get("items", [])}
            await self.call("patch", "sandboxsets", name, {"spec": spec["spec"]})
            for sandbox_name in old_sandboxes:
                try:
                    await self.call("delete", "sandboxes", sandbox_name)
                except ApiException as delete_exc:
                    if delete_exc.status != 404:
                        raise
            print(f"patched {name}; replacing {len(old_sandboxes)} sandbox(es)", flush=True)

        deadline = asyncio.get_running_loop().time() + self.args.ready_timeout
        last = None
        while asyncio.get_running_loop().time() < deadline:
            obj = await self.call("get", "sandboxsets", name)
            status = obj.get("status") or {}
            state = (int(status.get("availableReplicas", 0)), int(status.get("replicas", 0)))
            if state != last:
                print(f"  {name}: available={state[0]} replicas={state[1]}", flush=True)
                last = state
            if state[0] >= 1:
                if not old_sandboxes:
                    return name
                current = await self.call(
                    "list", "sandboxes",
                    label_selector=f"agents.kruise.io/sandbox-pool={name}",
                )
                current_names = {item["metadata"]["name"] for item in current.get("items", [])}
                if current_names - old_sandboxes:
                    return name
            await asyncio.sleep(15)
        raise TimeoutError(f"{name} did not become available; last={last}")

    async def run(self) -> None:
        tasks = [
            json.loads(line)["task_name"]
            for line in self.args.prompts.read_text().splitlines()
            if line
        ]
        # Seed the ACS image cache with one task from each project first; later
        # images from the same project share most of their layers.
        by_project: dict[str, list[str]] = {}
        for task in tasks:
            by_project.setdefault(task.split("__", 1)[0], []).append(task)
        ordered = [values.pop(0) for _, values in sorted(by_project.items())]
        ordered += [task for _, values in sorted(by_project.items()) for task in values]

        for start in range(0, len(ordered), self.args.max_in_flight):
            batch = ordered[start : start + self.args.max_in_flight]
            await asyncio.gather(*(self.ensure_ready(task) for task in batch))
            print(f"READY {start + len(batch)}/{len(ordered)}", flush=True)
        print(f"ALL_READY prefix={self.args.prefix}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/root/swe-bench-verified"))
    ap.add_argument(
        "--prompts", type=Path,
        default=Path("/root/slime/examples/remote_agent/prompts_swe_64.jsonl"),
    )
    ap.add_argument("--workspace-image")
    ap.add_argument("--image-pull-secret", default="acr-pro-registry-me")
    ap.add_argument("--max-in-flight", type=int, default=4)
    ap.add_argument("--ready-timeout", type=int, default=2400)
    args = ap.parse_args()

    config.load_incluster_config()
    if not args.workspace_image:
        pod = client.CoreV1Api().read_namespaced_pod(socket.gethostname(), args.namespace)
        args.workspace_image = pod.spec.containers[0].image
    if not args.dataset.is_dir():
        raise SystemExit(f"dataset root does not exist: {args.dataset}")
    asyncio.run(Prewarmer(args).run())


if __name__ == "__main__":
    main()
