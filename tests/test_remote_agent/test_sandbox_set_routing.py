"""Unit tests for per-task SandboxSet routing in the Harbor generate path.

Covers ``_read_task_sandbox_class`` and ``_resolve_sandbox_set_name`` — the
plumbing that converts a per-instance pod-size class (from ``task.toml`` or the
sample metadata) into a SandboxSet name passed to Harbor via
``environment_kwargs``.

The generate module transitively imports heavy runtime deps (shortuuid and the
in-process adapter service, which pulls in torch/aiohttp/tokenizers). We stub
those so the two pure resolver functions can be loaded and exercised without the
full stack.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HARBOR_CLIENT_PY = _REPO_ROOT / "slime" / "rollout" / "remote_agent" / "harbor_client.py"
_GENERATE_PY = _REPO_ROOT / "slime" / "rollout" / "remote_agent" / "generate.py"


def _load_harbor_client_module():
    spec = importlib.util.spec_from_file_location("_harbor_client_under_test", _HARBOR_CLIENT_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stub(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def _load_generate_module():
    """Load generate.py in isolation with its absolute imports stubbed."""
    _install_stub("shortuuid", uuid=lambda: "stubuuid")
    # slime package namespace + the things generate.py imports from slime
    for pkg in ("slime", "slime.rollout", "slime.rollout.remote_agent", "slime.utils"):
        _install_stub(pkg)
    _install_stub("slime.utils.misc", SingletonMeta=type)
    _install_stub("slime.utils.types", Sample=object)
    _install_stub(
        "slime.rollout.remote_agent.harbor_client",
        HarborAgentConfig=object,
        HarborClient=object,
        HarborRunResult=object,
        HarborVerifierConfig=object,
        run_local_trial=lambda *a, **k: None,
    )
    _install_stub("slime.rollout.remote_agent.adapter_service", HarborAdapterService=object)

    spec = importlib.util.spec_from_file_location("_harbor_generate_under_test", _GENERATE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


client = _load_harbor_client_module()
gen = _load_generate_module()


def test_k8s_resource_name_is_readable_and_dns_safe():
    assert client._make_k8s_resource_name("Pallets__Flask-5014/ABC_123") == "pallets-flask-5014-abc-123"


def test_k8s_resource_name_truncates_with_stable_hash():
    raw = f"Task__{'x' * 80}-ABC_123"
    name = client._make_k8s_resource_name(raw)

    assert len(name) <= 63
    assert name == client._make_k8s_resource_name(raw)
    assert name.startswith("task-")
    assert name[-9] == "-"


def _sample(metadata: dict):
    return types.SimpleNamespace(metadata=metadata, index=0)


def _args(**over) -> Namespace:
    base = dict(
        harbor_sandbox_class_key="sandbox_class",
        harbor_sandbox_set_name_template="{sandbox_class}",
    )
    base.update(over)
    return Namespace(**base)


def _write_task_toml(tmp_path: Path, body: str) -> str:
    (tmp_path / "task.toml").write_text(body)
    return str(tmp_path)


def test_read_class_from_metadata_section(tmp_path):
    task_path = _write_task_toml(tmp_path, '[metadata]\nsandbox_class = "large"\n')
    assert gen._read_task_sandbox_class(task_path, "sandbox_class") == "large"


def test_read_class_missing_file_returns_none(tmp_path):
    assert gen._read_task_sandbox_class(str(tmp_path), "sandbox_class") is None


def test_read_class_absent_key_returns_none(tmp_path):
    task_path = _write_task_toml(tmp_path, '[metadata]\ndifficulty = "hard"\n')
    assert gen._read_task_sandbox_class(task_path, "sandbox_class") is None


def test_resolve_explicit_name_from_metadata_wins(tmp_path):
    task_path = _write_task_toml(tmp_path, '[metadata]\nsandbox_class = "large"\n')
    sample = _sample({"sandbox_set_name": "explicit-pool", "sandbox_class": "small"})
    assert gen._resolve_sandbox_set_name(_args(), sample, task_path) == "explicit-pool"


def test_resolve_class_from_metadata_applies_template(tmp_path):
    sample = _sample({"sandbox_class": "medium"})
    args = _args(harbor_sandbox_set_name_template="swebench-verified-{sandbox_class}")
    assert gen._resolve_sandbox_set_name(args, sample, str(tmp_path)) == "swebench-verified-medium"


def test_resolve_falls_back_to_task_toml(tmp_path):
    task_path = _write_task_toml(tmp_path, '[metadata]\nsandbox_class = "xlarge"\n')
    args = _args(harbor_sandbox_set_name_template="pool-{sandbox_class}")
    assert gen._resolve_sandbox_set_name(args, _sample({}), task_path) == "pool-xlarge"


def test_resolve_returns_none_when_no_class(tmp_path):
    assert gen._resolve_sandbox_set_name(_args(), _sample({}), str(tmp_path)) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
