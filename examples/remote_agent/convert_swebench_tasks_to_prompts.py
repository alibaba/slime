#!/usr/bin/env python3
"""Convert harbor-format SWE-bench task directories into slime prompt JSONL.

输出的 prompts.jsonl 由统一启动器 ``run_swebench.sh`` 消费（按 metadata.sandbox_set_name
路由到预建 SandboxSet）。多模态模型（如 27B）用 ``--prompt-as-messages`` 生成消息 list 格式，
并在 run_swebench.sh 侧配 ``APPLY_CHAT_TEMPLATE=1``。参数详解见
examples/remote_agent/README.md（参数详解节）。

Scans ``<dataset-root>/<instance_id>/task.toml`` (harbor task format) and writes
one line per task in the format consumed by ``run_swebench.sh``:

    {"prompt": "...", "task_name": "astropy__astropy-14309",
     "metadata": {"sandbox_set_name": "slime-sbx-astropy-14309"}}

- ``prompt`` comes from ``problem_statement`` in ``task.toml``; newer task layouts
  keep it in a sibling ``instruction.md`` file instead (used as fallback).
- ``sandbox_set_name`` = ``--sandbox-set-prefix`` + a short name, chosen via ``--set-name-from``:
  ``suffix`` (default, part after ``__``: ``astropy__astropy-14309`` -> ``astropy-14309``,
  e.g. ``slime-sbx-astropy-14309``) or ``image`` (short name of
  ``[environment].docker_image``, e.g. ``astropy-astropy-14309``).
  Only relevant when your environment routes trials to named pools; otherwise omit
  the prefix and the environment uses its default.

Example::

    python examples/remote_agent/convert_swebench_tasks_to_prompts.py \
      --dataset-root ./tasks/swe-bench-verified \
      --output examples/remote_agent/small_prompts.jsonl

For multimodal models, add ``--prompt-as-messages`` so ``prompt`` becomes
``[{"role": "user", "content": ...}]`` and pass ``--apply-chat-template`` to the
training script.
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_DATASET_ROOT = "./tasks/swe-bench-verified"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "small_prompts.jsonl"
DEFAULT_SET_PREFIX = "slime-sbx-"


def load_task_toml(path: Path) -> dict:
    """Parse task.toml; falls back to regex extraction when tomllib/tomli is absent."""
    text = path.read_text(encoding="utf-8")
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(text)
    except ImportError:
        return {"__fallback__": text}


def _regex_value(text: str, key: str) -> str | None:
    """Minimal fallback: extract a TOML key value (multi-line or single-line string)."""
    m = re.search(rf'^\s*{key}\s*=\s*"""(.*?)"""', text, re.S | re.M)
    if m:
        return m.group(1)
    m = re.search(rf'^\s*{key}\s*=\s*"((?:[^"\\]|\\.)*)"\s*$', text, re.M)
    if m:
        return m.group(1).encode().decode("unicode_escape", errors="replace")
    return None


def _find_key(data: dict, key: str) -> str | None:
    """Look up ``key`` in top level and known task.toml sections."""
    if key in data and isinstance(data[key], str):
        return data[key]
    for section in ("issue", "environment", "task", "metadata"):
        value = data.get(section)
        if isinstance(value, dict) and isinstance(value.get(key), str):
            return value[key]
    return None


def image_short_name(instance_id: str, docker_image: str | None) -> str:
    """``reg.../swebench-verified/astropy-astropy-14309:tag`` -> ``astropy-astropy-14309``."""
    if docker_image:
        return docker_image.split("/")[-1].split(":")[0]
    return instance_id.replace("__", "-").lower()


def instance_suffix(instance_id: str) -> str:
    """``astropy__astropy-14309`` -> ``astropy-14309`` (part after ``__``)."""
    return instance_id.split("__", 1)[1] if "__" in instance_id else instance_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help=f"Harbor task root (default: {DEFAULT_DATASET_ROOT})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Output prompts JSONL (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--sandbox-set-prefix", default=DEFAULT_SET_PREFIX, help=f"SandboxSet name prefix (default: {DEFAULT_SET_PREFIX})")
    parser.add_argument(
        "--set-name-from",
        choices=["suffix", "image"],
        default="suffix",
        help="Short name source for sandbox_set_name: 'suffix' = instance_id part after '__' "
        "(default, e.g. slime-sbx-astropy-14309); 'image' = docker_image short name",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only convert the first N tasks (0 = all)")
    parser.add_argument("--prompt-as-messages", action="store_true", help='Emit prompt as [{"role": "user", "content": ...}] (multimodal models + --apply-chat-template)')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        sys.exit(f"dataset root not found: {dataset_root}")

    task_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir() and (d / "task.toml").is_file())
    if args.limit > 0:
        task_dirs = task_dirs[: args.limit]
    if not task_dirs:
        sys.exit(f"no <instance_id>/task.toml found under {dataset_root}")

    records, skipped = [], []
    for task_dir in task_dirs:
        instance_id = task_dir.name
        data = load_task_toml(task_dir / "task.toml")
        if "__fallback__" in data:
            problem = _regex_value(data["__fallback__"], "problem_statement")
            docker_image = _regex_value(data["__fallback__"], "docker_image")
        else:
            problem = _find_key(data, "problem_statement")
            docker_image = _find_key(data, "docker_image")
        if not problem:
            # Newer task layouts keep the prompt in a sibling instruction.md.
            instruction = task_dir / "instruction.md"
            if instruction.is_file():
                problem = instruction.read_text(encoding="utf-8").strip() or None
        if not problem:
            skipped.append(instance_id)
            continue

        prompt = [{"role": "user", "content": problem}] if args.prompt_as_messages else problem
        short = instance_suffix(instance_id) if args.set_name_from == "suffix" else image_short_name(instance_id, docker_image)
        records.append(
            {
                "prompt": prompt,
                "task_name": instance_id,
                "metadata": {"sandbox_set_name": args.sandbox_set_prefix + short},
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} tasks -> {output}")
    if skipped:
        print(f"Skipped {len(skipped)} tasks without problem_statement: {skipped}")


if __name__ == "__main__":
    main()
