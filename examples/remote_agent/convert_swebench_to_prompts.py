#!/usr/bin/env python3
"""Convert SWE-bench verified dataset to Slime prompt format.

适配的训练脚本（旧 Harbor/TokenProxy 链路，metadata 带 sandbox_image，无 sandbox_set_name）：
  - run_swebench_harbor.sh   （消费本脚本输出的 prompts.jsonl，见其头部 Usage）
  - harbor_qwen.sh           （同上，PROMPT_DATA=/path/to/prompts.jsonl）
不适配 ACK E2B adapter 链路（run_swebench_e2b*.sh）——那条链路需要 metadata.sandbox_set_name，
请改用 convert_swebench_tasks_to_prompts.py。
注：harbor_local_trial.sh / run_swebench_local.sh 直接读原始 swe-bench-verified.jsonl，无需转换。

Usage:
    python convert_swebench_to_prompts.py \
        --input swe-bench-verified.jsonl \
        --output prompts.jsonl \
        --task-root /data/tasks

This creates:
1. prompts.jsonl - Slime-compatible prompt data
2. Task directories at /data/tasks/{instance_id}/ with task.json
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert SWE-bench to Slime prompts")
    parser.add_argument("--input", required=True, help="Input SWE-bench JSONL file")
    parser.add_argument("--output", required=True, help="Output prompts JSONL file")
    parser.add_argument("--task-root", default="/data/tasks", help="Root directory for task data")
    parser.add_argument("--create-task-dirs", action="store_true", help="Create task directories")
    args = parser.parse_args()

    task_root = Path(args.task_root)
    prompts = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            task = json.loads(line)
            instance_id = task["task_name"]  # e.g., "sympy__sympy-12096"

            # Build prompt data for Slime
            # task_name is used directly by generate_with_harbor (not in metadata)
            prompt_data = {
                "prompt": task["prompt"],
                "task_name": instance_id,  # Direct field for swe-bench format
                "metadata": {
                    "task_id": task.get("id", ""),
                    "category": task.get("category", ""),
                    "sandbox_image": task.get("sandbox_image", ""),
                    "run_region": task.get("run_region", ""),
                    "score": task.get("score", 1.0),
                }
            }
            prompts.append(prompt_data)

            # Create task directory if requested
            if args.create_task_dirs:
                task_dir = task_root / instance_id
                task_dir.mkdir(parents=True, exist_ok=True)

                # Write task.json for Harbor
                task_json = {
                    "instance_id": instance_id,
                    "prompt": task["prompt"],
                    "sandbox_image": task.get("sandbox_image", ""),
                    "run_region": task.get("run_region", ""),
                    "start_script": task.get("start_script", ""),
                }
                with open(task_dir / "task.json", "w", encoding="utf-8") as tf:
                    json.dump(task_json, tf, indent=2, ensure_ascii=False)

    # Write prompts.jsonl
    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Converted {len(prompts)} tasks")
    print(f"Prompts written to: {args.output}")
    if args.create_task_dirs:
        print(f"Task directories created at: {task_root}")


if __name__ == "__main__":
    main()
