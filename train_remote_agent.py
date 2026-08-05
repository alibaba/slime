"""Training entry point for the Harbor remote-agent rollout.

Token capture is handled by an in-process ``OpenAIAdapter`` that
``generate_with_harbor`` starts lazily inside the ``RolloutManager`` actor
(see ``slime.rollout.remote_agent.adapter_service``): the remote agent's
OpenAI calls hit the adapter, which routes them to the sglang router and
records token-level data per session for loss computation.

There is nothing Harbor-specific left in the training driver itself — the
adapter starts on demand and reaches sglang via the router — so this entry
point simply reuses ``train.train``. It exists as a named entry point for the
Harbor examples/docs; select the Harbor path via
``--custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor``.

Usage::

    python train_remote_agent.py \
        --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
        --harbor-agent-name swe-agent \
        --harbor-model-name openai/qwen-max \
        --harbor-task-path-template '/data/tasks/{instance_id}' \
        --harbor-adapter-public-host <head-node-ip> \
        --harbor-adapter-port 18001 \
        --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \
        ...

For local trial mode (no remote Harbor server) add ``--harbor-use-local-trial``.
"""

from slime.utils.arguments import parse_args
from train import train

if __name__ == "__main__":
    args = parse_args()
    train(args)
