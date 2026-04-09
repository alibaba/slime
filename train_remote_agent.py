"""Training entry point with Remote Agent (Harbor) support.

This file wraps the standard ``train.py`` workflow with additional logic
to start the LLM TokenProxy before the training loop begins.  The proxy
intercepts OpenAI-compatible requests from the Harbor agent, routes them
to the SGLang rollout engines via Ray RPC, and records token-level data
for loss computation.

Usage::

    python train_remote_agent.py \
        --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
        --harbor-agent-name swe-agent \
        --harbor-model-name openai/qwen-max \
        --harbor-task-path-template '/data/tasks/{instance_id}' \
        --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \
        ...

For local trial mode (no remote Harbor server)::

    python train_remote_agent.py \
        --custom-generate-function-path slime.rollout.remote_agent.generate.generate_with_harbor \
        --harbor-use-local-trial \
        --harbor-agent-import-path 'my_agent.module:MyAgent' \
        --harbor-task-path-template '/data/tasks/{instance_id}' \
        --hf-checkpoint /path/to/Qwen2.5-7B-Instruct \
        ...
"""

import logging
import os

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _start_harbor_proxy(args, rollout_manager) -> str | None:
    """Start the Harbor LLM TokenProxy if remote agent args are configured.

    The proxy is created as a Ray named actor pinned to the head node.
    It serves OpenAI-compatible HTTP and records token data per session.

    Returns:
        The proxy URL (e.g. ``http://10.0.1.5:9123``), or ``None`` if
        remote agent is not configured.
    """
    if not (args.harbor_agent_name or args.harbor_agent_import_path):
        return None

    from slime.rollout.remote_agent.proxy import start_proxy_server

    # Obtain engine handles from the rollout manager
    engine_handles = ray.get(rollout_manager.get_engine_handles.remote())
    proxy_url = start_proxy_server(
        engine_handles=engine_handles,
        model_path=args.hf_checkpoint,
        host=args.harbor_proxy_host,
        port=args.harbor_proxy_port,
    )

    # Set LOCAL_IP so Harbor agents (running in Docker/K8s) can reach the proxy
    if not os.getenv("LOCAL_IP"):
        os.environ["LOCAL_IP"] = args.harbor_proxy_host

    logger.info("Harbor LLM Proxy started at %s", proxy_url)
    return proxy_url


def _stop_harbor_proxy():
    """Gracefully shut down the Harbor proxy actor."""
    try:
        from slime.rollout.remote_agent.proxy import stop_proxy_server
        stop_proxy_server()
        logger.info("Harbor LLM Proxy stopped")
    except Exception as e:
        logger.warning("Failed to stop Harbor proxy: %s", e)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # -- Harbor Remote Agent: start TokenProxy --
    proxy_url = _start_harbor_proxy(args, rollout_manager)

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # -- Cleanup --
    ray.get(rollout_manager.dispose.remote())
    if proxy_url is not None:
        _stop_harbor_proxy()
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
