import sys
from torch import multiprocessing
from torch.utils.tensorboard.writer import SummaryWriter
from continual_rl.utils.argparse_manager import ArgparseManager

import wandb

if __name__ == "__main__":

    # Pytorch multiprocessing requires either forkserver or spawn.
    try:
        multiprocessing.set_start_method("spawn")
    except ValueError as e:
        # Windows doesn't support forking, so fall back to spawn instead
        assert "cannot find context" in str(e)
        multiprocessing.set_start_method("spawn")

    experiment, policy, resume_id = ArgparseManager.parse(sys.argv[1:])

    config_wandb = {
        "mode": "online",
        "project": "4task_baselines",
        "entity": "continual-dv2",
        "name": "clear_cl_small_s" + str(resume_id),
        "group": "minihack",
        "tags": None,
        "notes": None,
    }

    wandb.init(
        reinit=True,
        resume=False,
        sync_tensorboard=True,
        **config_wandb,
    )

    if experiment is None:
        raise RuntimeError("No experiment started. Most likely there is no new run to start.")

    summary_writer = SummaryWriter(log_dir=experiment.output_dir)
    experiment.try_run(policy, summary_writer=summary_writer)

    wandb.finish()
