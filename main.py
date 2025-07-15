from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import time
import wandb

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain


def policy_iteration(max_eval_iters=10, max_epochs=20):
    env = MirroredCliffWalkingEnv()
    model = PolicyIterationTrain(
        env,
        gamma=0.99,
        max_eval_iters=max_eval_iters,
        goal_row=0
    )

    wandb_logger = WandbLogger(
        project="rl-unrolling",
        name=f"pol-it-{max_eval_iters}eval-{max_epochs}impr",
        # group=f"{max_epochs} impr"
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        accelerator='cpu',
        logger=wandb_logger,
    )
    
    t0 = time.perf_counter()
    trainer.fit(model, train_dataloaders=None)
    elapsed = time.perf_counter() - t0
    wandb_logger.experiment.summary["fit_time_sec"] = elapsed
    print(f"Ellapsed time: {elapsed:.2f} s")

    wandb.finish()

def unrl(K=10, num_unrolls=10, tau=100, beta=1.0, lr=1e-3, N=500, weight_sharing=False, group=""):
    env = CliffWalkingEnv()
    env_test = MirroredCliffWalkingEnv()

    model = UnrollingPolicyIterationTrain(
        env=env,
        env_test=env_test,
        K=K,
        num_unrolls=num_unrolls,
        gamma=0.99,
        lr=lr,
        tau=tau,
        beta=beta,
        N=N,
        weight_sharing=weight_sharing,
    )

    wandb_logger = WandbLogger(
        project="rl-unrolling",
        name=f"unrl-K{K}-{num_unrolls}unr-WS{weight_sharing}",
        group=group
    )

    trainer = Trainer(
        max_epochs=5000,
        log_every_n_steps=1,
        accelerator="cpu",
        logger=wandb_logger,
    )

    trainer.fit(model)
    wandb.finish()


if __name__ == "__main__":
    unrl(K=10, num_unrolls=10, tau=100, lr=5e-3, N=1, group="N", weight_sharing=False)
    # policy_iteration(max_eval_iters=10, max_epochs=20)
