from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import time
import wandb

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain


def policy_iteration(max_eval_iters=10, max_epochs=20):
    env = CliffWalkingEnv()
    model = PolicyIterationTrain(
        env,
        gamma=0.99,
        max_eval_iters=max_eval_iters
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

def unrl(K=10, num_unrolls=5, tau=100, beta=1.0, lr=1e-3, N=500, group=""):
    env = CliffWalkingEnv()

    model = UnrollingPolicyIterationTrain(
        env=env,
        K=K,
        num_unrolls=num_unrolls,
        gamma=0.99,
        lr=lr,
        tau=tau,
        beta=beta,
        N=N,
    )

    wandb_logger = WandbLogger(
        project="rl-unrolling",
        name=f"unrl-K{K}-{num_unrolls}unr",
        group=group
    )

    trainer = Trainer(
        max_epochs=50,
        log_every_n_steps=1,
        accelerator="cpu",
        logger=wandb_logger,
    )

    trainer.fit(model)
    wandb.finish()


if __name__ == "__main__":
    unrl(K=1, num_unrolls=2, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=5, num_unrolls=2, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=10, num_unrolls=2, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=20, num_unrolls=2, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=30, num_unrolls=2, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")

    unrl(K=1, num_unrolls=5, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=5, num_unrolls=5, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=10, num_unrolls=5, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=20, num_unrolls=5, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=30, num_unrolls=5, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")

    unrl(K=1, num_unrolls=10, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=5, num_unrolls=10, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=10, num_unrolls=10, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=20, num_unrolls=10, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=30, num_unrolls=10, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")

    unrl(K=1, num_unrolls=20, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=5, num_unrolls=20, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=10, num_unrolls=20, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=20, num_unrolls=20, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")
    unrl(K=30, num_unrolls=20, tau=1, lr=1e-3, N=500, group="unrol-tau1-lr1e-3")


    # unrl(K=20, num_unrolls=5, tau=10, lr=5e-3, N=500, group="N")

    

    
