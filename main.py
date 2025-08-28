"""Main training script for BellNet.

This script provides functions to train both standard policy iteration
and unrolled policy iteration models using the centralized configuration system.
"""

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import time
import wandb
from typing import Optional

from src.algorithms import UnrollingPolicyIterationTrain
from src import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms import PolicyIterationTrain
from config import get_config, Config


def policy_iteration(config: Optional[Config] = None, max_eval_iters: int = 10, max_epochs: int = 20) -> None:
    """Train standard policy iteration model.
    
    Args:
        config: Configuration object (optional)
        max_eval_iters: Maximum policy evaluation iterations
        max_epochs: Maximum training epochs
    """
    if config is None:
        config = get_config('default')
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
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
    )
    
    t0 = time.perf_counter()
    trainer.fit(model, train_dataloaders=None)
    elapsed = time.perf_counter() - t0
    wandb_logger.experiment.summary["fit_time_sec"] = elapsed
    print(f"Ellapsed time: {elapsed:.2f} s")

    wandb.finish()

def unrl(config: Optional[Config] = None, K: int = 10, num_unrolls: int = 10, 
         tau: float = 100, beta: float = 1.0, lr: float = 1e-3, N: int = 500, 
         weight_sharing: bool = False, group: str = "", init_q: str = "zeros",
         architecture_type: int = 1) -> None:
    """Train unrolled policy iteration model.
    
    Args:
        config: Configuration object (optional)
        K: Graph filter order
        num_unrolls: Number of unrolling steps
        tau: Temperature parameter
        beta: Bellman operator parameter
        lr: Learning rate
        N: Dataset size
        weight_sharing: Whether to share weights across layers
        group: Experiment group name
        init_q: Q initialization method
        architecture_type: Architecture type (1, 2, 3, or 5)
    """
    if config is None:
        config = get_config('default')
        # Override with provided parameters
        config.model.K = K
        config.model.num_unrolls = num_unrolls
        config.model.tau = tau
        config.model.beta = beta
        config.model.weight_sharing = weight_sharing
        config.model.init_q = init_q
        config.training.lr = lr
        config.training.N = N
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
        init_q=init_q,
        architecture_type=architecture_type,
    )

    wandb_logger = WandbLogger(
        project="rl-unrolling",
        name=f"unrl-K{K}-{num_unrolls}unr-WS{weight_sharing}-init{init_q}-arch{architecture_type}",
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
