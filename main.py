from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain


def policy_iteration():
    env = CliffWalkingEnv()
    model = PolicyIterationTrain(
        env,
        gamma=0.99,
        max_eval_iters=30
    )

    wandb_logger = WandbLogger(
        project="rl-unrolling",
        name="policy-iteration-30",
    )

    trainer = Trainer(
        max_epochs=20,
        log_every_n_steps=1,
        accelerator='cpu',
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloaders=None)

def unrl():
    env = CliffWalkingEnv()

    model = UnrollingPolicyIterationTrain(
        env=env,
        K=20,
        num_unrolls=5,
        gamma=0.99,
        lr=1e-3,
        tau=100
    )

    wandb_logger = WandbLogger(
        project="rl-unrolling",
        name="unrolled-policy-iteration",
    )

    trainer = Trainer(
        max_epochs=100,
        log_every_n_steps=1,
        accelerator="cpu",
        logger=wandb_logger,
    )

    trainer.fit(model)

if __name__ == "__main__":
    unrl()
