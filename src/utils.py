import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb

from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain


def get_optimal_q(max_eval_iters=50, max_epochs=50, group_name="", use_logger=True, log_every_n_steps=1):
    env = CliffWalkingEnv()
    model = PolicyIterationTrain(env, gamma=0.99, max_eval_iters=max_eval_iters)

    if use_logger:
            logger = WandbLogger(
            project="rl-unrolling",
            name=f"Opt_pol-{max_eval_iters}eval-{max_epochs}impr",
            group=group_name
        )
    else:
        logger = False

    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        accelerator='cpu',
        logger=logger,
    )
    
    trainer.fit(model, train_dataloaders=None)
    wandb.finish()
    return model.q.detach()


def test_pol_err(model, q_opt, max_eval_iters=200):
        q_opt = q_opt.to(model.device)

        # Get a deterministic policy
        nS, _ = model.Pi.shape
        greedy_actions = model.Pi.argmax(axis=1)
        Pi_det = np.zeros_like(model.Pi)
        Pi_det[np.arange(nS), greedy_actions] = 1.0

        # Run policy evaluation with learned policy
        env = CliffWalkingEnv()
        model = PolicyIterationTrain(env, max_eval_iters=max_eval_iters, Pi_init=torch.Tensor(Pi_det))
        model.on_fit_start()
        P_pi = model.compute_transition_matrix(model.P, model.Pi)
        q_est = model.policy_evaluation(P_pi, model.r).detach()

        q_opt = q_opt.to(model.device)
        err1 = (torch.norm(q_est - q_opt) / torch.norm(q_opt)) ** 2
        err2 = (torch.norm(q_est/torch.norm(q_est) - q_opt/torch.norm(q_opt))) ** 2
        return err1.cpu().numpy(), err2.cpu().numpy()


def plot_errors(errs, x_vals, exps, xlabel, ylabel, deviation=None, agg='mean', skip_idx=[]):
    _, axes = plt.subplots(figsize=(8, 5))

    if agg == 'median':
        agg_errs = np.median(errs, axis=0)
    elif agg == 'mean':
        agg_errs = np.mean(errs, axis=0)
    else:
        agg_errs = errs

    std = np.std(errs, axis=0)
    prctile25 = np.percentile(errs, 25, axis=0)
    prctile75 = np.percentile(errs, 75, axis=0)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue

        plt.plot(x_vals, agg_errs[i], exp['fmt'], label=exp['name'])

        if deviation == 'prctile':
            up_ci = prctile25[i]
            low_ci = prctile75[i]
        elif deviation == 'std':
            up_ci = agg_errs[i] + std[i]
            low_ci = np.maximum(agg_errs[i] - std[i], 0)
        else:
             continue
        axes.fill_between(x_vals, low_ci, up_ci, alpha=.25)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()