"""Utility functions for BellNet experiments.

This module contains helper functions for running experiments,
evaluating policies, and processing results.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb

from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain


def get_optimal_q(max_eval_iters: int = 50, max_epochs: int = 50, 
                  group_name: str = "", mirror_env: bool = False,
                  use_logger: bool = True, log_every_n_steps: int = 1) -> torch.Tensor:
    """Compute optimal Q-values using policy iteration.
    
    Args:
        max_eval_iters: Maximum policy evaluation iterations
        max_epochs: Maximum policy improvement epochs
        group_name: Experiment group name for logging
        mirror_env: Whether to use mirrored cliff environment
        use_logger: Whether to log to wandb
        log_every_n_steps: Logging frequency
        
    Returns:
        Optimal Q-values tensor
    """
    if mirror_env:
        env =  MirroredCliffWalkingEnv()
        model = PolicyIterationTrain(env, gamma=0.99, goal_row=0, max_eval_iters=max_eval_iters)
    else:
        env = CliffWalkingEnv()
        model = PolicyIterationTrain(env, gamma=0.99, goal_row=3, max_eval_iters=max_eval_iters)
    
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


def test_pol_err(Pi: torch.Tensor, q_opt: torch.Tensor, mirror_env: bool = False, 
                 max_eval_iters: int = 200, device: str = "cpu") -> Tuple[float, float]:
    """Test policy error against optimal Q-values.
    
    Args:
        Pi: Policy to evaluate
        q_opt: Optimal Q-values
        mirror_env: Whether to use mirrored environment
        max_eval_iters: Maximum evaluation iterations
        device: Device to run on
        
    Returns:
        Tuple of (relative_error, normalized_error)
    """
        q_opt = q_opt.to(device)

        # Get a deterministic policy
        nS, _ = Pi.shape
        # greedy_actions = Pi.argmax(axis=1)
        max_vals = Pi.max(dim=1, keepdim=True).values
        is_max = Pi == max_vals
        greedy_actions = torch.multinomial(is_max.float(), num_samples=1).squeeze(1)
        Pi_det = np.zeros_like(Pi)
        Pi_det[np.arange(nS), greedy_actions] = 1.0
        
        # Run policy evaluation with learned policy
        if mirror_env:
            env =  MirroredCliffWalkingEnv()
            model_polit = PolicyIterationTrain(env, gamma=0.99, goal_row=0, max_eval_iters=max_eval_iters, Pi_init=torch.Tensor(Pi_det))
        else:
            env = CliffWalkingEnv()
            model_polit = PolicyIterationTrain(env, gamma=0.99, goal_row=3, max_eval_iters=max_eval_iters, Pi_init=torch.Tensor(Pi_det))

        model_polit.on_fit_start()
        P_pi = model_polit.compute_transition_matrix(model_polit.P, model_polit.Pi)
        q_est = model_polit.policy_evaluation(P_pi, model_polit.r).detach()

        q_opt = q_opt.to(device)
        err1 = (torch.norm(q_est - q_opt) / torch.norm(q_opt)) ** 2
        err2 = (torch.norm(q_est/torch.norm(q_est) - q_opt/torch.norm(q_opt))) ** 2
        return err1.cpu().numpy(), err2.cpu().numpy()

def plot_errors(errs: np.ndarray, x_vals: List, exps: List[Dict[str, Any]], 
                xlabel: str, ylabel: str, deviation: Optional[str] = None, 
                agg: str = 'mean', skip_idx: List[int] = []) -> None:
    """Plot experimental errors with optional confidence intervals.
    
    Args:
        errs: Error matrix of shape (n_experiments, n_points)
        x_vals: X-axis values
        exps: List of experiment configurations
        xlabel: X-axis label
        ylabel: Y-axis label
        deviation: Type of deviation to plot ('std', 'prctile', None)
        agg: Aggregation method ('mean', 'median')
        skip_idx: Indices of experiments to skip
    """
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

def save_error_matrix_to_csv(error_matrix: np.ndarray, xaxis: List, 
                            exps: List[Dict[str, Any]], filename: str, 
                            delimiter: str = ';') -> None:
    """Save error matrix to CSV file.
    
    Args:
        error_matrix: Matrix of experimental errors
        xaxis: X-axis values
        exps: List of experiment configurations
        filename: Output filename
        delimiter: CSV delimiter
    """
    # Find first unique names and their indices
    seen = set()
    unique_indices = []
    unique_names = []

    for i, exp in enumerate(exps):
        name = exp["name"]
        if name not in seen:
            seen.add(name)
            unique_indices.append(i)
            unique_names.append(name)
        else:
            print(f"Warning: Duplicate experiment '{name}' found. Only the first occurrence will be saved.")

    # Extract only the relevant columns
    error_matrix_filtered = error_matrix[unique_indices] if error_matrix.shape[0] == len(exps) else error_matrix[:, unique_indices]

    # Transpose to (n_unrolls, n_experiments)
    if error_matrix_filtered.shape[0] == len(unique_names):
        data = error_matrix_filtered.T
    else:
        data = error_matrix_filtered

    # Insert xaxis as the first column
    xaxis = np.asarray(xaxis).reshape(-1, 1)  # Ensure it's a column vector
    data_with_x = np.hstack((xaxis, data))   # Add as first column

    # Create header
    header = delimiter.join(['xaxis'] + unique_names)

    # Save to CSV
    np.savetxt(filename, data_with_x, delimiter=delimiter, header=header, comments='')
    print("Data saved to csv file:", filename)