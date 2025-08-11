#!/usr/bin/env python3
"""
Specific experiments runner for BN-5 WS only with random initialization and no detach.

This script implements the specific experiment configuration requested:
- BN-5 (BellNet with 5 unrolls)  
- Solo la versión WS (Weight Sharing)
- Random initialization 
- Sin detach (gradients flow through)
- Reproduce figures A, B, C from the paper
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from time import perf_counter
import wandb
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

# Import necessary modules
from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.utils import get_optimal_q, test_pol_err, plot_errors, save_error_matrix_to_csv


def run_bn5_ws_experiments(
    use_logger: bool = True, 
    log_every_n_steps: int = 1, 
    verbose: bool = True,
    save_results: bool = True,
    results_path: str = "results/bn5_ws_specific/"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run specific experiments for BN-5 WS only with random initialization.
    
    Reproduces experiments for Figures A, B, C but only with:
    - BN-5 (5 unrolls)
    - Weight Sharing version only
    - Random initialization
    - No detach in gradient flow
    
    Args:
        use_logger: Whether to use wandb logging
        log_every_n_steps: Logging frequency
        verbose: Whether to print progress
        save_results: Whether to save results to files
        results_path: Path to save results
        
    Returns:
        Tuple of (errors1, errors2, bellman_errors) arrays
    """
    
    # Experiment configuration - BN-5 WS only
    num_unrolls = 5
    group_name = f"bn5-ws-random-init-no-detach"
    
    # Test different K values for Figure A equivalent
    Ks = np.array([1, 2, 3, 5, 10, 15])
    
    # Experiments: Only BN-5 WS and corresponding baseline
    Exps = [
        # Baseline: Policy iteration with same number of evaluations
        {
            "model": "pol-it", 
            "args": {"max_eval_iters": num_unrolls}, 
            "fmt": "x-", 
            "name": f"pol-it-{num_unrolls}eval"
        },
        # Our method: BN-5 with Weight Sharing and random init
        {
            "model": "unroll", 
            "args": {
                "num_unrolls": num_unrolls, 
                "tau": 5, 
                "lr": 5e-3, 
                "weight_sharing": True,  # Only WS version
                "init_q": "random"       # Random initialization
            }, 
            "fmt": "o-", 
            "name": f"unr-{num_unrolls}unrolls-WS-random"
        },
    ]
    
    # Get optimal Q functions
    if verbose:
        print("Computing optimal Q functions...")
    q_opt = get_optimal_q(
        use_logger=use_logger, 
        log_every_n_steps=log_every_n_steps, 
        group_name=group_name
    )
    
    # Number of runs for statistical significance
    n_runs = 5
    
    # Initialize result arrays
    errs1 = np.zeros((n_runs, len(Exps), Ks.size))
    errs2 = np.zeros((n_runs, len(Exps), Ks.size))
    bell_errs = np.zeros((n_runs, len(Exps), Ks.size))
    
    if verbose:
        print(f"Running {n_runs} runs for each configuration...")
        print(f"Testing K values: {Ks}")
        print(f"Experiments: {[exp['name'] for exp in Exps]}")
    
    t_init = perf_counter()
    
    # Run experiments
    for run_idx in range(n_runs):
        if verbose:
            print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
            
        for k_idx, K in enumerate(Ks):
            K = int(K)
            if verbose:
                print(f"  Testing K={K}")
                
            for exp_idx, exp in enumerate(Exps):
                if verbose:
                    print(f"    Running {exp['name']}")
                
                env = CliffWalkingEnv()
                
                if exp["model"] == "unroll":
                    # BellNet model
                    model = UnrollingPolicyIterationTrain(
                        env=env, 
                        env_test=env, 
                        K=K, 
                        **exp["args"]
                    )
                    
                    if use_logger and run_idx == 0:  # Log only first run
                        logger = WandbLogger(
                            project="rl-unrolling-specific", 
                            name=f"{exp['name']}-K{K}",
                            group=group_name
                        )
                    else:
                        logger = False
                        
                    trainer = Trainer(
                        max_epochs=3000, 
                        log_every_n_steps=log_every_n_steps, 
                        accelerator="cpu", 
                        logger=logger
                    )
                    
                elif exp["model"] == "pol-it":
                    # Policy iteration baseline
                    model = PolicyIterationTrain(
                        env=env, 
                        **exp["args"]
                    )
                    
                    if use_logger and run_idx == 0:  # Log only first run
                        logger = WandbLogger(
                            project="rl-unrolling-specific", 
                            name=f"{exp['name']}-{K}impr",
                            group=group_name
                        )
                    else:
                        logger = False
                        
                    trainer = Trainer(
                        max_epochs=exp['args']['max_eval_iters'], 
                        log_every_n_steps=log_every_n_steps, 
                        accelerator='cpu',
                        logger=logger
                    )
                else:
                    raise ValueError(f"Unknown model type: {exp['model']}")
                
                # Train model
                trainer.fit(model)
                if wandb.run is not None:
                    wandb.finish()
                
                # Compute errors
                err1, err2 = test_pol_err(model.Pi, q_opt)
                bell_err = model.bellman_error.cpu().numpy()
                
                # Store results
                errs1[run_idx, exp_idx, k_idx] = err1
                errs2[run_idx, exp_idx, k_idx] = err2
                bell_errs[run_idx, exp_idx, k_idx] = bell_err
                
                if verbose:
                    print(f"      Err1: {err1:.4f}, Err2: {err2:.4f}, Bell: {bell_err:.4f}")
    
    t_end = perf_counter()
    
    if verbose:
        print(f'\n----- Completed in {(t_end-t_init)/60:.2f} minutes -----')
        print(f"Results shape: {errs1.shape} (runs x experiments x K_values)")
    
    # Save results if requested
    if save_results:
        import os
        os.makedirs(results_path, exist_ok=True)
        
        file_name = results_path + f"{group_name}_data.npz"
        np.savez(
            file_name, 
            Ks=Ks, 
            Exps=Exps, 
            errs1=errs1, 
            errs2=errs2, 
            bell_errs=bell_errs,
            n_runs=n_runs
        )
        if verbose:
            print(f"Results saved to: {file_name}")
    
    return errs1, errs2, bell_errs


def plot_bn5_ws_results(
    errs1: np.ndarray, 
    errs2: np.ndarray, 
    bell_errs: np.ndarray,
    Ks: np.ndarray,
    Exps: List[Dict[str, Any]],
    save_plots: bool = True,
    results_path: str = "results/bn5_ws_specific/"
) -> None:
    """
    Generate plots equivalent to Figures A, B, C but for BN-5 WS only.
    
    Args:
        errs1: Error type 1 results
        errs2: Error type 2 results  
        bell_errs: Bellman error results
        Ks: K values tested
        Exps: Experiment configurations
        save_plots: Whether to save plot data
        results_path: Path to save results
    """
    
    print("\nGenerating plots for BN-5 WS specific experiments...")
    
    # Plot Figure A equivalent: Performance vs K
    skip_idx = []
    xlabel = "K (Graph Filter Order)"
    
    # Plot all error types
    plot_errors(
        errs1, Ks, Exps, xlabel, "Q Error Type 1 (BN-5 WS)", 
        skip_idx=skip_idx, agg="median", deviation='prctile'
    )
    
    plot_errors(
        errs2, Ks, Exps, xlabel, "Q Error Type 2 (BN-5 WS)", 
        skip_idx=skip_idx, agg="median", deviation='prctile'
    )
    
    plot_errors(
        bell_errs, Ks, Exps, xlabel, "Bellman Error (BN-5 WS)", 
        skip_idx=skip_idx, agg="median", deviation='prctile'
    )
    
    # Save CSV files for further analysis if requested
    if save_plots:
        import os
        os.makedirs(results_path, exist_ok=True)
        
        # Save median errors
        file_name = results_path + "bn5_ws_median_errors.csv"
        save_error_matrix_to_csv(
            np.median(errs2, axis=0), Ks, Exps, file_name
        )
        
        # Save percentile bounds
        file_name = results_path + "bn5_ws_p25_errors.csv"
        save_error_matrix_to_csv(
            np.percentile(errs2, 25, axis=0), Ks, Exps, file_name
        )
        
        file_name = results_path + "bn5_ws_p75_errors.csv"
        save_error_matrix_to_csv(
            np.percentile(errs2, 75, axis=0), Ks, Exps, file_name
        )
        
        print(f"Plot data saved to: {results_path}")


def run_transferability_bn5_ws(
    use_logger: bool = True,
    log_every_n_steps: int = 1,
    verbose: bool = True,
    save_results: bool = True,
    results_path: str = "results/bn5_ws_specific/"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run transferability experiments for BN-5 WS (Figure B equivalent).
    
    Tests how well BN-5 WS transfers across different environments.
    """
    
    print("Running transferability experiments for BN-5 WS...")
    
    K = 5  # Fixed K for BN-5
    group_name = f"transfer-bn5-ws-random-init"
    
    # Test different numbers of unrolls
    N_unrolls = np.arange(2, 11, 2)
    
    # Only BN-5 WS experiment
    Exps = [
        # Baseline: Value iteration
        {
            "model": "pol-it", 
            "args": {"max_eval_iters": 1}, 
            "fmt": "^-", 
            "name": "val-it"
        },
        # Our method: BN-5 WS with random init
        {
            "model": "unroll", 
            "args": {
                "K": K, 
                "tau": 5, 
                "lr": 5e-3, 
                "weight_sharing": True,
                "init_q": "random"
            }, 
            "fmt": "o-", 
            "name": f"unr-K{K}-WS-random"
        },
    ]
    
    # Get optimal Q functions for both environments
    q_opt = get_optimal_q(
        mirror_env=False, 
        use_logger=use_logger, 
        log_every_n_steps=log_every_n_steps, 
        group_name=group_name
    )
    q_opt_mirr = get_optimal_q(
        mirror_env=True, 
        use_logger=use_logger, 
        log_every_n_steps=log_every_n_steps, 
        group_name=group_name
    )
    
    # Run transferability experiment
    n_runs = 5
    errs = np.zeros((n_runs, len(Exps), N_unrolls.size))
    errs_trans = np.zeros((n_runs, len(Exps), N_unrolls.size))
    bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))
    
    t_init = perf_counter()
    
    for run_idx in range(n_runs):
        if verbose:
            print(f"\n--- Transferability Run {run_idx + 1}/{n_runs} ---")
            
        for unroll_idx, n_unrolls in enumerate(N_unrolls):
            n_unrolls = int(n_unrolls)
            if verbose:
                print(f"  Testing {n_unrolls} unrolls")
                
            for exp_idx, exp in enumerate(Exps):
                env = CliffWalkingEnv()
                env_test = MirroredCliffWalkingEnv()
                
                if exp["model"] == "unroll":
                    model = UnrollingPolicyIterationTrain(
                        env=env, 
                        env_test=env_test, 
                        num_unrolls=n_unrolls, 
                        **exp["args"]
                    )
                    
                    if use_logger and run_idx == 0:
                        logger = WandbLogger(
                            project="rl-unrolling-transfer", 
                            name=f"{exp['name']}-{n_unrolls}unrolls",
                            group=group_name
                        )
                    else:
                        logger = False
                        
                    trainer = Trainer(
                        max_epochs=3000, 
                        log_every_n_steps=log_every_n_steps, 
                        accelerator="cpu", 
                        logger=logger
                    )
                    
                    trainer.fit(model)
                    if wandb.run is not None:
                        wandb.finish()
                    
                    # Test on both environments
                    _, err = test_pol_err(model.Pi, q_opt, mirror_env=False, device=model.device)
                    _, err_trans = test_pol_err(model.Pi_test, q_opt_mirr, mirror_env=True, device=model.device)
                    bell_err = model.bellman_error_test.cpu().numpy()
                    
                elif exp["model"] == "pol-it":
                    model = PolicyIterationTrain(
                        env=env_test, 
                        goal_row=0, 
                        max_eval_iters=n_unrolls
                    )
                    
                    if use_logger and run_idx == 0:
                        logger = WandbLogger(
                            project="rl-unrolling-transfer", 
                            name=f"{exp['name']}-{n_unrolls}impr",
                            group=group_name
                        )
                    else:
                        logger = False
                        
                    trainer = Trainer(
                        max_epochs=n_unrolls, 
                        log_every_n_steps=log_every_n_steps, 
                        accelerator='cpu', 
                        logger=logger
                    )
                    
                    trainer.fit(model)
                    if wandb.run is not None:
                        wandb.finish()
                    
                    _, err = test_pol_err(model.Pi, q_opt_mirr, mirror_env=True, device=model.device)
                    err_trans = err  # Same for policy iteration
                    bell_err = model.bellman_error.cpu().numpy()
                
                # Store results
                errs[run_idx, exp_idx, unroll_idx] = err
                errs_trans[run_idx, exp_idx, unroll_idx] = err_trans
                bell_errs[run_idx, exp_idx, unroll_idx] = bell_err
                
                if verbose:
                    print(f"    {exp['name']}: Train={err:.4f}, Transfer={err_trans:.4f}")
    
    t_end = perf_counter()
    if verbose:
        print(f'\n----- Transfer experiments completed in {(t_end-t_init)/60:.2f} minutes -----')
    
    # Save results
    if save_results:
        import os
        os.makedirs(results_path, exist_ok=True)
        
        file_name = results_path + f"{group_name}_transfer_data.npz"
        np.savez(
            file_name, 
            N_unrolls=N_unrolls, 
            Exps=Exps, 
            errs=errs, 
            errs_trans=errs_trans, 
            bell_errs=bell_errs,
            n_runs=n_runs
        )
        if verbose:
            print(f"Transfer results saved to: {file_name}")
    
    return errs, errs_trans, bell_errs


if __name__ == "__main__":
    print("Running specific experiments: BN-5 WS with random init and no detach")
    print("=" * 60)
    
    # Configuration
    use_logger = False  # Set to True for W&B logging
    verbose = True
    save_results = True
    
    # 1. Run main experiments (Figure A equivalent)
    print("\n1. Running main K-value experiments...")
    errs1, errs2, bell_errs = run_bn5_ws_experiments(
        use_logger=use_logger,
        verbose=verbose, 
        save_results=save_results
    )
    
    # 2. Generate plots
    print("\n2. Generating plots...")
    Ks = np.array([1, 2, 3, 5, 10, 15])
    Exps = [
        {"name": f"pol-it-5eval", "fmt": "x-"},
        {"name": f"unr-5unrolls-WS-random", "fmt": "o-"}
    ]
    
    plot_bn5_ws_results(
        errs1, errs2, bell_errs, Ks, Exps, 
        save_plots=save_results
    )
    
    # 3. Run transferability experiments (Figure B equivalent)  
    print("\n3. Running transferability experiments...")
    errs_train, errs_transfer, bell_errs_transfer = run_transferability_bn5_ws(
        use_logger=use_logger,
        verbose=verbose,
        save_results=save_results
    )
    
    print("\n" + "=" * 60)
    print("All BN-5 WS specific experiments completed!")
    print("Results saved in: results/bn5_ws_specific/")