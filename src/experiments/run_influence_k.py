#!/usr/bin/env python3
"""
Runner for influence of K (graph filter order) experiments.

This script reproduces the filter order experiments (Figure A equivalent)
testing different values of K with various model configurations.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from time import perf_counter
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    get_optimal_q, 
    run_influence_k_experiment, 
    plot_errors, 
    save_error_matrix_to_csv
)


def main(
    unroll_values: List[int] = [5, 10, 15],
    k_values: List[int] = [1, 2, 3, 5, 10, 15],
    n_runs: int = 5,
    use_logger: bool = False,
    save_results: bool = True,
    results_path: str = "results/filter_order/",
    verbose: bool = True
) -> None:
    """
    Run influence of K experiments for different numbers of unrolls.
    
    Args:
        unroll_values: List of unroll numbers to test
        k_values: List of K values to test  
        n_runs: Number of statistical runs
        use_logger: Whether to log to wandb
        save_results: Whether to save results to files
        results_path: Directory to save results
        verbose: Whether to print progress
    """
    
    print("Running influence of K (filter order) experiments")
    print("=" * 60)
    print(f"Unroll values: {unroll_values}")
    print(f"K values: {k_values}")
    print(f"Number of runs: {n_runs}")
    print()
    
    # Create results directory
    if save_results:
        os.makedirs(results_path, exist_ok=True)
    
    Ks = np.array(k_values)
    
    for num_unrolls in unroll_values:
        print(f"\n{'='*20} UNROLLS = {num_unrolls} {'='*20}")
        
        group_name = f"filter_order-{num_unrolls}"
        
        # Define experiments for this unroll value
        experiments: List[Dict[str, Any]] = [
            {
                "model": "pol-it", 
                "args": {"max_epochs": num_unrolls}, 
                "fmt": "x-", 
                "name": f"pol-it-{num_unrolls}eval"
            },
            {
                "model": "unroll", 
                "args": {
                    "num_unrolls": num_unrolls, 
                    "tau": 5, 
                    "lr": 5e-3, 
                    "weight_sharing": True
                }, 
                "fmt": "o-", 
                "name": f"unr-{num_unrolls}unrolls-WS"
            },
            {
                "model": "unroll", 
                "args": {
                    "num_unrolls": num_unrolls, 
                    "tau": 5, 
                    "lr": 5e-3, 
                    "weight_sharing": False
                }, 
                "fmt": "o-", 
                "name": f"unr-{num_unrolls}unrolls"
            },
        ]
        
        # Get optimal Q
        if verbose:
            print("Computing optimal Q-values...")
        q_opt = get_optimal_q(
            use_logger=use_logger, 
            log_every_n_steps=1, 
            group_name=group_name
        )
        
        # Initialize result arrays
        errs1 = np.zeros((n_runs, len(experiments), Ks.size))
        errs2 = np.zeros((n_runs, len(experiments), Ks.size))
        bell_errs = np.zeros((n_runs, len(experiments), Ks.size))
        
        if verbose:
            print(f"Running {n_runs} statistical runs...")
        
        t_init = perf_counter()
        
        # Run experiments
        for run_idx in range(n_runs):
            if verbose:
                print(f"  Run {run_idx + 1}/{n_runs}")
            
            errs1[run_idx], errs2[run_idx], bell_errs[run_idx] = run_influence_k_experiment(
                run_idx=run_idx,
                Ks=Ks,
                experiments=experiments,
                q_opt=q_opt,
                group_name=group_name,
                use_logger=use_logger,
                log_every_n_steps=1,
                verbose=verbose
            )
        
        t_end = perf_counter()
        
        if verbose:
            print(f"Completed in {(t_end-t_init)/60:.2f} minutes")
        
        # Save results
        if save_results:
            file_name = results_path + f"{group_name}_data.npz"
            np.savez(
                file_name, 
                Ks=Ks, 
                Exps=experiments, 
                errs1=errs1, 
                errs2=errs2, 
                bell_errs=bell_errs
            )
            if verbose:
                print(f"Data saved as: {file_name}")
        
        # Generate plots
        if verbose:
            print("Generating plots...")
        
        skip_idx: List[int] = []
        xlabel = "K"
        
        plot_errors(
            errs1, Ks, experiments, xlabel, "Q err", 
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            errs2, Ks, experiments, xlabel, "Q err 2", 
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            bell_errs, Ks, experiments, xlabel, "Bellman err", 
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        
        # Save CSV data
        if save_results:
            csv_file = results_path + f"{group_name}_median_err.csv"
            save_error_matrix_to_csv(
                np.median(errs2, axis=0), Ks, experiments, csv_file
            )
            
            csv_file = results_path + f"{group_name}_p25_err.csv"
            save_error_matrix_to_csv(
                np.percentile(errs2, 25, axis=0), Ks, experiments, csv_file
            )
            
            csv_file = results_path + f"{group_name}_p75_err.csv"
            save_error_matrix_to_csv(
                np.percentile(errs2, 75, axis=0), Ks, experiments, csv_file
            )

    print(f"\n{'='*60}")
    print("All influence of K experiments completed!")
    if save_results:
        print(f"Results saved in: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run influence of K (filter order) experiments"
    )
    parser.add_argument(
        "--unrolls", 
        nargs="+", 
        type=int, 
        default=[5, 10, 15],
        help="List of unroll values to test (default: [5, 10, 15])"
    )
    parser.add_argument(
        "--k-values", 
        nargs="+", 
        type=int, 
        default=[1, 2, 3, 5, 10, 15],
        help="List of K values to test (default: [1, 2, 3, 5, 10, 15])"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=5,
        help="Number of statistical runs (default: 5)"
    )
    parser.add_argument(
        "--use-logger", 
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Disable saving results to files"
    )
    parser.add_argument(
        "--results-path", 
        type=str, 
        default="results/filter_order/",
        help="Directory to save results (default: results/filter_order/)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    main(
        unroll_values=args.unrolls,
        k_values=args.k_values,
        n_runs=args.runs,
        use_logger=args.use_logger,
        save_results=not args.no_save,
        results_path=args.results_path,
        verbose=not args.quiet
    )