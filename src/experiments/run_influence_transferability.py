#!/usr/bin/env python3
"""
Runner for transferability experiments.

This script reproduces the transferability experiments (Figure B equivalent)
testing how well models transfer between different environments.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from time import perf_counter
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    get_optimal_q, 
    run_transfer_experiment, 
    plot_errors, 
    save_error_matrix_to_csv
)


def main(
    k_values: List[int] = [3, 5, 10, 15],
    unroll_range: tuple = (2, 11, 2),
    n_runs: int = 5,
    use_logger: bool = False,
    save_results: bool = True,
    results_path: str = "results/transfer/",
    verbose: bool = True
) -> None:
    """
    Run transferability experiments for different K values.
    
    Args:
        k_values: List of K values to test
        unroll_range: Range of unrolls to test (start, stop, step)
        n_runs: Number of statistical runs
        use_logger: Whether to log to wandb
        save_results: Whether to save results to files
        results_path: Directory to save results
        verbose: Whether to print progress
    """
    
    print("Running transferability experiments")
    print("=" * 50)
    print(f"K values: {k_values}")
    print(f"Unroll range: {unroll_range}")
    print(f"Number of runs: {n_runs}")
    print()
    
    # Create results directory
    if save_results:
        os.makedirs(results_path, exist_ok=True)
    
    N_unrolls = np.arange(*unroll_range)
    
    for K in k_values:
        print(f"\n{'='*20} K = {K} {'='*20}")
        
        group_name = f"transfer-K{K}"
        
        # Define experiments for this K value
        experiments: List[Dict[str, Any]] = [
            {
                "model": "pol-it", 
                "args": {"max_eval_iters": 1}, 
                "fmt": "^-", 
                "name": "val-it"
            },
            {
                "model": "pol-it", 
                "args": {"max_eval_iters": K}, 
                "fmt": "x-", 
                "name": f"pol-it-{K}eval"
            },
            {
                "model": "unroll", 
                "args": {
                    "K": K, 
                    "tau": 5, 
                    "lr": 5e-3, 
                    "weight_sharing": True
                }, 
                "fmt": "o-", 
                "name": f"unr-K{K}-WS"
            },
            # Optionally add non-weight-sharing version
            # {
            #     "model": "unroll",
            #     "args": {
            #         "K": K,
            #         "tau": 5,
            #         "lr": 5e-3,
            #         "weight_sharing": False
            #     },
            #     "fmt": "o--",
            #     "name": f"unr-K{K}-no-share"
            # },
        ]
        # Get optimal Q functions for both environments
        if verbose:
            print("Computing optimal Q-values for both environments...")

        q_opt = get_optimal_q(
            mirror_env=False, 
            use_logger=use_logger, 
            log_every_n_steps=1, 
            group_name=group_name
        )
        q_opt_mirr = get_optimal_q(
            mirror_env=True, 
            use_logger=use_logger, 
            log_every_n_steps=1, 
            group_name=group_name
        )

        # Initialize result arrays
        errs = np.zeros((n_runs, len(experiments), N_unrolls.size))
        errs_trans = np.zeros((n_runs, len(experiments), N_unrolls.size))
        bell_errs = np.zeros((n_runs, len(experiments), N_unrolls.size))

        if verbose:
            print(f"Running {n_runs} statistical runs...")

        t_init = perf_counter()

        # Run experiments
        for run_idx in range(n_runs):
            if verbose:
                print(f"  Run {run_idx + 1}/{n_runs}")
            
            errs[run_idx], errs_trans[run_idx], bell_errs[run_idx] = run_transfer_experiment(
                run_idx=run_idx,
                N_unrolls=N_unrolls,
                experiments=experiments,
                q_opt=q_opt,
                q_opt_mirr=q_opt_mirr,
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
                N_unrolls=N_unrolls, 
                Exps=experiments, 
                errs=errs, 
                errs_trans=errs_trans, 
                bell_errs=bell_errs
            )
            if verbose:
                print(f"Data saved as: {file_name}")

        # Generate plots
        if verbose:
            print("Generating plots...")

        skip_idx: List[int] = []
        xlabel = "Number of unrolls"

        plot_errors(
            errs, N_unrolls, experiments, xlabel, "Q err (training)", 
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            errs_trans, N_unrolls, experiments, xlabel, "Q err (Transfer)", 
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            bell_errs, N_unrolls, experiments, xlabel, "Bellman err", 
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )

        # Save CSV data  
        if save_results:
            csv_file = results_path + f"{group_name}_transfer_median_err.csv"
            save_error_matrix_to_csv(
                np.median(errs_trans, axis=0), N_unrolls, experiments, csv_file
            )
            
            csv_file = results_path + f"{group_name}_transfer_p25_err.csv"
            save_error_matrix_to_csv(
                np.percentile(errs_trans, 25, axis=0), N_unrolls, experiments, csv_file
            )
            
            csv_file = results_path + f"{group_name}_transfer_p75_err.csv"
            save_error_matrix_to_csv(
                np.percentile(errs_trans, 75, axis=0), N_unrolls, experiments, csv_file
            )

        print(f"\n{'='*50}")
        print("All transferability experiments completed!")
        if save_results:
            print(f"Results saved in: {results_path}")


def plot_combined_results(
    k_values: List[int], 
    results_path: str = "results/transfer/",
    save_combined: bool = True
) -> None:
    """
    Load and plot combined results from all K values.
    
    Args:
        k_values: List of K values that were tested
        results_path: Directory containing results
        save_combined: Whether to save combined CSV files
    """
    print("\nGenerating combined plots...")
    
    files = [f"transfer-K{K}_data.npz" for K in k_values]
    
    all_exps = []
    errs_list = []
    errs_trans_list = []
    bell_errs_list = []
    
    for file in files:
        file_path = results_path + file
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping...")
            continue
            
        data = np.load(file_path, allow_pickle=True)
        
        if 'N_unrolls' not in locals():
            N_unrolls = data["N_unrolls"]
        
        all_exps += list(data["Exps"])
        errs_list.append(data["errs"])
        errs_trans_list.append(data["errs_trans"])
        bell_errs_list.append(data["bell_errs"])
    
    if not errs_list:
        print("No data files found for combined plotting.")
        return
    
    # Concatenate all data
    errs = np.concatenate(errs_list, axis=1)
    errs_trans = np.concatenate(errs_trans_list, axis=1)
    bell_errs = np.concatenate(bell_errs_list, axis=1)
    
    # Plot combined results (skip policy evaluation baselines)
    skip_idx = [i for i, exp in enumerate(all_exps) if "val-it" in exp.get("name", "")]
    xlabel = "Number of unrolls"
    
    plot_errors(
        errs_trans, N_unrolls, all_exps, xlabel, "Q Err (Transfer)", 
        skip_idx=skip_idx, agg="median", deviation='prctile'
    )
    plot_errors(
        errs, N_unrolls, all_exps, xlabel, "Q Err (Training)", 
        skip_idx=skip_idx, agg="median", deviation='prctile'
    )
    plot_errors(
        bell_errs, N_unrolls, all_exps, xlabel, "Bellman Err (Transfer)", 
        skip_idx=skip_idx, agg="median", deviation='prctile'
    )
    
    # Save combined results
    if save_combined:
        csv_file = results_path + "transfer_all_median_err.csv"
        save_error_matrix_to_csv(
            np.median(errs_trans, axis=0), N_unrolls, all_exps, csv_file
        )
        
        csv_file = results_path + "transfer_all_p25_err.csv"
        save_error_matrix_to_csv(
            np.percentile(errs_trans, 25, axis=0), N_unrolls, all_exps, csv_file
        )
        
        csv_file = results_path + "transfer_all_p75_err.csv"
        save_error_matrix_to_csv(
            np.percentile(errs_trans, 75, axis=0), N_unrolls, all_exps, csv_file
        )
        
        print(f"Combined results saved in: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run transferability experiments"
    )
    parser.add_argument(
        "--k-values", 
        nargs="+", 
        type=int, 
        default=[3, 5, 10, 15],
        help="List of K values to test (default: [3, 5, 10, 15])"
    )
    parser.add_argument(
        "--unroll-start", 
        type=int, 
        default=2,
        help="Start of unroll range (default: 2)"
    )
    parser.add_argument(
        "--unroll-stop", 
        type=int, 
        default=11,
        help="Stop of unroll range (default: 11)"
    )
    parser.add_argument(
        "--unroll-step", 
        type=int, 
        default=2,
        help="Step of unroll range (default: 2)"
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
        default="results/transfer/",
        help="Directory to save results (default: results/transfer/)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--plot-combined", 
        action="store_true",
        help="Only generate combined plots from existing data"
    )
    
    args = parser.parse_args()
    
    if args.plot_combined:
        # Only plot combined results from existing data
        plot_combined_results(
            k_values=args.k_values,
            results_path=args.results_path,
            save_combined=not args.no_save
        )
    else:
        # Run full experiments
        main(
            k_values=args.k_values,
            unroll_range=(args.unroll_start, args.unroll_stop, args.unroll_step),
            n_runs=args.runs,
            use_logger=args.use_logger,
            save_results=not args.no_save,
            results_path=args.results_path,
            verbose=not args.quiet
        )
        
        # Optionally plot combined results
        if not args.no_save:
            plot_combined_results(
                k_values=args.k_values,
                results_path=args.results_path,
                save_combined=True
            )