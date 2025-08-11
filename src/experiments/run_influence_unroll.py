#!/usr/bin/env python3
"""
Runner for influence of number of unrolls experiments.

This script reproduces the number of unrolls experiments (Figure C equivalent)
testing different numbers of unroll steps with various model configurations.
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
    run_unroll_experiment, 
    plot_errors, 
    save_error_matrix_to_csv
)


def main(
    k_values: List[int] = [5, 10, 15],
    unroll_range: tuple = (2, 11, 2),
    n_runs: int = 15,
    use_logger: bool = False,
    save_results: bool = True,
    results_path: str = "results/n_unrolls/",
    verbose: bool = True
) -> None:
    """
    Run influence of number of unrolls experiments for different K values.
    
    Args:
        k_values: List of K values to test
        unroll_range: Range of unrolls to test (start, stop, step)
        n_runs: Number of statistical runs
        use_logger: Whether to log to wandb
        save_results: Whether to save results to files
        results_path: Directory to save results
        verbose: Whether to print progress
    """
    
    print("Running influence of number of unrolls experiments")
    print("=" * 60)
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
        
        group_name = f"n_unrolls-K{K}"
        
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
            {
                "model": "unroll", 
                "args": {
                    "K": K, 
                    "tau": 5, 
                    "lr": 5e-3, 
                    "weight_sharing": False
                }, 
                "fmt": "o--", 
                "name": f"unr-K{K}"
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
        errs1 = np.zeros((n_runs, len(experiments), N_unrolls.size))
        errs2 = np.zeros((n_runs, len(experiments), N_unrolls.size))
        bell_errs = np.zeros((n_runs, len(experiments), N_unrolls.size))

        if verbose:
            print(f"Running {n_runs} statistical runs...")

        t_init = perf_counter()

        # Run experiments
        for run_idx in range(n_runs):
            if verbose:
                print(f"  Run {run_idx + 1}/{n_runs}")

            errs1[run_idx], errs2[run_idx], bell_errs[run_idx] = run_unroll_experiment(
                run_idx=run_idx,
                N_unrolls=N_unrolls,
                experiments=experiments,
                q_opt=q_opt,
                group_name=group_name,
                use_logger=use_logger,
                log_every_n_steps=1,
                verbose=verbose
            )

        t_end = perf_counter()

        if verbose:
            print(f"Completed in {(t_end - t_init) / 60:.2f} minutes")

        # Save results
        if save_results:
            file_name = results_path + f"{group_name}_data.npz"
            np.savez(
                file_name,
                N_unrolls=N_unrolls,
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
        xlabel = "Number of unrolls"

        plot_errors(
            errs1, N_unrolls, experiments, xlabel, "Q err",
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            errs2, N_unrolls, experiments, xlabel, "Q err 2",
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            bell_errs, N_unrolls, experiments, xlabel, "Bellman err",
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )

        # Save CSV data
        if save_results:
            csv_file = results_path + f"{group_name}_median_err.csv"
            save_error_matrix_to_csv(
                np.median(errs2, axis=0), N_unrolls, experiments, csv_file
            )

            csv_file = results_path + f"{group_name}_p25_err.csv"
            save_error_matrix_to_csv(
                np.percentile(errs2, 25, axis=0), N_unrolls, experiments, csv_file
            )

            csv_file = results_path + f"{group_name}_p75_err.csv"
            save_error_matrix_to_csv(
                np.percentile(errs2, 75, axis=0), N_unrolls, experiments, csv_file
            )

    print(f"\n{'='*60}")
    print("All number of unrolls experiments completed!")
    if save_results:
        print(f"Results saved in: {results_path}")


def plot_combined_results(
    k_values: List[int],
    results_path: str = "results/n_unrolls/",
    save_combined: bool = True,
    specific_comparison: bool = False
) -> None:
    """
    Load and plot combined results from all K values.

    Args:
        k_values: List of K values that were tested
        results_path: Directory containing results
        save_combined: Whether to save combined CSV files
        specific_comparison: Whether to generate specific comparison plots
    """
    print("\nGenerating combined plots...")

    files = [f"n_unrolls-K{K}_data.npz" for K in k_values]

    all_exps = []
    errs1_list = []
    errs2_list = []
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
        errs1_list.append(data["errs1"])
        errs2_list.append(data["errs2"])
        bell_errs_list.append(data["bell_errs"])

    if not errs1_list:
        print("No data files found for combined plotting.")
        return

    # Concatenate all data
    errs1 = np.concatenate(errs1_list, axis=1)
    errs2 = np.concatenate(errs2_list, axis=1)
    bell_errs = np.concatenate(bell_errs_list, axis=1)

    xlabel = "Number of unrolls"

    if specific_comparison:

        # Specific comparison plots (similar to original notebooks)

        # 1. Policy iteration vs Unrolling with/without WS (only 5 and 10 unrolls)
        skip_idx_1 = [i for i, exp in enumerate(all_exps)
                      if "K15" in exp.get("name", "") or "val-it" in exp.get("name", "")]

        print("Plotting: Policy iteration vs Unrolling with/without WS")
        plot_errors(
            errs2, N_unrolls, all_exps, xlabel, "Q err 2",
            skip_idx=skip_idx_1, agg="median", deviation='prctile'
        )

        # 2. Policy iteration vs Unrolling with WS only
        skip_idx_2 = [i for i, exp in enumerate(all_exps)
                      if not exp.get("name", "").endswith("-WS") and "pol-it" not in exp.get("name", "")]

        print("Plotting: Policy iteration vs Unrolling with WS only")
        plot_errors(
            errs2, N_unrolls, all_exps, xlabel, "Q err 2",
            skip_idx=skip_idx_2, agg="median", deviation='prctile'
        )

    else:
        # General plots with all experiments
        skip_idx: List[int] = []

        plot_errors(
            errs1, N_unrolls, all_exps, xlabel, "Q err",
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            errs2, N_unrolls, all_exps, xlabel, "Q err 2",
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )
        plot_errors(
            bell_errs, N_unrolls, all_exps, xlabel, "Bellman err",
            skip_idx=skip_idx, agg="median", deviation='prctile'
        )

    # Save combined results
    if save_combined:
        csv_file = results_path + "n_unrolls_all_median_err.csv"
        save_error_matrix_to_csv(
            np.median(errs2, axis=0), N_unrolls, all_exps, csv_file
        )

        csv_file = results_path + "n_unrolls_all_p25_err.csv"
        save_error_matrix_to_csv(
            np.percentile(errs2, 25, axis=0), N_unrolls, all_exps, csv_file
        )

        csv_file = results_path + "n_unrolls_all_p75_err.csv"
        save_error_matrix_to_csv(
            np.percentile(errs2, 75, axis=0), N_unrolls, all_exps, csv_file
        )

        print(f"Combined results saved in: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run influence of number of unrolls experiments"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10, 15],
        help="List of K values to test (default: [5, 10, 15])"
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
        default=15,
        help="Number of statistical runs (default: 15)"
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
        default="results/n_unrolls/",
        help="Directory to save results (default: results/n_unrolls/)"
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
    parser.add_argument(
        "--specific-comparison", 
        action="store_true",
        help="Generate specific comparison plots (like in original notebooks)"
    )

    args = parser.parse_args()

    if args.plot_combined:
        # Only plot combined results from existing data
        plot_combined_results(
            k_values=args.k_values,
            results_path=args.results_path,
            save_combined=not args.no_save,
            specific_comparison=args.specific_comparison
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
                save_combined=True,
                specific_comparison=args.specific_comparison
            )