#!/usr/bin/env python3
"""
Main entry point for BellNet experiments.

This script provides a unified interface to run all types of experiments:
- Influence of K (graph filter order)
- Transferability experiments
- Influence of number of unrolls
- BN-5 WS specific experiments

It can generate all plots from the paper and manage experimental configurations.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "experiments"))

from config import get_config, update_config, Config


def run_influence_k_experiments(args, config: Config) -> None:
    """Run influence of K experiments."""
    print("Running influence of K experiments...")
    
    # Import here to avoid circular imports
    from src.experiments.run_influence_k import main as run_k_main
    
    run_k_main(
        unroll_values=args.unroll_values,
        k_values=args.k_values,
        n_runs=args.runs,
        use_logger=args.use_logger,
        save_results=not args.no_save,
        results_path=args.results_path or "results/filter_order/",
        verbose=not args.quiet
    )


def run_transferability_experiments(args, config: Config) -> None:
    """Run transferability experiments."""
    print("Running transferability experiments...")
    
    from src.experiments.run_influence_transferability import main as run_transfer_main
    
    run_transfer_main(
        k_values=args.k_values,
        unroll_range=(args.unroll_start, args.unroll_stop, args.unroll_step),
        n_runs=args.runs,
        use_logger=args.use_logger,
        save_results=not args.no_save,
        results_path=args.results_path or "results/transfer/",
        verbose=not args.quiet
    )


def run_unroll_experiments(args, config: Config) -> None:
    """Run influence of number of unrolls experiments."""
    print("Running influence of number of unrolls experiments...")
    
    from src.experiments.run_influence_unroll import main as run_unroll_main
    
    run_unroll_main(
        k_values=args.k_values,
        unroll_range=(args.unroll_start, args.unroll_stop, args.unroll_step),
        n_runs=args.runs,
        use_logger=args.use_logger,
        save_results=not args.no_save,
        results_path=args.results_path or "results/n_unrolls/",
        verbose=not args.quiet
    )


def run_bn5_ws_experiments(args, config: Config) -> None:
    """Run BN-5 WS specific experiments."""
    print("Running BN-5 WS specific experiments...")
    
    from experiments.run_specific_experiments import (
        run_bn5_ws_experiments as run_bn5_main,
        run_transferability_bn5_ws,
        plot_bn5_ws_results
    )
    
    # Main experiments
    errs1, errs2, bell_errs = run_bn5_main(
        use_logger=args.use_logger,
        verbose=not args.quiet,
        save_results=not args.no_save,
        results_path=args.results_path or "results/bn5_ws_specific/"
    )
    
    # Transferability experiments
    errs_train, errs_transfer, bell_errs_transfer = run_transferability_bn5_ws(
        use_logger=args.use_logger,
        verbose=not args.quiet,
        save_results=not args.no_save,
        results_path=args.results_path or "results/bn5_ws_specific/"
    )


def run_single_experiment(args, config: Config) -> None:
    """Run a single experiment using the config-based training functions."""
    print("Running single experiment...")
    
    # Import training functions from current directory
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    if args.model_type == "policy_iteration":
        # Create a simple policy iteration runner
        from src.algorithms import PolicyIterationTrain
        from src import CliffWalkingEnv
        from pytorch_lightning import Trainer
        from lightning.pytorch.loggers import WandbLogger
        
        env = CliffWalkingEnv()
        model = PolicyIterationTrain(
            env,
            gamma=0.99,
            max_eval_iters=args.max_eval_iters or 10,
            goal_row=3
        )
        
        logger = WandbLogger(
            project="rl-unrolling",
            name=f"single-pol-it-{args.max_eval_iters or 10}eval",
            group=args.group or "single-experiment"
        ) if args.use_logger else False
        
        trainer = Trainer(
            max_epochs=args.max_epochs or 20,
            log_every_n_steps=1,
            accelerator='cpu',
            logger=logger,
        )
        
        trainer.fit(model, train_dataloaders=None)
        
    elif args.model_type == "unrolling":
        from src.algorithms import UnrollingPolicyIterationTrain
        from src import CliffWalkingEnv
        from pytorch_lightning import Trainer
        from lightning.pytorch.loggers import WandbLogger
        
        env = CliffWalkingEnv()
        model = UnrollingPolicyIterationTrain(
            env=env,
            env_test=env,
            K=args.k_value or config.model.K,
            num_unrolls=args.num_unrolls or config.model.num_unrolls,
            gamma=0.99,
            lr=args.lr or config.training.lr,
            tau=args.tau or config.model.tau,
            beta=args.beta or config.model.beta,
            N=args.dataset_size or config.training.N,
            weight_sharing=args.weight_sharing,
            init_q=args.init_q or config.model.init_q,
        )
        
        logger = WandbLogger(
            project="rl-unrolling",
            name=f"single-unrl-K{args.k_value or config.model.K}-{args.num_unrolls or config.model.num_unrolls}unr",
            group=args.group or "single-experiment"
        ) if args.use_logger else False
        
        trainer = Trainer(
            max_epochs=5000, 
            log_every_n_steps=1,
            accelerator="cpu",
            logger=logger,
        )
        
        trainer.fit(model)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def main() -> None:
    """Main function with argument parsing and experiment dispatch."""
    parser = argparse.ArgumentParser(
        description="BellNet experiments runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python main.py all --runs 5
  
  # Run influence of K experiments
  python main.py influence-k --k-values 1 2 3 5 10 15
  
  # Run transferability experiments
  python main.py transferability --k-values 5 10 15
  
  # Run unroll experiments
  python main.py unroll --k-values 5 10 15
  
  # Run BN-5 WS specific experiments
  python main.py bn5-ws
  
  # Run single unrolling experiment
  python main.py single --model-type unrolling --k-value 10 --num-unrolls 5 --weight-sharing
  
  # Use specific config
  python main.py single --config fast_training --model-type policy_iteration
        """
    )
    
    # Main command
    parser.add_argument(
        "command",
        choices=["all", "influence-k", "transferability", "unroll", "bn5-ws", "single"],
        help="Experiment type to run"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Configuration name to use (default: default)"
    )
    
    # General experiment parameters
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
        help="Directory to save results (default: auto-detect by experiment type)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce output verbosity"
    )
    
    # Experiment-specific parameters
    parser.add_argument(
        "--k-values", 
        nargs="+", 
        type=int,
        help="List of K values to test (default: varies by experiment)"
    )
    parser.add_argument(
        "--unroll-values", 
        nargs="+", 
        type=int,
        help="List of unroll values to test (for influence-k only)"
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
    
    # Single experiment parameters
    parser.add_argument(
        "--model-type",
        choices=["policy_iteration", "unrolling"],
        help="Model type for single experiment"
    )
    parser.add_argument("--k-value", type=int, help="K value for single experiment")
    parser.add_argument("--num-unrolls", type=int, help="Number of unrolls for single experiment")
    parser.add_argument("--tau", type=float, help="Tau parameter")
    parser.add_argument("--beta", type=float, help="Beta parameter")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--dataset-size", type=int, help="Dataset size N")
    parser.add_argument("--weight-sharing", action="store_true", help="Enable weight sharing")
    parser.add_argument("--group", type=str, help="Experiment group name")
    parser.add_argument("--init-q", type=str, choices=["zeros", "ones", "random"], help="Q initialization method")
    parser.add_argument("--max-eval-iters", type=int, help="Max evaluation iterations (policy iteration)")
    parser.add_argument("--max-epochs", type=int, help="Max epochs (policy iteration)")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = get_config(args.config)
        print(f"Using configuration: {args.config}")
    except KeyError:
        print(f"Warning: Configuration '{args.config}' not found. Using default.")
        config = get_config('default')
    
    # Set default K values based on experiment type if not provided
    if not args.k_values:
        if args.command == "influence-k":
            args.k_values = [1, 2, 3, 5, 10, 15]
        elif args.command == "transferability":
            args.k_values = [3, 5, 10, 15]
        elif args.command == "unroll":
            args.k_values = [5, 10, 15]
        elif args.command == "bn5-ws":
            args.k_values = [5]  # Only BN-5
        elif args.command == "all":
            args.k_values = [1, 2, 3, 5, 10, 15]
    
    # Set default unroll values for influence-k
    if args.command == "influence-k" and not args.unroll_values:
        args.unroll_values = [5, 10, 15]
    
    # Dispatch to appropriate experiment runner
    if args.command == "all":
        print("Running all experiments...")
        print("=" * 60)
        
        # Run all experiment types
        run_influence_k_experiments(args, config)
        run_transferability_experiments(args, config)
        run_unroll_experiments(args, config)
        run_bn5_ws_experiments(args, config)
        
        print("\n" + "=" * 60)
        print("All experiments completed!")
        
    elif args.command == "influence-k":
        run_influence_k_experiments(args, config)
        
    elif args.command == "transferability":
        run_transferability_experiments(args, config)
        
    elif args.command == "unroll":
        run_unroll_experiments(args, config)
        
    elif args.command == "bn5-ws":
        run_bn5_ws_experiments(args, config)
        
    elif args.command == "single":
        if not args.model_type:
            parser.error("--model-type is required for single experiments")
        run_single_experiment(args, config)
    
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()