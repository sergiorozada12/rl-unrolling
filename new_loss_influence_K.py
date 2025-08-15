import numpy as np
from time import perf_counter
import os
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.utils import get_optimal_q, test_pol_err
import wandb

# ------------------------------
# Execution configuration
# ------------------------------
verbose = True
use_logger = False
log_every_n_steps = 1
num_unrolls = 5
group_name = f"new_loss_init_filter_order-{num_unrolls}"
Ks = np.array([1, 2, 3, 5, 10, 15])
SAVE = True
PATH = "./"

# ------------------------------
# Define combinations
# ------------------------------
loss_types = ["original_with_detach", "original_no_detach", "max_with_detach", "max_no_detach"]
init_types = ["zeros", "random"]

# Create experiments for all combinations
Exps_new = []

# Add all combinations of loss functions and initializations
for loss_type in loss_types:
    for init_type in init_types:
        exp_name = f"unr-{num_unrolls}unrolls-{loss_type}-{init_type}"
        Exps_new.append({
            "model": "unroll",
            "args": {
                "num_unrolls": num_unrolls,
                "tau": 5,
                "lr": 5e-3,
                "weight_sharing": True,
                "loss_type": loss_type,
                "init_q": init_type
            },
            "fmt": "s-",
            "name": exp_name
        })

print(f"Total experiments: {len(Exps_new)}")
for i, exp in enumerate(Exps_new):
    print(f"{i}: {exp['name']}")

# ------------------------------
# Run function adapted for K with detailed progress
# ------------------------------
def run(g, Ks, Exps, q_opt, group_name, use_logger=True, log_every_n_steps=1, verbose=False):
    err1 = np.zeros((len(Exps), Ks.size))
    err2 = np.zeros((len(Exps), Ks.size))
    bell_err = np.zeros((len(Exps), Ks.size))
    
    use_logger = use_logger and g == 0
    total_combinations = len(Ks) * len(Exps)
    current_combination = 0
    
    print(f"\n🔄 Batch {g+1}: Starting {total_combinations} experiments (K×Exp: {len(Ks)}×{len(Exps)})")
    batch_start_time = perf_counter()

    for i, K in enumerate(Ks):
        K = int(K)
        print(f"\n📊 K={K} ({i+1}/{len(Ks)}) - Running {len(Exps)} experiments...")
        k_start_time = perf_counter()
        
        for j, exp in enumerate(Exps):
            current_combination += 1
            exp_start_time = perf_counter()
            
            # Progress indicator
            progress_pct = (current_combination / total_combinations) * 100
            print(f"  ⚡ [{current_combination:2d}/{total_combinations}] ({progress_pct:5.1f}%) K={K}, Exp: {exp['name'][:25]:<25}", end=" ")
            
            env = CliffWalkingEnv()

            if exp["model"] == "unroll":
                model = UnrollingPolicyIterationTrain(env=env, env_test=env, K=K, **exp["args"])
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-K{K}",
                                         group=group_name)
                else:
                    logger = False
                trainer = Trainer(max_epochs=3000, log_every_n_steps=log_every_n_steps, accelerator="cpu", logger=logger)

            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env, max_eval_iters=K)
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-{K}impr",
                                         group=group_name)
                else:
                    logger = False
                trainer = Trainer(max_epochs=exp['args']['max_epochs'], log_every_n_steps=log_every_n_steps, 
                                accelerator='cpu', logger=logger)
            else:
                raise ValueError(f"Unknown model type: {exp['model']}")

            trainer.fit(model)
            wandb.finish()

            err1[j,i], err2[j,i] = test_pol_err(model.Pi, q_opt)
            bell_err[j,i] = model.bellman_error.cpu().numpy()

            # Time for this experiment
            exp_time = perf_counter() - exp_start_time
            print(f"✅ {exp_time:5.1f}s (Err1: {err1[j,i]:.3f})")
            
            if verbose:
                print(f"    📈 Detailed: K={K}, Model={exp['name']}, Err1={err1[j,i]:.3f}, Bell_err={bell_err[j,i]:.3f}")
        
        # Summary for this K
        k_time = perf_counter() - k_start_time
        print(f"  ✨ K={K} completed in {k_time:.1f}s")
    
    # Batch summary
    batch_time = perf_counter() - batch_start_time
    print(f"\n🎉 Batch {g+1} completed in {batch_time/60:.1f} minutes!")
    
    return err1, err2, bell_err

# ------------------------------
# Main execution with detailed tracking
# ------------------------------
def main():
    print("="*70)
    print("🚀 STARTING FILTER ORDER EXPERIMENTS")
    print("="*70)
    print(f"📋 Experiments to run: {len(Exps_new)}")
    print(f"🔢 K values: {Ks}")
    print(f"🏷️  Group name: {group_name}")
    print(f"🔄 Number of runs: 5")
    print(f"📊 Total experiments: {len(Exps_new) * len(Ks) * 5} (Exp×K×Runs)")
    print("="*70)
    
    # Get optimal Q
    print("🎯 Computing optimal Q...")
    q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)
    print("✅ Optimal Q computed!")

    # Set up experiments
    n_runs = 5
    errs1_new = np.zeros((n_runs, len(Exps_new), Ks.size))
    errs2_new = np.zeros((n_runs, len(Exps_new), Ks.size))
    bell_errs_new = np.zeros((n_runs, len(Exps_new), Ks.size))

    # Run experiments with full tracking
    total_start_time = perf_counter()
    
    for g in range(n_runs):
        run_start_time = perf_counter()
        
        print(f"\n{'='*50}")
        print(f"🏃 BATCH {g+1}/{n_runs}")
        print(f"{'='*50}")
        
        errs1_new[g], errs2_new[g], bell_errs_new[g] = run(
            g, Ks, Exps_new, q_opt, group_name, use_logger, log_every_n_steps, verbose
        )
        
        # Progress summary after each batch
        run_time = perf_counter() - run_start_time
        total_elapsed = perf_counter() - total_start_time
        
        # Time estimates
        avg_time_per_run = total_elapsed / (g + 1)
        estimated_total_time = avg_time_per_run * n_runs
        remaining_time = estimated_total_time - total_elapsed
        
        print(f"\n📊 BATCH {g+1} SUMMARY:")
        print(f"  ⏱️  This batch: {run_time/60:.1f} minutes")
        print(f"  🕐 Total elapsed: {total_elapsed/60:.1f} minutes")
        print(f"  ⏳ Estimated remaining: {remaining_time/60:.1f} minutes")
        print(f"  🎯 Estimated completion: {estimated_total_time/60:.1f} minutes total")
        print(f"  📈 Progress: {((g+1)/n_runs)*100:.1f}%")
        
        if g < n_runs - 1:  # Not the last iteration
            print(f"\n⏭️  Next: Batch {g+2}/{n_runs}")
        
    total_time = perf_counter() - total_start_time
    print(f'\n{"="*70}')
    print(f"🎉 ALL EXPERIMENTS COMPLETED in {total_time/60:.1f} minutes!")
    print("="*70)

if __name__ == "__main__":
    main()
