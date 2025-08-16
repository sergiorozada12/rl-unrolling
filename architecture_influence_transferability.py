import numpy as np
from time import perf_counter
import os
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.utils import get_optimal_q, test_pol_err
import wandb

# ------------------------------
# Run configuration
# ------------------------------
verbose = True
use_logger = False
log_every_n_steps = 1
group_name = "architecture_comparison_transferability"
N_unrolls = np.arange(2, 11, 2)
SAVE = True
PATH = "./"

# ------------------------------
# Auxiliary functions
# ------------------------------
def run(g, N_unrolls, Exps, q_opt, q_opt_mirr, group_name, use_logger=True, log_every_n_steps=1, verbose=False):
    err = np.zeros((len(Exps), N_unrolls.size))
    err_tranf = np.zeros((len(Exps), N_unrolls.size))
    bell_err_tranf = np.zeros((len(Exps), N_unrolls.size))
    
    use_logger = use_logger and g == 0
    total_combinations = len(N_unrolls) * len(Exps)
    current_combination = 0
    
    print(f"\n🔄 Batch {g+1}: Starting {total_combinations} experiments (Unrolls×Arch: {len(N_unrolls)}×{len(Exps)})")
    batch_start_time = perf_counter()

    for i, n_unrolls in enumerate(N_unrolls):
        n_unrolls = int(n_unrolls)
        print(f"\n📊 Unrolls={n_unrolls} ({i+1}/{len(N_unrolls)}) - Running {len(Exps)} architecture experiments...")
        unroll_start_time = perf_counter()
        
        for j, exp in enumerate(Exps):
            current_combination += 1
            exp_start_time = perf_counter()
            
            # Progress indicator
            progress_pct = (current_combination / total_combinations) * 100
            print(f"  ⚡ [{current_combination:2d}/{total_combinations}] ({progress_pct:5.1f}%) Unrolls={n_unrolls}, Arch: {exp['name'][:30]:<30}", end=" ")
            
            env = CliffWalkingEnv()
            env_test = MirroredCliffWalkingEnv()
            
            if exp["model"] == "unroll":
                model = UnrollingPolicyIterationTrain(env=env, env_test=env_test, num_unrolls=n_unrolls, **exp["args"])
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-{n_unrolls}unrolls",
                                         group=group_name)
                else:
                    logger = False
                trainer = Trainer(max_epochs=3000, log_every_n_steps=log_every_n_steps, accelerator="cpu", logger=logger)

                trainer.fit(model)
                if use_logger:
                    wandb.finish()

                _, err[j,i] = test_pol_err(model.Pi, q_opt, mirror_env=False, device=model.device)
                _, err_tranf[j,i] = test_pol_err(model.Pi_test, q_opt_mirr, mirror_env=True, device=model.device)
                bell_err_tranf[j,i] = model.bellman_error_test.cpu().numpy()

            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env_test, goal_row=0, **exp["args"])
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-{n_unrolls}impr",
                                         group=group_name)
                else:
                    logger = False
                trainer = Trainer(max_epochs=n_unrolls, log_every_n_steps=log_every_n_steps, accelerator='cpu', logger=logger)

                trainer.fit(model)
                if use_logger:
                    wandb.finish()

                _, err[j,i] = test_pol_err(model.Pi, q_opt_mirr, mirror_env=True, device=model.device)
                err_tranf[j,i] = err[j,i]
                bell_err_tranf[j,i] = model.bellman_error.cpu().numpy()
                
            else:
                raise Exception("Unknown model")

            # Time for this experiment
            exp_time = perf_counter() - exp_start_time
            print(f"✅ {exp_time:5.1f}s (Err: {err[j,i]:.3f}, Trans: {err_tranf[j,i]:.3f})")

            if verbose:
                print(f"    📈 Detailed: Unrolls={n_unrolls}, Architecture={exp.get('args', {}).get('architecture_type', 'N/A')}, Err={err[j,i]:.3f}, Err_tranf={err_tranf[j,i]:.3f}, Bell_err={bell_err_tranf[j,i]:.3f}")
        
        # Summary for this unroll setting
        unroll_time = perf_counter() - unroll_start_time
        print(f"  ✨ Unrolls={n_unrolls} completed in {unroll_time:.1f}s")
    
    # Batch summary
    batch_time = perf_counter() - batch_start_time
    print(f"\n🎉 Batch {g+1} completed in {batch_time/60:.1f} minutes!")
    
    return err, err_tranf, bell_err_tranf

# ------------------------------
# Define Architecture Experiments
# Configuration: WS-5, with detach, init random
# ------------------------------
Exps_architectures = []

# Architecture 2: separate parameters for r and q_0 terms
Exps_architectures.append({
    "model": "unroll",
    "args": {
        "K": 5,
        "tau": 5,
        "lr": 5e-3,
        "weight_sharing": True,
        "loss_type": "original_with_detach",
        "init_q": "random",
        "architecture_type": 2
    },
    "fmt": "s-",
    "name": "arch2-WS5-with_detach-random"
})

# Architecture 3: joint filter for concatenated [r; q_0]
Exps_architectures.append({
    "model": "unroll",
    "args": {
        "K": 5,
        "tau": 5,
        "lr": 5e-3,
        "weight_sharing": True,
        "loss_type": "original_with_detach",
        "init_q": "random",
        "architecture_type": 3
    },
    "fmt": "o-",
    "name": "arch3-WS5-with_detach-random"
})

# Architecture 5: matrix filters and final linear layer
Exps_architectures.append({
    "model": "unroll",
    "args": {
        "K": 5,
        "tau": 5,
        "lr": 5e-3,
        "weight_sharing": True,
        "loss_type": "original_with_detach",
        "init_q": "random",
        "architecture_type": 5
    },
    "fmt": "^-",
    "name": "arch5-WS5-with_detach-random"
})

print(f"Total architecture experiments: {len(Exps_architectures)}")
for i, exp in enumerate(Exps_architectures):
    print(f"{i}: {exp['name']}")

# ------------------------------
# Main execution
# ------------------------------
def main():
    print("="*70)
    print("🚀 STARTING ARCHITECTURE COMPARISON - TRANSFERABILITY EXPERIMENTS")
    print("="*70)
    print(f"🏗️  Architectures: 2, 3, 5")
    print(f"⚙️  Configuration: WS-5, with_detach, init_random")
    print(f"📋 Experiments to run: {len(Exps_architectures)}")
    print(f"🔢 Unroll values: {N_unrolls}")
    print(f"🏷️  Group name: {group_name}")
    print(f"🔄 Number of runs: 15")
    print(f"📊 Total experiments: {len(Exps_architectures) * len(N_unrolls) * 15} (Arch×Unrolls×Runs)")
    print("="*70)
    
    # Get optimal Q for both environments
    print("🎯 Computing optimal Q for original environment...")
    q_opt = get_optimal_q(mirror_env=False, use_logger=use_logger, log_every_n_steps=log_every_n_steps, 
                          group_name=group_name)
    print("✅ Optimal Q for original environment computed!")

    print("🎯 Computing optimal Q for mirrored environment...")
    q_opt_mirr = get_optimal_q(mirror_env=True, use_logger=use_logger, log_every_n_steps=log_every_n_steps, 
                               group_name=group_name)
    print("✅ Optimal Q for mirrored environment computed!")

    # Run experiments
    n_runs = 15
    errs_arch = np.zeros((n_runs, len(Exps_architectures), N_unrolls.size))
    errs_trans_arch = np.zeros((n_runs, len(Exps_architectures), N_unrolls.size))
    bell_errs_arch = np.zeros((n_runs, len(Exps_architectures), N_unrolls.size))

    # Run experiments with full tracking
    total_start_time = perf_counter()
    
    for g in range(n_runs):
        run_start_time = perf_counter()
        
        print(f"\n{'='*50}")
        print(f"🏃 BATCH {g+1}/{n_runs} - ARCHITECTURE COMPARISON")
        print(f"{'='*50}")
        
        errs_arch[g], errs_trans_arch[g], bell_errs_arch[g] = run(
            g, N_unrolls, Exps_architectures, q_opt, q_opt_mirr, group_name, use_logger, log_every_n_steps, verbose
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
    print(f"🎉 ALL ARCHITECTURE EXPERIMENTS COMPLETED in {total_time/60:.1f} minutes!")
    print("="*70)

    # ------------------------------
    # Save results
    # ------------------------------
    if SAVE:
        file_name = os.path.join(PATH, f"{group_name}_architectures_data.npz")
        np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps_architectures,
                 errs=errs_arch, errs_trans=errs_trans_arch, bell_errs=bell_errs_arch)
        print(f"📁 Architecture comparison data saved as: {file_name}")

if __name__ == "__main__":
    main()