import numpy as np
from time import perf_counter
import os
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
import wandb

# ------------------------------
# Run configuration
# ------------------------------
verbose = True
use_logger = False
log_every_n_steps = 1
K = 5
group_name = f"new_loss_init_K{K}"
N_unrolls = np.arange(2, 11, 2)
SAVE = True
PATH = "./" 

# ------------------------------
# Definition of combinations
# ------------------------------
loss_types = ["original_with_detach", "original_no_detach", "max_with_detach", "max_no_detach"]
init_types = ["zeros", "random"]

Exps_new = []
for loss_type in loss_types:
    for init_type in init_types:
        exp_name = f"unr-K{K}-{loss_type}-{init_type}"
        Exps_new.append({
            "model": "unroll",
            "args": {
                "K": K,
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
# Required functions
# ------------------------------
from src.utils import get_optimal_q, test_pol_err

def run(g, N_unrolls, Exps, q_opt, group_name, use_logger=True, log_every_n_steps=1, verbose=False):
    err1 = np.zeros((len(Exps), N_unrolls.size))
    err2 = np.zeros((len(Exps), N_unrolls.size))
    bell_err = np.zeros((len(Exps), N_unrolls.size))
    
    use_logger = use_logger and g == 0

    for i, n_unrolls in enumerate(N_unrolls):
        n_unrolls = int(n_unrolls)
        for j, exp in enumerate(Exps):
            env = CliffWalkingEnv()

            if exp["model"] == "unroll":
                model = UnrollingPolicyIterationTrain(env=env, env_test=env, num_unrolls=n_unrolls, **exp["args"])
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-{n_unrolls}unrolls",
                                         group=group_name)
                else:
                    logger = False
                trainer = Trainer(max_epochs=3000, log_every_n_steps=log_every_n_steps, accelerator="cpu", logger=logger)

            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env, **exp["args"])
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-{n_unrolls}impr",
                                         group=group_name)
                else:
                    logger = False
                trainer = Trainer(max_epochs=n_unrolls, log_every_n_steps=log_every_n_steps, accelerator='cpu', logger=logger)
            else:
                raise ValueError(f"Unknown model type: {exp['model']}")

            trainer.fit(model)
            wandb.finish()

            err1[j,i], err2[j,i] = test_pol_err(model.Pi, q_opt)
            bell_err[j,i] = model.bellman_error.cpu().numpy()

            if verbose:
                print(f"- {g}. Unrolls {n_unrolls}: Model: {exp['name']} Err1: {err1[j,i]:.3f} | bell_err: {bell_err[j,i]:.3f}")
    return err1, err2, bell_err


# ------------------------------
# Execution
# ------------------------------
q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

n_runs = 15
errs1_new = np.zeros((n_runs, len(Exps_new), N_unrolls.size))
errs2_new = np.zeros((n_runs, len(Exps_new), N_unrolls.size))
bell_errs_new = np.zeros((n_runs, len(Exps_new), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1_new[g], errs2_new[g], bell_errs_new[g] = run(
        g, N_unrolls, Exps_new, q_opt, group_name, use_logger, log_every_n_steps, verbose
    )
t_end = perf_counter()

print(f'----- New experiments solved in {(t_end-t_init)/60:.3f} minutes -----')

# ------------------------------
# Save results
# ------------------------------
if SAVE:
    file_name = os.path.join(PATH, f"{group_name}_data.npz")
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps_new,
             errs1=errs1_new, errs2=errs2_new, bell_errs=bell_errs_new)
    print("Data saved as:", file_name)
