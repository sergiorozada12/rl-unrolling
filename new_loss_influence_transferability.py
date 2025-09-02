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
group_name = "new_loss_transferability"
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

    for i, n_unrolls in enumerate(N_unrolls):
        n_unrolls = int(n_unrolls)
        for j, exp in enumerate(Exps):
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
                wandb.finish()

                _, err[j,i] = test_pol_err(model.Pi, q_opt_mirr, mirror_env=True, device=model.device)
                err_tranf[j,i] = err[j,i]
                bell_err_tranf[j,i] = model.bellman_error.cpu().numpy()
                
            else:
                raise Exception("Unknown model")

            if verbose:
                print(f"- {g}. Unrolls {n_unrolls}: Model: {exp['name']} Err: {err[j,i]:.3f} | Err tranf: {err_tranf[j,i]:.3f} | bell_err: {bell_err_tranf[j,i]:.3f}")
    return err, err_tranf, bell_err_tranf

# ------------------------------
# Definition of combinations
# ------------------------------
loss_types = ["original_with_detach", "original_no_detach", "max_with_detach", "max_no_detach"]
init_types = ["zeros", "random"]

# Create experiments list
Exps_new = []

# Val-it configuration (equivalent to "val-it" from original transferability script)
Exps_new.append({
    "model": "pol-it",
    "args": {"max_eval_iters": 1},
    "fmt": "^-",
    "name": "val-it"
})

# Pol-it 5 configuration (equivalent to "pol-it-5eval" from original transferability script)
Exps_new.append({
    "model": "pol-it", 
    "args": {"max_eval_iters": 5},
    "fmt": "x-",
    "name": "pol-it-5eval"
})

# BN-WS 5 configuration (equivalent to "unr-K5-WS" from original transferability script)
Exps_new.append({
    "model": "unroll",
    "args": {
        "K": 5,
        "tau": 5,
        "lr": 5e-3,
        "weight_sharing": True,
        "loss_type": "original_with_detach",
        "init_q": "zeros"
    },
    "fmt": "o-",
    "name": "unr-K5-WS"
})

# Add all combinations of loss functions and initializations
for loss_type in loss_types:
    for init_type in init_types:
        exp_name = f"unr-K5-{loss_type}-{init_type}"
        Exps_new.append({
            "model": "unroll",
            "args": {
                "K": 5,
                "tau": 5,
                "lr": 5e-3,
                "weight_sharing": True,
                "loss_type": loss_type,
                "init_q": init_type
            },
            "fmt": "d-",
            "name": exp_name
        })

print(f"Total experiments: {len(Exps_new)}")
for i, exp in enumerate(Exps_new):
    print(f"{i}: {exp['name']}")

# ------------------------------
# Execution
# ------------------------------
# Get optimal Q for both environments
print("Computing optimal Q for original environment...")
q_opt = get_optimal_q(mirror_env=False, use_logger=use_logger, log_every_n_steps=log_every_n_steps, 
                      group_name=group_name)

print("Computing optimal Q for mirrored environment...")
q_opt_mirr = get_optimal_q(mirror_env=True, use_logger=use_logger, log_every_n_steps=log_every_n_steps, 
                           group_name=group_name)

# Run experiments
n_runs = 15
errs_new = np.zeros((n_runs, len(Exps_new), N_unrolls.size))
errs_trans_new = np.zeros((n_runs, len(Exps_new), N_unrolls.size))
bell_errs_new = np.zeros((n_runs, len(Exps_new), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs_new[g], errs_trans_new[g], bell_errs_new[g] = run(
        g, N_unrolls, Exps_new, q_opt, q_opt_mirr, group_name, use_logger, log_every_n_steps, verbose
    )
t_end = perf_counter()

print(f'----- New transferability experiments solved in {(t_end-t_init)/60:.3f} minutes -----')

# ------------------------------
# Save results
# ------------------------------
if SAVE:
    file_name = os.path.join(PATH, f"{group_name}_data.npz")
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps_new,
             errs=errs_new, errs_trans=errs_trans_new, bell_errs=bell_errs_new)
    print("Data saved as:", file_name)