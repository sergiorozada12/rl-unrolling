# %%
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from time import perf_counter
import wandb
import numpy as np

import torch
torch.set_float32_matmul_precision('medium')

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.utils import get_optimal_q, test_pol_err, plot_errors

SAVE = True
PATH = "results/n_unrolls/"

# %% [markdown]
# ## Auxiliary functions

# %%
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
                trainer = Trainer(max_epochs=2000, log_every_n_steps=log_every_n_steps, accelerator="cpu", logger=logger)

            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env, **exp["args"])
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-{n_unrolls}impr",
                                         group=group_name)
                else:
                    logger = False
                trainer = Trainer(max_epochs=n_unrolls, log_every_n_steps=log_every_n_steps, accelerator='cpu', logger=logger)
            else:
                raise Exception("Unknown model")

            trainer.fit(model)
            wandb.finish()

            err1[j,i], err2[j,i] = test_pol_err(model, q_opt)
            bell_err[j,i] = model.bellman_error.cpu().numpy()

            if verbose:
                print(f"- {g}. Unrolls {n_unrolls}: Model: {exp["name"]} Err1: {err1[j,i]:.3f} | bell_err: {bell_err[j,i]:.3f}")
    return err1, err2, bell_err


# %% [markdown]
# ## K = 5

# %%
verbose = True
use_logger = False
log_every_n_steps = 1
K = 5
group_name = f"n_unrolls-K{K}"

N_unrolls = np.arange(2,11, 2)
Exps = [
    {"model": "pol-it", "args": {"max_eval_iters": 1}, "fmt": "^-", "name": "val-it"},
    {"model": "pol-it", "args": {"max_eval_iters": K}, "fmt": "x-", "name": f"pol-it-{K}eval"},

    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-K{K}-WS"},
    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o--", "name": f"unr-K{K}"},
]

q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 15

errs1 = np.zeros((n_runs, len(Exps), N_unrolls.size))
errs2 = np.zeros((n_runs, len(Exps), N_unrolls.size))
bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1[g], errs2[g], bell_errs[g] = run(g, N_unrolls, Exps, q_opt, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps, errs1=errs1, errs2=errs2, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "n_unrolls-K5_data.npz", allow_pickle=True)
# N_unrolls = data["N_unrolls"]
# Exps = data["Exps"]
# errs1 = data["errs1"]
# errs2 = data["errs2"]
# bell_errs = data["bell_errs"]

# %%
skip_idx = []
xlabel = "Number of unrolls"
plot_errors(errs1, N_unrolls, Exps, xlabel, "Q err", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs2, N_unrolls, Exps, xlabel, "Q err 2", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, N_unrolls, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')


# %% [markdown]
# ## K=10

# %%
verbose = True
use_logger = True
log_every_n_steps = 1
K = 10
group_name = f"n_unrolls-K{K}"
N_unrolls = np.arange(2,11, 2)
Exps = [
    {"model": "pol-it", "args": {"max_eval_iters": 1}, "fmt": "^-", "name": "val-it"},
    {"model": "pol-it", "args": {"max_eval_iters": K}, "fmt": "x-", "name": f"pol-it-{K}eval"},
    # {"model": "pol-it", "args": {"max_eval_iters": 20}, "name": "pol-it-20eval"},

    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-K{K}-WS"},
    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o--", "name": f"unr-K{K}"},
]

q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 15

errs1 = np.zeros((n_runs, len(Exps), N_unrolls.size))
errs2 = np.zeros((n_runs, len(Exps), N_unrolls.size))
bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1[g], errs2[g], bell_errs[g] = run(g, N_unrolls, Exps, q_opt, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps, errs1=errs1, errs2=errs2, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "n_unrolls-K10_data.npz", allow_pickle=True)
# N_unrolls = data["N_unrolls"]
# Exps = data["Exps"]
# errs1 = data["errs1"]
# errs2 = data["errs2"]
# bell_errs = data["bell_errs"]

# %%
skip_idx = []
xlabel = "Number of unrolls"
plot_errors(errs1, N_unrolls, Exps, xlabel, "Q err", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs2, N_unrolls, Exps, xlabel, "Q err 2", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, N_unrolls, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')


# %% [markdown]
# ## K=15

# %%
verbose = True
use_logger = True
log_every_n_steps = 1
K = 15
group_name = f"n_unrolls-K{K}"
N_unrolls = np.arange(2,11, 2)
Exps = [
    {"model": "pol-it", "args": {"max_eval_iters": 1}, "fmt": "^-", "name": "val-it"},
    {"model": "pol-it", "args": {"max_eval_iters": K}, "fmt": "x-", "name": f"pol-it-{K}eval"},
    # {"model": "pol-it", "args": {"max_eval_iters": 20}, "name": "pol-it-20eval"},

    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-K{K}-WS"},
    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o--", "name": f"unr-K{K}"},
]

q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 15

errs1 = np.zeros((n_runs, len(Exps), N_unrolls.size))
errs2 = np.zeros((n_runs, len(Exps), N_unrolls.size))
bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1[g], errs2[g], bell_errs[g] = run(g, N_unrolls, Exps, q_opt, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps, errs1=errs1, errs2=errs2, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "n_unrolls-K10_data.npz", allow_pickle=True)
# N_unrolls = data["N_unrolls"]
# Exps = data["Exps"]
# errs1 = data["errs1"]
# errs2 = data["errs2"]
# bell_errs = data["bell_errs"]

# %%
skip_idx = []
xlabel = "Number of unrolls"
plot_errors(errs1, N_unrolls, Exps, xlabel, "Q err", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs2, N_unrolls, Exps, xlabel, "Q err 2", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, N_unrolls, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')



