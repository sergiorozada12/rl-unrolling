# %%
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from time import perf_counter
import wandb
import numpy as np

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.utils import get_optimal_q, test_pol_err, plot_errors

SAVE = True
PATH = "results/filter_order/"

# %% [markdown]
# ## Auxiliary functions

# %%
def run(g, Ks, Exps, q_opt, group_name, use_logger=True, log_every_n_steps=1, verbose=False):
    err1 = np.zeros((len(Exps), Ks.size))
    err2 = np.zeros((len(Exps), Ks.size))
    bell_err = np.zeros((len(Exps), Ks.size))
    
    use_logger = use_logger and g == 0

    for i, K in enumerate(Ks):
        K = int(K)
        for j, exp in enumerate(Exps):
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
                trainer = Trainer(max_epochs=exp['args']['max_epochs'], log_every_n_steps=log_every_n_steps, accelerator='cpu',
                                  logger=logger)
            else:
                raise Exception("Unknown model")

            trainer.fit(model)
            wandb.finish()

            err1[j,i], err2[j,i] = test_pol_err(model, q_opt)
            bell_err[j,i] = model.bellman_error.cpu().numpy()

            if verbose:
                print(f"- {g}. K {K}: Model: {exp['name']} Err1: {err1[j,i]:.3f} | bell_err: {bell_err[j,i]:.3f}")
    return err1, err2, bell_err


# %% [markdown]
# ## Unrolls = 5

# %%
verbose = True
use_logger = True
log_every_n_steps = 1
num_unrolls = 5
group_name = f"filter_order-{num_unrolls}"

Ks = np.array([1, 2, 3, 5, 10, 15]) # np.arange(1,30, 2)
Exps = [
    {"model": "pol-it", "args": {"max_epochs": num_unrolls}, "fmt": "x-", "name": f"pol-it-{num_unrolls}eval"},

    {"model": "unroll", "args": {"num_unrolls": num_unrolls, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-{num_unrolls}unrolls-WS"},
    {"model": "unroll", "args": {"num_unrolls": num_unrolls, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o-", "name": f"unr-{num_unrolls}unrolls"},

    # {"model": "unroll", "args": {"num_unrolls": 5, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-5unrolls"},
    # {"model": "unroll", "args": {"num_unrolls": 10, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-10unrolls"},
]

q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 5

errs1 = np.zeros((n_runs, len(Exps), Ks.size))
errs2 = np.zeros((n_runs, len(Exps), Ks.size))
bell_errs = np.zeros((n_runs, len(Exps), Ks.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1[g], errs2[g], bell_errs[g] = run(g, Ks, Exps, q_opt, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=Ks, Exps=Exps, errs1=errs1, errs2=errs2, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "data_v2.npz", allow_pickle=True)
# Ks = data["Ks"]
# Exps = data["Exps"]
# errs1 = data["errs1"] 

# errs2 = data["errs2"]

# %%
skip_idx = []
xlabel = "K"
plot_errors(errs1, Ks, Exps, xlabel, "Q err", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs2, Ks, Exps, xlabel, "Q err 2", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, Ks, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')


# %% [markdown]
# ## Unrolls = 10

# %%
verbose = True
use_logger = True
log_every_n_steps = 1
num_unrolls = 10
group_name = f"filter_order-{num_unrolls}"

Ks = np.array([1, 2, 3, 5, 10, 15]) # np.arange(1,30, 2)
Exps = [
    {"model": "pol-it", "args": {"max_epochs": num_unrolls}, "fmt": "x-", "name": f"pol-it-{num_unrolls}eval"},

    {"model": "unroll", "args": {"num_unrolls": num_unrolls, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-{num_unrolls}unrolls-WS"},
    {"model": "unroll", "args": {"num_unrolls": num_unrolls, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o-", "name": f"unr-{num_unrolls}unrolls"},

    # {"model": "unroll", "args": {"num_unrolls": 5, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-5unrolls"},
    # {"model": "unroll", "args": {"num_unrolls": 10, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-10unrolls"},
]

q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 5

errs1 = np.zeros((n_runs, len(Exps), Ks.size))
errs2 = np.zeros((n_runs, len(Exps), Ks.size))
bell_errs = np.zeros((n_runs, len(Exps), Ks.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1[g], errs2[g], bell_errs[g] = run(g, Ks, Exps, q_opt, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=Ks, Exps=Exps, errs1=errs1, errs2=errs2, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "data_v2.npz", allow_pickle=True)
# Ks = data["Ks"]
# Exps = data["Exps"]
# errs1 = data["errs1"] 

# errs2 = data["errs2"]

# %%
skip_idx = []
xlabel = "K"
plot_errors(errs1, Ks, Exps, xlabel, "Q err", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs2, Ks, Exps, xlabel, "Q err 2", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, Ks, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')


# %% [markdown]
# ## Unrolls = 15

# %%
verbose = True
use_logger = True
log_every_n_steps = 1
num_unrolls = 15
group_name = f"filter_order-{num_unrolls}"

Ks = np.array([1, 2, 3, 5, 10, 15]) # np.arange(1,30, 2)
Exps = [
    {"model": "pol-it", "args": {"max_epochs": num_unrolls}, "fmt": "x-", "name": f"pol-it-{num_unrolls}eval"},

    {"model": "unroll", "args": {"num_unrolls": num_unrolls, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-{num_unrolls}unrolls-WS"},
    {"model": "unroll", "args": {"num_unrolls": num_unrolls, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o-", "name": f"unr-{num_unrolls}unrolls"},

    # {"model": "unroll", "args": {"num_unrolls": 5, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-5unrolls"},
    # {"model": "unroll", "args": {"num_unrolls": 10, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-10unrolls"},
]

q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 5

errs1 = np.zeros((n_runs, len(Exps), Ks.size))
errs2 = np.zeros((n_runs, len(Exps), Ks.size))
bell_errs = np.zeros((n_runs, len(Exps), Ks.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1[g], errs2[g], bell_errs[g] = run(g, Ks, Exps, q_opt, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=Ks, Exps=Exps, errs1=errs1, errs2=errs2, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "data_v2.npz", allow_pickle=True)
# Ks = data["Ks"]
# Exps = data["Exps"]
# errs1 = data["errs1"] 

# errs2 = data["errs2"]

# %%
skip_idx = []
xlabel = "K"
plot_errors(errs1, Ks, Exps, xlabel, "Q err", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs2, Ks, Exps, xlabel, "Q err 2", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, Ks, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')



