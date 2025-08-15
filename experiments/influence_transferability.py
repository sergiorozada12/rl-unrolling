# %%
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from time import perf_counter
import wandb
import numpy as np

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.utils import get_optimal_q, test_pol_err, plot_errors, save_error_matrix_to_csv

SAVE = True
PATH = "results/transfer/"

# %% [markdown]
# ## Auxiliary functions

# %%
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

            # trainer.fit(model)
            # wandb.finish()

            # # err1[j,i], err2[j,i] = test_pol_err(model, q_opt)
            # err1[j,i], err2[j,i] = test_pol_err(model, q_opt, env_builder=MirroredCliffWalkingEnv)
            # bell_err[j,i] = model.bellman_error.cpu().numpy()

            if verbose:
                print(f"- {g}. Unrolls {n_unrolls}: Model: {exp['name']} Err: {err[j,i]:.3f} | Err tranf: {err_tranf[j,i]:.3f} | bell_err: {bell_err_tranf[j,i]:.3f}")
    return err, err_tranf, bell_err_tranf


# %% [markdown]
# ## K = 5

# %%
verbose = True
use_logger = True
log_every_n_steps = 1
K = 5
group_name = f"transfer-K{K}"

N_unrolls = np.arange(2,11, 2)
Exps = [
    {"model": "pol-it", "args": {"max_eval_iters": 1}, "fmt": "^-", "name": "val-it"},
    {"model": "pol-it", "args": {"max_eval_iters": K}, "fmt": "x-", "name": f"pol-it-{K}eval"},

    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-K{K}-WS"},
    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o--", "name": f"unr-K{K}"},
]

q_opt = get_optimal_q(mirror_env=False, use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)
q_opt_mirr = get_optimal_q(mirror_env=True, use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 15

errs = np.zeros((n_runs, len(Exps), N_unrolls.size))
errs_trans = np.zeros((n_runs, len(Exps), N_unrolls.size))
bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs[g], errs_trans[g], bell_errs[g] = run(g, N_unrolls, Exps, q_opt, q_opt_mirr, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps, errs=errs, errs_trans=errs_trans, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "transfer-K5_data.npz", allow_pickle=True)
# N_unrolls = data["N_unrolls"]
# Exps = data["Exps"]
# errs = data["errs"]
# errs_trans = data["errs_trans"]
# bell_errs = data["bell_errs"]

# %%
skip_idx = []
xlabel = "Number of unrolls"
plot_errors(errs, N_unrolls, Exps, xlabel, "Q err (training)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs_trans, N_unrolls, Exps, xlabel, "Q err (Transfer)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, N_unrolls, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')


# %% [markdown]
# ## K=10

# %%
verbose = True
use_logger = True
log_every_n_steps = 1
K = 10
group_name = f"transfer-K{K}"
N_unrolls = np.arange(2,11, 2)
Exps = [
    {"model": "pol-it", "args": {"max_eval_iters": 1}, "fmt": "^-", "name": "val-it"},
    {"model": "pol-it", "args": {"max_eval_iters": K}, "fmt": "x-", "name": f"pol-it-{K}eval"},
    # {"model": "pol-it", "args": {"max_eval_iters": 20}, "name": "pol-it-20eval"},

    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-K{K}-WS"},
    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o--", "name": f"unr-K{K}"},
]

q_opt = get_optimal_q(mirror_env=False, use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)
q_opt_mirr = get_optimal_q(mirror_env=True, use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 15

errs = np.zeros((n_runs, len(Exps), N_unrolls.size))
errs_trans = np.zeros((n_runs, len(Exps), N_unrolls.size))
bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs[g], errs_trans[g], bell_errs[g] = run(g, N_unrolls, Exps, q_opt, q_opt_mirr, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps, errs=errs, errs_trans=errs_trans, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "transfer-K10_data.npz", allow_pickle=True)
# N_unrolls = data["N_unrolls"]
# Exps = data["Exps"]
# errs = data["errs"]
# errs_trans = data["errs_trans"]
# bell_errs = data["bell_errs"]

# %%
skip_idx = []
xlabel = "Number of unrolls"
plot_errors(errs, N_unrolls, Exps, xlabel, "Q err (training)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs_trans, N_unrolls, Exps, xlabel, "Q err (Transfer)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, N_unrolls, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')


# %% [markdown]
# ## K=15

# %%
verbose = True
use_logger = False
log_every_n_steps = 1
K = 15
group_name = f"transfer-K{K}"
N_unrolls = np.arange(2,11, 2)
Exps = [
    {"model": "pol-it", "args": {"max_eval_iters": 1}, "fmt": "^-", "name": "val-it"},
    {"model": "pol-it", "args": {"max_eval_iters": K}, "fmt": "x-", "name": f"pol-it-{K}eval"},
    # {"model": "pol-it", "args": {"max_eval_iters": 20}, "name": "pol-it-20eval"},

    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "fmt": "o-", "name": f"unr-K{K}-WS"},
    {"model": "unroll", "args": {"K": K, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "fmt": "o--", "name": f"unr-K{K}"},
]

q_opt = get_optimal_q(mirror_env=False, use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)
q_opt_mirr = get_optimal_q(mirror_env=True, use_logger=use_logger, log_every_n_steps=log_every_n_steps, group_name=group_name)

# %%
n_runs = 15

errs = np.zeros((n_runs, len(Exps), N_unrolls.size))
errs_trans = np.zeros((n_runs, len(Exps), N_unrolls.size))
bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs[g], errs_trans[g], bell_errs[g] = run(g, N_unrolls, Exps, q_opt, q_opt_mirr, group_name, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + f"{group_name}_data.npz"
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps, errs=errs, errs_trans=errs_trans, bell_errs=bell_errs)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "transfer-K15_data.npz", allow_pickle=True)
# N_unrolls = data["N_unrolls"]
# Exps = data["Exps"]
# errs = data["errs"]
# errs_trans = data["errs_trans"]
# bell_errs = data["bell_errs"]

# %%
skip_idx = []
xlabel = "Number of unrolls"
plot_errors(errs, N_unrolls, Exps, xlabel, "Q err (training)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs_trans, N_unrolls, Exps, xlabel, "Q err (Transfer)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, N_unrolls, Exps, xlabel, "Bellman err", skip_idx=skip_idx, agg="median", deviation='prctile')


# %% [markdown]
# ## Plot All

# %%
# Load data
files = ["transfer-K5_data.npz", "transfer-K10_data.npz", "transfer-K15_data.npz"]

Exps = []
errs_list = []
errs_trans_list = []
bell_errs_list = []
for file in files:
    data = np.load(PATH + file, allow_pickle=True)
    
    if 'N_unrolls' not in locals():  # Save only once
        N_unrolls = data["N_unrolls"]
    
    Exps += list(data["Exps"])
    
    errs_list.append(data["errs"])
    errs_trans_list.append(data["errs_trans"])
    bell_errs_list.append(data["bell_errs"])

# Concatenate all data
errs = np.concatenate(errs_list, axis=1)
errs_trans = np.concatenate(errs_trans_list, axis=1)
bell_errs = np.concatenate(bell_errs_list, axis=1)

# %%
# Indexes 0, 4 and 8 are all policy evaluation
skip_idx = [4, 8]
xlabel = "Number of unrolls"
plot_errors(errs_trans, N_unrolls, Exps, xlabel, "Q Err (trans)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(errs, N_unrolls, Exps, xlabel, "Q Err (trans)", skip_idx=skip_idx, agg="median", deviation='prctile')
plot_errors(bell_errs, N_unrolls, Exps, xlabel, "Q Err (trans)", skip_idx=skip_idx, agg="median", deviation='prctile')

if SAVE:
    file_name = PATH + "n_unrolls_all_data_med_err.csv"
    save_error_matrix_to_csv(np.median(errs_trans, axis=0), N_unrolls, Exps, file_name)
    file_name = PATH + "n_unrolls_all_data_prctile25.csv"
    save_error_matrix_to_csv(np.percentile(errs_trans, 25, axis=0), N_unrolls, Exps, file_name)
    file_name = PATH + "n_unrolls_all_data_prctile75.csv"
    save_error_matrix_to_csv(np.percentile(errs_trans, 75, axis=0), N_unrolls, Exps, file_name)



