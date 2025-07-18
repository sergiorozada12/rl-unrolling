# %%
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from time import perf_counter
import wandb
import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.unrolling_policy_iteration import UnrollingPolicyIterationTrain
from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain

GROUP_NAME = "N_Unrolls"
SAVE = True
PATH = "results/n_unrolls/"

# %% [markdown]
# ## Auxiliary functions

# %%
def plot_errors(errs, N_unrolls, Exps, skip_idx=[]):

    plt.figure(figsize=(8, 5))
    
    for i, exp in enumerate(Exps):
        if i in skip_idx:
            continue
        label = exp.get("name", f"Exp {i}")
        plt.plot(N_unrolls, errs[i], marker='o', label=label)
    
    plt.xlabel("Number of unrolls")
    plt.ylabel("Q Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_optimal_q(max_eval_iters=50, max_epochs=50, use_logger=True, log_every_n_steps=1):
    env = CliffWalkingEnv()
    model = PolicyIterationTrain(env, gamma=0.99, max_eval_iters=max_eval_iters)

    if use_logger:
            logger = WandbLogger(
            project="rl-unrolling",
            name=f"Optimal_pol-{max_eval_iters}eval-{max_epochs}impr",
            group=GROUP_NAME
        )
    else:
        logger = False

    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        accelerator='cpu',
        logger=logger,
    )
    
    trainer.fit(model, train_dataloaders=None)
    wandb.finish()
    return model.q.detach()

def run(g, N_unrolls, Exps, q_opt, use_logger=True, log_every_n_steps=1, verbose=False):
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
                                         group=GROUP_NAME)
                else:
                    logger = False
                trainer = Trainer(max_epochs=5000, log_every_n_steps=log_every_n_steps, accelerator="auto",
                                  strategy=DDPStrategy(find_unused_parameters=True), logger=logger)

            elif exp["model"] == "pol-it":
                model = PolicyIterationTrain(env=env, **exp["args"])
                if use_logger:
                    logger = WandbLogger(project="rl-unrolling", name=f"{exp['name']}-{n_unrolls}impr",
                                         group=GROUP_NAME)
                else:
                    logger = False
                trainer = Trainer(max_epochs=int(n_unrolls), log_every_n_steps=log_every_n_steps, accelerator='cpu', logger=logger)
            else:
                raise Exception("Unknown model")

            trainer.fit(model)
            wandb.finish()

            err1[j,i], err2[j,i] = model.test_pol_err(q_opt)
            bell_err[j,i] = model.bellman_error.cpu().numpy()

            if verbose:
                print(f"- {g}. Unrolls {n_unrolls}: Model: {exp["name"]} Err1: {err1[j,i]:.3f} | bell_err: {bell_err[j,i]:.3f}")
    return err1, err2, bell_err


# %% [markdown]
# ## Running different models

# %%
verbose = True
use_logger = True
log_every_n_steps = 1

N_unrolls = np.array([1, 2, 3, 5])  #np.arange(1,16)
Exps = [
    {"model": "pol-it", "args": {"max_eval_iters": 1}, "name": "val-it"},
    {"model": "pol-it", "args": {"max_eval_iters": 10}, "name": "pol-it-10eval"},
    {"model": "pol-it", "args": {"max_eval_iters": 20}, "name": "pol-it-20eval"},

    # {"model": "unroll", "args": {"K": 5, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "name": "unr-K5-WS"},
    {"model": "unroll", "args": {"K": 10, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "name": "unr-K10-WS"},
    {"model": "unroll", "args": {"K": 20, "tau": 5, "lr": 5e-3, "weight_sharing": True}, "name": "unr-K20-WS"},

    # {"model": "unroll", "args": {"K": 5, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-5"},
    {"model": "unroll", "args": {"K": 10, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-K10"},
    {"model": "unroll", "args": {"K": 20, "tau": 5, "lr": 5e-3, "weight_sharing": False}, "name": "unr-K20"},
]

q_opt = get_optimal_q(use_logger=use_logger, log_every_n_steps=log_every_n_steps)

# %%
n_runs = 5

errs1 = np.zeros((n_runs, len(Exps), N_unrolls.size))
errs2 = np.zeros((n_runs, len(Exps), N_unrolls.size))
bell_errs = np.zeros((n_runs, len(Exps), N_unrolls.size))

t_init = perf_counter()
for g in range(n_runs):
    errs1[g], errs2[g], bell_errs[g] = run(g, N_unrolls, Exps, q_opt, use_logger, log_every_n_steps, verbose)

t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

if SAVE:
    file_name = PATH + "n_unrolls.npz"
    np.savez(file_name, N_unrolls=N_unrolls, Exps=Exps, errs1=errs1, errs2=errs2)
    print("Data saved as:", file_name)

# %%
# # Load data
# data = np.load(PATH + "data_v2.npz", allow_pickle=True)
# N_unrolls = data["N_unrolls"]
# Exps = data["Exps"]
# errs1 = data["errs1"]
# errs2 = data["errs2"]

# %%
skip_idx = []
plot_errors(np.mean(errs1, axis=0), N_unrolls, Exps, skip_idx=skip_idx)
plot_errors(np.mean(errs2, axis=0), N_unrolls, Exps, skip_idx=skip_idx)
plot_errors(np.mean(bell_errs, axis=0), N_unrolls, Exps, skip_idx=skip_idx)



