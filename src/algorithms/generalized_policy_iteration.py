import torch
import pytorch_lightning as pl

import wandb
import matplotlib.pyplot as plt

from src.plots import plot_policy_and_value


class PolicyIterationTrain(pl.LightningModule):
    def __init__(self, env, gamma=0.99, max_eval_iters=1000):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.nS, self.nA = env.nS, env.nA
        self.register_buffer("P",  env.P.clone())
        self.register_buffer("r",  env.r.clone())

        self.Pi = None
        self.q  = None
        self.gamma = gamma
        self.max_eval_iters = max_eval_iters

    def lift_policy_matrix(self, Pi):
        Pi_ext = torch.zeros(self.nS, self.nS * self.nA, device=self.device)
        for s in range(self.nS):
            for a in range(self.nA):
                Pi_ext[s, s * self.nA + a] = Pi[s, a]
        return Pi_ext

    def compute_transition_matrix(self, P, Pi):
        Pi_ext = self.lift_policy_matrix(Pi)
        return P @ Pi_ext

    def policy_evaluation(self, P_pi, r):
        q = self.q[::]
        for _ in range(self.max_eval_iters):
            q = r + self.gamma * (P_pi @ q)
        return q

    def policy_improvement(self, q):
        q_reshaped = q.view(self.nS, self.nA)
        greedy_actions = torch.argmax(q_reshaped, dim=1)
        Pi_new = torch.zeros((self.nS, self.nA), device=self.device)
        Pi_new[torch.arange(self.nS), greedy_actions] = 1.0
        return Pi_new

    def on_fit_start(self):
        self.Pi = torch.full((self.nS, self.nA), 1 / self.nA, device=self.device)
        self.q  = torch.zeros(self.nS * self.nA, device=self.device)

    def on_fit_end(self):
        q = self.q.view(self.nS, self.nA)

        # Standard plot
        #fig = plot_policy_and_value(
        #    q,
        #    self.Pi,
        #    shape=(4, 12),
        #    goal_pos=(3, 11),
        #    cliff_row=3,
        #    cliff_cols=range(1, 11)
        #)

        # Mirrored plot
        #fig = plot_policy_and_value(
        #    q,
        #    self.Pi,
        #    shape=(4, 12),
        #    goal_pos=(0, 11),
        #    cliff_row=0,
        #    cliff_cols=range(1, 11)
        #)

        # HighRes plot
        fig = plot_policy_and_value(
            q,
            self.Pi,
            shape=(8, 24),
            goal_pos=(7, 23),
            cliff_row=7,
            cliff_cols=range(1, 23)
        )

        #fig = plot_policy_and_value(q, self.Pi, goal_row=self.goal_row)
        wandb.log({"policy_plot": wandb.Image(fig)})
        plt.close(fig)

    def training_step(self, batch, batch_idx):
        P_pi = self.compute_transition_matrix(self.P, self.Pi)
        q = self.policy_evaluation(P_pi, self.r)

        bellman_error = torch.norm(q - (self.r + self.gamma * P_pi @ q))
        policy_diff = torch.norm(self.Pi - self.policy_improvement(q))
        q_norm = torch.norm(q)

        self.log("bellman_error", bellman_error, on_step=True, on_epoch=False, prog_bar=True)
        self.log("policy_diff", policy_diff, on_step=True, on_epoch=False, prog_bar=True)
        self.log("q_norm", q_norm, on_step=True, on_epoch=False, prog_bar=True)

        self.Pi = self.policy_improvement(q)
        self.q = q

    def configure_optimizers(self):
        return None

    def train_dataloader(self):
        return torch.utils.data.DataLoader([0])
