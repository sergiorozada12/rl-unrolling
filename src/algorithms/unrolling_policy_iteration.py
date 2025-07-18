import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
import numpy as np
from numpy.linalg import eig, matrix_rank

from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss
from src.plots import plot_policy_and_value, plot_Pi, plot_filter_coefs
from src.models import UnrolledPolicyIterationModel, PolicyEvaluationLayer

from src.environments import CliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain


# TODO: move to utils folder?
def rew_smoothness(P_pi, r):
        diff = r.unsqueeze(1) - r.unsqueeze(0)
        smoothness = (P_pi * diff.square()).sum() / r.square().sum()
        return smoothness

def safe_wandb_log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)


class UnrollingDataset(Dataset):
    def __init__(self, nS, nA, N=1):
        self.policies = torch.ones(N, nS, nA)
        self.policies = self.policies / self.policies.sum(dim=-1, keepdim=True)
        self.qs = torch.zeros(N, nS * nA)

    def __len__(self):
        return len(self.policies)

    def __getitem__(self, idx):
        return self.qs[idx], self.policies[idx]


class UnrollingPolicyIterationTrain(pl.LightningModule):
    def __init__(self, env, env_test, K=3, num_unrolls=5, gamma=0.99, lr=1e-3, tau=1.0, beta=1.0, freq_plots=10, N=1, weight_sharing=False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.nS, self.nA = env.nS, env.nA
        self.register_buffer("P", env.P.clone())
        self.register_buffer("r", env.r.clone())
        self.register_buffer("P_test", env_test.P.clone())
        self.register_buffer("r_test", env_test.r.clone())
        self.gamma = gamma
        self.lr = lr
        self.N = N

        self.freq_plots = freq_plots
        self.Pi_train = []

        self.model = UnrolledPolicyIterationModel(self.P, self.r, self.nS, self.nA, K, num_unrolls, tau, beta, weight_sharing)
        self.model_test = UnrolledPolicyIterationModel(self.P_test, self.r_test, self.nS, self.nA, K, num_unrolls, tau, beta, weight_sharing)

    def training_step(self, batch, batch_idx):
        q_in, Pi_in = batch
        q_pred, Pi_pred = self.model(q_in, Pi_in)

        P_pi = self.model.layers[-2].compute_transition_matrix(Pi_pred).detach()
        target = self.r + self.gamma * (P_pi @ q_pred.detach()) # Detach future

        q_reshaped = q_pred.view(self.nS, self.nA)
        target_reshaped = target.view(self.nS, self.nA)

        loss = mse_loss(q_reshaped, target_reshaped)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        smoothness = rew_smoothness(P_pi, self.r)
        self.log("reward_smoothness", smoothness, on_step=True, on_epoch=False, prog_bar=True)
        

        # For debug
        q_pred = q_pred.detach()
        bellman_error = torch.norm(q_pred - target)
        self.log("bellman_error", bellman_error, on_step=True, on_epoch=False, prog_bar=True)

        if batch_idx == 0 and self.current_epoch % self.freq_plots == 0:
            self.Pi_train.append(Pi_pred.detach().cpu().numpy())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(UnrollingDataset(self.nS, self.nA, N=self.N), batch_size=1, shuffle=True)

    def on_fit_end(self):
        dataset = UnrollingDataset(self.nS, self.nA, N=self.N)
        q_sample, Pi_sample = dataset[0]
        q_sample = q_sample.to(self.device)
        Pi_sample = Pi_sample.to(self.device)
        q, Pi_out = self.model(q_sample, Pi_sample)
    
        # Compute Bellman's err
        P_pi = self.model.layers[-2].compute_transition_matrix(Pi_out).detach()
        target = self.r + self.gamma * (P_pi @ q.detach())
        bellman_error = torch.norm(q - target) / torch.norm(target)

        # Save predicted policy and Bellman's err
        self.q = q.detach()
        self.Pi = Pi_out.detach()
        self.bellman_error = bellman_error.detach()

        fig_policy = plot_policy_and_value(q.view(self.nS, self.nA), Pi_out)
        fig_policy_full = plot_policy_and_value(q.view(self.nS, self.nA), Pi_out, plot_all_trans=True)
        fig_P = plot_Pi(Pi_out.detach().cpu().numpy())
        safe_wandb_log({
            "policy_plot": wandb.Image(fig_policy),
            "full_policy_plot": wandb.Image(fig_policy_full),
            "Pi_plot": wandb.Image(fig_P)})

        plt.close(fig_policy_full)
        plt.close(fig_policy)
        plt.close(fig_P)

        if self.model.h is not None:
            fig_h = plot_filter_coefs(self.model.h.detach().cpu().numpy())
            safe_wandb_log({"shared_h_coefficients": wandb.Image(fig_h)})
            plt.close(fig_h)

        # Check if P_pi is diagonalizable
        P_pi_np = P_pi.numpy()
        eigenvals, eigenvectors = eig(P_pi_np)
        try:
            P_hat = eigenvectors @ np.diag(eigenvals) @ np.linalg.inv(eigenvectors)
            diff = np.linalg.norm(P_pi_np - P_hat)                # Frobenius norm by default
            print("P_pi is diagonalizable: ", diff < 1e-6)
        except np.linalg.LinAlgError:
            print("P_pi is NOT diagonalizable")

        # TEST PERMUTABILIDAD
        with torch.no_grad():
            for layer1, layer2 in zip(self.model.layers, self.model_test.layers):
                if isinstance(layer1, PolicyEvaluationLayer) and isinstance(layer2, PolicyEvaluationLayer):
                    layer2.h.copy_(layer1.h)

        dataset = UnrollingDataset(self.nS, self.nA, N=self.N)
        q_sample, Pi_sample = dataset[0]
        q_sample = q_sample.to(self.device)
        Pi_sample = Pi_sample.to(self.device)
        q, Pi_out = self.model_test(q_sample, Pi_sample)

        fig_policy = plot_policy_and_value(q.view(self.nS, self.nA), Pi_out, goal_row=0)
        fig_policy_full = plot_policy_and_value(q.view(self.nS, self.nA), Pi_out, goal_row=0, plot_all_trans=True)

        P_pi = self.model.layers[-2].compute_transition_matrix(Pi_out).detach().numpy()
        fig_P = plot_Pi(Pi_out.detach().numpy())

        safe_wandb_log({
            "policy_transf_plot": wandb.Image(fig_policy),
            "full_policy_transf_plot": wandb.Image(fig_policy_full),
            "Pi_transf_plot": wandb.Image(fig_P)})
        plt.close(fig_policy_full)
        plt.close(fig_policy)
        plt.close(fig_P)

    def test_pol_err(self, q_opt, max_eval_iters=200):
        q_opt = q_opt.to(self.device)

        # Get a deterministic policy
        nS, _ = self.Pi.shape
        greedy_actions = self.Pi.argmax(axis=1)
        Pi_det = np.zeros_like(self.Pi)
        Pi_det[np.arange(nS), greedy_actions] = 1.0

        # Run policy evaluation with learned policy
        env = CliffWalkingEnv()
        model = PolicyIterationTrain(env, max_eval_iters=max_eval_iters, Pi_init=torch.Tensor(Pi_det))
        model.on_fit_start()
        P_pi = model.compute_transition_matrix(model.P, model.Pi)
        q_est = model.policy_evaluation(P_pi, model.r).detach()

        # Compute errors
        err1 = (torch.norm(q_est - q_opt) / torch.norm(q_opt)) ** 2
        err2 = (torch.norm(q_est/torch.norm(q_est) - q_opt/torch.norm(q_opt))) ** 2
        return err1.cpu().numpy(), err2.cpu().numpy()