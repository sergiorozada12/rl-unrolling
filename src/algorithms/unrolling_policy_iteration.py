import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss
from src.plots import plot_policy_and_value
from src.models import UnrolledPolicyIterationModel


class UnrollingDataset(Dataset):
    def __init__(self, nS, nA, N=500):
        # self.policies = torch.rand(N, nS, nA)
        self.policies = torch.ones(N, nS, nA)
        self.policies = self.policies / self.policies.sum(dim=-1, keepdim=True)
        self.qs = torch.randn(N, nS * nA) * 0.01

    def __len__(self):
        return len(self.policies)

    def __getitem__(self, idx):
        return self.qs[idx], self.policies[idx]


class UnrollingPolicyIterationTrain(pl.LightningModule):
    def __init__(self, env, K=3, num_unrolls=5, gamma=0.99, lr=1e-3, tau=1.0, beta=1.0):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.nS, self.nA = env.nS, env.nA
        self.register_buffer("P", env.P.clone())
        self.register_buffer("r", env.r.clone())
        self.gamma = gamma
        self.lr = lr

        self.model = UnrolledPolicyIterationModel(self.P, self.r, self.nS, self.nA, K, num_unrolls, tau, beta)

    def training_step(self, batch, batch_idx):
        q_in, Pi_in = batch
        q_pred, Pi_pred = self.model(q_in, Pi_in)

        P_pi = self.model.layers[-2].compute_transition_matrix(Pi_pred).detach()
        target = self.r + self.gamma * (P_pi @ q_pred.detach()) # Detach future
        # target = self.r + self.gamma * (P_pi @ q_pred) # No detach future

        q_reshaped = q_pred.view(self.nS, self.nA)
        target_reshaped = target.view(self.nS, self.nA)

        loss = mse_loss(q_reshaped, target_reshaped)

        self.log("bellman_error", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(UnrollingDataset(self.nS, self.nA, N=500), batch_size=1, shuffle=True)

    def on_fit_end(self):
        dataset = UnrollingDataset(self.nS, self.nA, N=500)
        q_sample, Pi_sample = dataset[0]
        q_sample = q_sample.to(self.device)
        Pi_sample = Pi_sample.to(self.device)
        q, Pi_out = self.model(q_sample, Pi_sample)

        fig = plot_policy_and_value(q.view(self.nS, self.nA), Pi_out)
        wandb.log({"policy_plot": wandb.Image(fig)})
        plt.close(fig)
