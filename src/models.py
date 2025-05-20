import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyEvaluationLayer(nn.Module):
    def __init__(self, P, r, nS, nA, K):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.K = K

        self.register_buffer("P", P)  # shape: (nS * nA, nS)
        self.register_buffer("r", r)  # shape: (nS * nA,)

        self.h = nn.Parameter(torch.randn(K))  # shape: (K,)

    def lift_policy_matrix(self, Pi):
        Pi_ext = torch.zeros(self.nS, self.nS * self.nA, device=Pi.device)
        rows = torch.arange(self.nS, device=Pi.device).repeat_interleave(self.nA)
        cols = torch.arange(self.nS * self.nA, device=Pi.device)
        Pi_ext[rows, cols] = Pi.flatten()
        return Pi_ext

    def compute_transition_matrix(self, Pi):
        Pi_ext = self.lift_policy_matrix(Pi)  # shape: (nS, nS*nA)
        return self.P @ Pi_ext  # shape: (nS*nA, nS*nA)

    def forward(self, Pi):
        P_pi = self.compute_transition_matrix(Pi)
        z = self.r
        q = self.h[0] * z
        z_power = z.clone()

        for k in range(1, self.K):
            z_power = P_pi @ z_power
            q += self.h[k] * z_power

        return q


class PolicyImprovementLayer(nn.Module):
    def __init__(self, nS, nA, tau):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.tau = tau

    def forward(self, q):
        q_reshaped = q.view(self.nS, self.nA)
        Pi = F.softmax(q_reshaped / self.tau, dim=1)
        return Pi


class UnrolledPolicyIterationModel(nn.Module):
    def __init__(self, P, r, nS, nA, K=3, num_unrolls=5, tau=1):
        super().__init__()
        self.nS = nS
        self.nA = nA

        self.layers = nn.ModuleList()
        for _ in range(num_unrolls):
            self.layers.append(PolicyEvaluationLayer(P, r, nS, nA, K))
            self.layers.append(PolicyImprovementLayer(nS, nA, tau))

    def forward(self, Pi_init):
        Pi = Pi_init
        q = None
        for layer in self.layers:
            if isinstance(layer, PolicyEvaluationLayer):
                q = layer(Pi)
            elif isinstance(layer, PolicyImprovementLayer):
                Pi = layer(q)
        return q, Pi
