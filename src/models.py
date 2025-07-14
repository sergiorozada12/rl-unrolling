import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyEvaluationLayer(nn.Module):
    def __init__(self, P, r, nS, nA, K, beta, shared_h=None):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.K = K
        self.beta = beta

        self.register_buffer("P", P)  # shape: (nS * nA, nS)
        self.register_buffer("r", r)  # shape: (nS * nA,)

        if shared_h is None:
            self.h = nn.Parameter(torch.randn(K + 1))  # shape: (K + 1,) K powers and extra parameters for q
        else:
            self.h = shared_h

    def lift_policy_matrix(self, Pi):
        Pi_ext = torch.zeros(self.nS, self.nS * self.nA, device=Pi.device)
        rows = torch.arange(self.nS, device=Pi.device).repeat_interleave(self.nA)
        cols = torch.arange(self.nS * self.nA, device=Pi.device)
        Pi_ext[rows, cols] = Pi.flatten()
        return Pi_ext

    def compute_transition_matrix(self, Pi):
        Pi_ext = self.lift_policy_matrix(Pi)  # shape: (nS, nS*nA)
        return self.P @ Pi_ext  # shape: (nS*nA, nS*nA)

    def forward(self, q, Pi):
        P_pi = self.compute_transition_matrix(Pi)

        # q_prime = self.h[0] * self.r.clone()
        q_prime = self.h[0] * self.r
        q_power = q.clone()
        r_power = self.r.clone()

        for k in range(1, self.K):
            r_power = P_pi @ r_power
            q_power = P_pi @ q_power
            q_prime += self.h[k] * r_power
        q_power = self.h[self.K] * q_power

        return q_prime + self.beta * q_power


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
    def __init__(self, P, r, nS, nA, K=3, num_unrolls=5, tau=1, beta=1.0, weight_sharing=False):
        super().__init__()
        self.nS = nS
        self.nA = nA

        if weight_sharing:
            self.h = nn.Parameter(torch.randn(K + 1))  # shape: (K + 1,) K powers and extra parameters for q
        else:
            self.h = None

        self.layers = nn.ModuleList()
        for _ in range(num_unrolls):
            self.layers.append(PolicyEvaluationLayer(P, r, nS, nA, K, beta, self.h))
            self.layers.append(PolicyImprovementLayer(nS, nA, tau))

    def forward(self, q_init, Pi_init):
        q = q_init.squeeze()
        Pi = Pi_init
        for layer in self.layers:
            if isinstance(layer, PolicyEvaluationLayer):
                q = layer(q, Pi)
            elif isinstance(layer, PolicyImprovementLayer):
                Pi = layer(q)
        return q, Pi
