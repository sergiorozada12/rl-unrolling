"""Neural network models for BellNet.

This module implements the core neural network components for unrolled
policy iteration including policy evaluation and improvement layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PolicyEvaluationLayer(nn.Module):
    """Policy evaluation layer using graph filters.
    
    This layer implements the policy evaluation step of policy iteration
    using graph filters to parameterize the Bellman operator.
    
    Args:
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
        nS: Number of states
        nA: Number of actions  
        K: Graph filter order
        beta: Bellman operator parameter
        shared_h: Optional shared filter coefficients
        architecture_type: Architecture type (1, 2, 3, or 5)
    """
    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int, 
                 K: int, beta: float, shared_h: Optional[nn.Parameter] = None, 
                 architecture_type: int = 1):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.K = K
        self.beta = beta
        self.architecture_type = architecture_type

        self.register_buffer("P", P)  # shape: (nS * nA, nS)
        self.register_buffer("r", r)  # shape: (nS * nA,)

        if shared_h is None:
            self.h = nn.Parameter(torch.randn(K + 1) * 0.1 )  # shape: (K + 1,) K powers and extra parameters for q
        else:
            self.h = shared_h

        # Additional parameters for different architectures
        if architecture_type == 2:
            # Architecture 2: separate parameters for r and q_0 terms
            self.w = nn.Parameter(torch.randn(K + 2) * 0.1)  # w_k for q_0 terms
        elif architecture_type == 3:
            # Architecture 3: joint filter for concatenated [r; q_0]
            self.h = nn.Parameter(torch.randn(K + 2) * 0.1)  # h_k for joint X = [r; q_0]
        elif architecture_type == 5:
            # Architecture 5: matrix filters H_k and final linear layer
            sa_dim = nS * nA
            self.H = nn.ParameterList([nn.Parameter(torch.randn(sa_dim, 2 * sa_dim) * 0.1) for _ in range(K + 2)])
            self.w_final = nn.Parameter(torch.randn(sa_dim) * 0.1)  # final linear weights

    def lift_policy_matrix(self, Pi: torch.Tensor) -> torch.Tensor:
        """Lift policy matrix to state-action space.
        
        Args:
            Pi: Policy matrix of shape (nS, nA)
            
        Returns:
            Extended policy matrix of shape (nS, nS * nA)
        """
        Pi_ext = torch.zeros(self.nS, self.nS * self.nA, device=Pi.device)
        rows = torch.arange(self.nS, device=Pi.device).repeat_interleave(self.nA)
        cols = torch.arange(self.nS * self.nA, device=Pi.device)
        Pi_ext[rows, cols] = Pi.flatten()
        return Pi_ext

    def compute_transition_matrix(self, Pi: torch.Tensor) -> torch.Tensor:
        """Compute transition matrix under policy Pi.
        
        Args:
            Pi: Policy matrix of shape (nS, nA)
            
        Returns:
            Transition matrix P_π of shape (nS*nA, nS*nA)
        """
        Pi_ext = self.lift_policy_matrix(Pi)  # shape: (nS, nS*nA)
        return self.P @ Pi_ext  # shape: (nS*nA, nS*nA)

    def forward_architecture_1(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Original architecture: q̂ = Σ(k=0 to K) h_k P^k r + h_{K+1} P^{K+1} q_0"""
        P_pi = self.compute_transition_matrix(Pi)

        q_prime = self.h[0] * self.r
        q_power = q.clone()
        r_power = self.r.clone()

        for k in range(1, self.K):
            r_power = P_pi @ r_power
            q_power = P_pi @ q_power
            q_prime += self.h[k] * r_power
        q_power = self.h[self.K] * q_power

        return q_prime + self.beta * q_power

    def forward_architecture_2(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Architecture 2: q̂ = Σ(k=0 to K) h_k P^k r + Σ(k=0 to K+1) w_k P^k q_0"""
        P_pi = self.compute_transition_matrix(Pi)

        # Reward terms: Σ(k=0 to K) h_k P^k r
        q_prime = self.h[0] * self.r
        r_power = self.r.clone()
        for k in range(1, self.K + 1):
            r_power = P_pi @ r_power
            q_prime += self.h[k] * r_power

        # Q-value terms: Σ(k=0 to K+1) w_k P^k q_0
        q_term = self.w[0] * q
        q_power = q.clone()
        for k in range(1, self.K + 2):
            q_power = P_pi @ q_power
            q_term += self.w[k] * q_power

        return q_prime + self.beta * q_term

    def forward_architecture_3(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Architecture 3: q̂ = Σ(k=0 to K+1) h_k P^k X 1, with X = [r; q_0]"""
        P_pi = self.compute_transition_matrix(Pi)

        # Concatenate X = [r; q_0]
        X = torch.cat([self.r, q], dim=0)  # shape: (2 * nS * nA,)

        # Extend P_pi to work with concatenated space
        zero_block = torch.zeros_like(P_pi)
        P_pi_ext = torch.block_diag(P_pi, P_pi)  # shape: (2*nS*nA, 2*nS*nA)

        # Apply filters to X and sum over all elements to get scalar, then broadcast
        result_scalar = self.h[0] * X.sum()
        X_power = X.clone()
        for k in range(1, self.K + 2):
            X_power = P_pi_ext @ X_power
            result_scalar += self.h[k] * X_power.sum()

        # Return the scalar broadcasted to match q shape
        return result_scalar * torch.ones_like(q)

    def forward_architecture_5(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Architecture 5: Q̂ = Σ(k=0 to K+1) P^k X H_k, with X = [r; q_0], q̂ = σ(Q̂ w)"""
        P_pi = self.compute_transition_matrix(Pi)

        # Concatenate X = [r; q_0]
        X = torch.cat([self.r, q], dim=0)  # shape: (2 * nS * nA,)

        # Extend P_pi to work with concatenated space
        P_pi_ext = torch.block_diag(P_pi, P_pi)  # shape: (2*nS*nA, 2*nS*nA)

        # Apply matrix filters H_k to get intermediate representations
        Q_hat = self.H[0] @ X  # shape: (nS * nA,)
        X_power = X.clone()
        for k in range(1, self.K + 2):
            X_power = P_pi_ext @ X_power
            Q_hat += self.H[k] @ X_power

        # Apply final linear transformation: q̂ = σ(Q̂ w)
        q_prime = torch.sigmoid(Q_hat * self.w_final)
        return q_prime

    def forward(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Forward pass for policy evaluation.
        
        Args:
            q: Current Q-values of shape (nS * nA,)
            Pi: Current policy of shape (nS, nA)
            
        Returns:
            Updated Q-values of shape (nS * nA,)
        """
        if self.architecture_type == 1:
            return self.forward_architecture_1(q, Pi)
        elif self.architecture_type == 2:
            return self.forward_architecture_2(q, Pi)
        elif self.architecture_type == 3:
            return self.forward_architecture_3(q, Pi)
        elif self.architecture_type == 5:
            return self.forward_architecture_5(q, Pi)
        else:
            raise ValueError(f"Unsupported architecture type: {self.architecture_type}")


class PolicyImprovementLayer(nn.Module):
    """Policy improvement layer using softmax.
    
    This layer implements the policy improvement step using softmax
    with temperature scaling.
    
    Args:
        nS: Number of states
        nA: Number of actions
        tau: Temperature parameter for softmax
    """
    def __init__(self, nS: int, nA: int, tau: float):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.tau = tau

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Forward pass for policy improvement.
        
        Args:
            q: Q-values of shape (nS * nA,)
            
        Returns:
            Improved policy of shape (nS, nA)
        """
        q_reshaped = q.view(self.nS, self.nA)
        Pi = F.softmax(q_reshaped / self.tau, dim=1)
        return Pi


class UnrolledPolicyIterationModel(nn.Module):
    """Unrolled policy iteration neural network.
    
    This model implements unrolled policy iteration by stacking
    alternating policy evaluation and improvement layers.
    
    Args:
        P: Transition probability tensor of shape (nS * nA, nS)
        r: Reward tensor of shape (nS * nA,)
        nS: Number of states
        nA: Number of actions
        K: Graph filter order
        num_unrolls: Number of unrolling steps
        tau: Temperature parameter
        beta: Bellman operator parameter  
        weight_sharing: Whether to share weights across layers
        architecture_type: Architecture type (1, 2, 3, or 5)
    """
    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int, 
                 K: int = 3, num_unrolls: int = 5, tau: float = 1, 
                 beta: float = 1.0, weight_sharing: bool = False,
                 architecture_type: int = 1):
        super().__init__()
        self.nS = nS
        self.nA = nA

        if weight_sharing:
            self.h = nn.Parameter(torch.randn(K + 1) * 0.1 ) # shape: (K + 1,) K powers and extra parameters for q
        else:
            self.h = None

        self.layers = nn.ModuleList()
        for _ in range(num_unrolls):
            self.layers.append(PolicyEvaluationLayer(P, r, nS, nA, K, beta, self.h, architecture_type))
            self.layers.append(PolicyImprovementLayer(nS, nA, tau))

    def forward(self, q_init: torch.Tensor, Pi_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through unrolled policy iteration.
        
        Args:
            q_init: Initial Q-values of shape (nS * nA,)
            Pi_init: Initial policy of shape (nS, nA)
            
        Returns:
            Tuple of (final_q_values, final_policy)
        """
        q = q_init.squeeze()
        Pi = Pi_init
        for layer in self.layers:
            if isinstance(layer, PolicyEvaluationLayer):
                q = layer(q, Pi)
            elif isinstance(layer, PolicyImprovementLayer):
                Pi = layer(q)
        return q, Pi
