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
    """
    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int, 
                 K: int, beta: float, shared_h: Optional[nn.Parameter] = None):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.K = K
        self.beta = beta

        self.register_buffer("P", P)  # shape: (nS * nA, nS)
        self.register_buffer("r", r)  # shape: (nS * nA,)

        if shared_h is None:
            self.h = nn.Parameter(torch.randn(K + 1) * 0.1 )  # shape: (K + 1,) K powers and extra parameters for q
        else:
            self.h = shared_h

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

    def forward(self, q: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
        """Forward pass for policy evaluation.
        
        Args:
            q: Current Q-values of shape (nS * nA,)
            Pi: Current policy of shape (nS, nA)
            
        Returns:
            Updated Q-values of shape (nS * nA,)
        """
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
    """
    def __init__(self, P: torch.Tensor, r: torch.Tensor, nS: int, nA: int, 
                 K: int = 3, num_unrolls: int = 5, tau: float = 1, 
                 beta: float = 1.0, weight_sharing: bool = False):
        super().__init__()
        self.nS = nS
        self.nA = nA

        if weight_sharing:
            self.h = nn.Parameter(torch.randn(K + 1) * 0.1 ) # shape: (K + 1,) K powers and extra parameters for q
        else:
            self.h = None

        self.layers = nn.ModuleList()
        for _ in range(num_unrolls):
            self.layers.append(PolicyEvaluationLayer(P, r, nS, nA, K, beta, self.h))
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
