"""
BellNet: Dynamic Programming via Graph Filters

A research library implementing unrolled policy iteration networks
for reinforcement learning using graph signal processing.
"""

from .environments import CliffWalkingEnv, MirroredCliffWalkingEnv
from .models import (
    PolicyEvaluationLayer,
    PolicyImprovementLayer, 
    UnrolledPolicyIterationModel
)
from .utils import *
from .plots import plot_policy_and_value, plot_Pi, plot_filter_coefs

__version__ = "1.0.0"
__author__ = "BellNet Research Team"

__all__ = [
    "CliffWalkingEnv",
    "MirroredCliffWalkingEnv", 
    "PolicyEvaluationLayer",
    "PolicyImprovementLayer",
    "UnrolledPolicyIterationModel",
    "plot_policy_and_value", 
    "plot_Pi",
    "plot_filter_coefs"
]