"""
BellNet: Unrolling Dynamic Programming via Graph Filters

A research library implementing learnable policy iteration
using graph signal processing for efficient reinforcement
learning.
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
__author__ = "Sergio Rozada, Samuel Rey, Gonzalo Mateos, and Antonio G. Marques"

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