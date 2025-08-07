#!/usr/bin/env python3
"""
Simple script to run individual q-bar initialization experiments.
"""

import os
import sys
import wandb
from main import unrl

def run_single_experiment(init_q: str):
    """Run a single experiment with the given initialization."""
    print(f"Running experiment with init_q='{init_q}'")
    
    # Run experiment with reduced parameters for speed
    unrl(
        K=5,
        num_unrolls=5,
        tau=100,
        beta=1.0,
        lr=5e-3,
        N=50,
        weight_sharing=False,
        group=f"q_init_comparison",
        init_q=init_q
    )
    
    print(f"Completed experiment for init_q='{init_q}'")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['zeros', 'ones', 'random']:
        print("Usage: python run_simple_experiments.py [zeros|ones|random]")
        sys.exit(1)
    
    init_q = sys.argv[1]
    run_single_experiment(init_q)