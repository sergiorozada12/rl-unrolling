# BellNet Experiments Usage Guide

This document explains how to use the refactored experiment system.

## Quick Start

### Run All Experiments
```bash
python main.py all --runs 5
```

### Run Specific Experiments

#### 1. Influence of K (Filter Order)
```bash
# Basic usage
python main.py influence-k

# Custom parameters
python main.py influence-k --k-values 1 2 3 5 10 15 --unroll-values 5 10 --runs 3
```

#### 2. Transferability Experiments
```bash
# Basic usage
python main.py transferability

# Custom parameters  
python main.py transferability --k-values 5 10 15 --runs 5
```

#### 3. Number of Unrolls Experiments
```bash
# Basic usage
python main.py unroll

# Custom parameters
python main.py unroll --k-values 5 10 --unroll-start 2 --unroll-stop 11 --unroll-step 2
```

#### 4. BN-5 WS Specific Experiments
```bash
# Run the specific configuration (BN-5 WS only, random init, no detach)
python main.py bn5-ws
```

#### 5. Single Experiments
```bash
# Policy iteration
python main.py single --model-type policy_iteration --max-eval-iters 10 --max-epochs 20

# Unrolling
python main.py single --model-type unrolling --k-value 10 --num-unrolls 5 --weight-sharing --init-q random
```

## Common Options

- `--runs N`: Number of statistical runs (default: 5)
- `--use-logger`: Enable W&B logging
- `--no-save`: Disable saving results to files
- `--results-path PATH`: Custom results directory
- `--quiet`: Reduce output verbosity
- `--config NAME`: Use specific configuration (default, fast_training, etc.)

## Individual Runner Scripts

You can also run individual experiment types directly:

```bash
# Influence of K
python src/experiments/run_influence_k.py --unrolls 5 10 15 --runs 5

# Transferability
python src/experiments/run_influence_transferability.py --k-values 5 10 --runs 5

# Number of unrolls
python src/experiments/run_influence_unroll.py --k-values 5 10 15 --runs 10

# BN-5 WS specific
python experiments/run_specific_experiments.py
```

## Configuration System

The system uses the centralized configuration in `config.py`. You can:

1. Use predefined configurations:
   - `default`: Standard settings
   - `fast_training`: Reduced epochs for quick testing
   - `high_capacity`: More complex models
   - `transfer_learning`: Optimized for transfer experiments

2. Override specific parameters via command line

## Results Structure

Results are saved in organized directories:

```
results/
├── filter_order/          # Influence of K results
├── transfer/              # Transferability results  
├── n_unrolls/             # Number of unrolls results
└── bn5_ws_specific/       # BN-5 WS specific results
```

Each experiment saves:
- `.npz` files with raw data
- `.csv` files with processed results
- Plots are displayed during execution

## Examples

### Reproduce Paper Figures

For the BN-5 WS specific experiments (with random init and no detach):
```bash
python main.py bn5-ws --runs 5
```

For all paper figures with standard settings:
```bash
python main.py all --runs 5 --use-logger
```

### Quick Testing

For fast testing with reduced runs:
```bash
python main.py influence-k --config fast_training --runs 2 --quiet
```

### Custom Experiment

For a specific custom experiment:
```bash
python main.py single --model-type unrolling --k-value 5 --num-unrolls 5 --weight-sharing --init-q random --use-logger --group "custom-test"
```