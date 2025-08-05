# BellNet: Dynamic Programming via Graph Filters

A research project implementing BellNet, a neural network approach to reinforcement learning that uses graph filters to solve dynamic programming problems through unrolled policy iteration.

## 🔍 Overview

This project explores the intersection of graph signal processing and reinforcement learning by implementing unrolled policy iteration networks. The core idea is to parameterize the Bellman operator using graph filters and learn optimal policies through differentiable dynamic programming.

### Key Features

- **Unrolled Policy Iteration**: Neural networks that unroll the policy iteration algorithm
- **Graph Filter Parameterization**: Bellman operator implemented using learnable graph filters
- **Environment Support**: CliffWalking and MirroredCliffWalking environments
- **Transfer Learning**: Analysis of policy transferability between environments
- **Comprehensive Experiments**: Multiple analysis notebooks for different aspects

## 📁 Project Structure

```
rl-unrolling/
├── src/                           # Core library code
│   ├── algorithms/                # RL algorithms
│   │   ├── generalized_policy_iteration.py
│   │   └── unrolling_policy_iteration.py
│   ├── environments.py            # Environment implementations
│   ├── models.py                  # Neural network models
│   ├── plots.py                   # Visualization utilities
│   └── utils.py                   # Utility functions
├── experiments/                   # Experiment scripts and notebooks
│   ├── influence_K.ipynb         # Analysis of K parameter influence
│   ├── influence_transferability.ipynb  # Transfer learning analysis
│   └── influence_unroll.ipynb    # Unrolling depth analysis
├── main.py                       # Main training script
├── config.py                     # Configuration management
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sergiorozada12/rl-unrolling.git
cd rl-unrolling
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Weights & Biases (optional but recommended):
```bash
wandb login
```

## 📊 Usage

### Basic Training

Run the main training script with default parameters:

```bash
python main.py
```

### Custom Configuration

Modify the configuration in `config.py` or pass parameters directly:

```python
from main import unrl

# Train with custom parameters
unrl(
    K=10,                    # Filter order
    num_unrolls=10,         # Number of unrolling steps
    tau=100,                # Temperature parameter
    beta=1.0,               # Bellman operator parameter
    lr=1e-3,                # Learning rate
    N=500,                  # Dataset size
    weight_sharing=False    # Whether to share weights across layers
)
```

### Running Experiments

Execute individual experiment notebooks:

```bash
jupyter notebook experiments/influence_K.ipynb
```

## 🧪 Experiments

### 1. Filter Order Analysis (`influence_K.ipynb`)
Analyzes how the graph filter order K affects learning performance and convergence.

### 2. Transfer Learning Analysis (`influence_transferability.ipynb`)
Studies how policies learned on one environment transfer to related environments.

### 3. Unrolling Depth Analysis (`influence_unroll.ipynb`)
Investigates the impact of unrolling depth on learning efficiency and final performance.

## 🏗️ Architecture

### Core Components

#### UnrolledPolicyIterationModel
The main neural network that implements unrolled policy iteration:
- **PolicyEvaluationLayer**: Computes value functions using graph filters
- **PolicyImprovementLayer**: Updates policies using softmax temperature scaling

#### Environment Implementations
- **CliffWalkingEnv**: Standard cliff walking environment
- **MirroredCliffWalkingEnv**: Modified version with mirrored cliff placement

### Graph Filter Parameterization

The Bellman operator is parameterized as:
```
T_π = h_0 * R + Σ(k=1 to K) h_k * (P_π)^k * R + β * h_{K+1} * (P_π)^K * V
```

Where:
- `h_k` are learnable filter coefficients
- `P_π` is the transition matrix under policy π
- `R` is the reward vector
- `β` controls the influence of the previous value estimate

## 📈 Monitoring

The project integrates with Weights & Biases for experiment tracking:

- Training loss and metrics
- Policy visualizations
- Filter coefficient analysis
- Transfer learning performance

## 🔧 Configuration

Key hyperparameters can be configured in `config.py`:

```python
CONFIG = {
    'model': {
        'K': 10,                # Graph filter order
        'num_unrolls': 10,      # Unrolling depth
        'tau': 100,             # Temperature parameter
        'beta': 1.0,            # Bellman parameter
        'weight_sharing': False  # Weight sharing across layers
    },
    'training': {
        'lr': 1e-3,             # Learning rate
        'max_epochs': 5000,     # Training epochs
        'N': 500,               # Dataset size
    },
    'logging': {
        'project': 'rl-unrolling',
        'freq_plots': 10,       # Plot frequency
    }
}
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{rozada2025unrollingdynamicprogramminggraph,
  title={Unrolling Dynamic Programming via Graph Filters}, 
  author={Sergio Rozada and Samuel Rey and Gonzalo Mateos and Antonio G. Marques},
  year={2025},
  eprint={2507.21705},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2507.21705}, 
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions and support, please open an issue on GitHub or contact s.rozada.2019@alumnos.urjc.es.