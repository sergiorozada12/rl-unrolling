"""
Configuration management for BellNet project.

This module centralizes all configuration parameters for training,
model architecture, and logging settings.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    K: int = 10                    # Graph filter order
    num_unrolls: int = 10          # Number of unrolling steps
    tau: float = 100.0             # Temperature parameter for policy improvement
    beta: float = 1.0              # Bellman operator parameter
    weight_sharing: bool = False   # Whether to share weights across layers
    gamma: float = 0.99            # Discount factor


@dataclass 
class TrainingConfig:
    """Training configuration."""
    lr: float = 1e-3               # Learning rate
    max_epochs: int = 5000         # Maximum training epochs
    N: int = 500                   # Dataset size
    accelerator: str = "cpu"       # Training accelerator ("cpu", "gpu", "auto")
    log_every_n_steps: int = 1     # Logging frequency


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    env_name: str = "CliffWalking"         # Environment name
    test_env_name: str = "MirroredCliffWalking"  # Test environment name
    goal_row: int = 0              # Goal row for environments


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    project: str = "rl-unrolling"        # W&B project name
    entity: Optional[str] = None          # W&B entity
    freq_plots: int = 10                  # Plot frequency
    save_checkpoints: bool = True         # Whether to save checkpoints
    checkpoint_dir: str = "checkpoints"   # Checkpoint directory


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    logging: LoggingConfig = LoggingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'environment': self.environment.__dict__,
            'logging': self.logging.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            environment=EnvironmentConfig(**config_dict.get('environment', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )


# Default configuration
DEFAULT_CONFIG = Config()

# Environment-specific configurations
CONFIGS = {
    'default': Config(),
    
    'fast_training': Config(
        training=TrainingConfig(
            max_epochs=1000,
            N=100
        ),
        logging=LoggingConfig(freq_plots=5)
    ),
    
    'high_capacity': Config(
        model=ModelConfig(
            K=20,
            num_unrolls=20,
            weight_sharing=True
        ),
        training=TrainingConfig(
            lr=5e-4,
            max_epochs=10000
        )
    ),
    
    'transfer_learning': Config(
        model=ModelConfig(K=15, num_unrolls=15),
        training=TrainingConfig(N=1000, max_epochs=3000),
        logging=LoggingConfig(project="rl-unrolling-transfer")
    )
}


def get_config(config_name: str = 'default') -> Config:
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        Config object
        
    Raises:
        KeyError: If config_name is not found
    """
    if config_name not in CONFIGS:
        available = list(CONFIGS.keys())
        raise KeyError(f"Config '{config_name}' not found. Available: {available}")
    
    return CONFIGS[config_name]


def load_config_from_env() -> Config:
    """
    Load configuration from environment variables.
    
    Environment variables should be prefixed with BELLNET_
    Examples:
        BELLNET_MODEL_K=15
        BELLNET_TRAINING_LR=0.001
        BELLNET_LOGGING_PROJECT=my-project
    """
    config = Config()
    
    # Model parameters
    if os.getenv('BELLNET_MODEL_K'):
        config.model.K = int(os.getenv('BELLNET_MODEL_K'))
    if os.getenv('BELLNET_MODEL_NUM_UNROLLS'):
        config.model.num_unrolls = int(os.getenv('BELLNET_MODEL_NUM_UNROLLS'))
    if os.getenv('BELLNET_MODEL_TAU'):
        config.model.tau = float(os.getenv('BELLNET_MODEL_TAU'))
    if os.getenv('BELLNET_MODEL_BETA'):
        config.model.beta = float(os.getenv('BELLNET_MODEL_BETA'))
    if os.getenv('BELLNET_MODEL_WEIGHT_SHARING'):
        config.model.weight_sharing = os.getenv('BELLNET_MODEL_WEIGHT_SHARING').lower() == 'true'
    
    # Training parameters
    if os.getenv('BELLNET_TRAINING_LR'):
        config.training.lr = float(os.getenv('BELLNET_TRAINING_LR'))
    if os.getenv('BELLNET_TRAINING_MAX_EPOCHS'):
        config.training.max_epochs = int(os.getenv('BELLNET_TRAINING_MAX_EPOCHS'))
    if os.getenv('BELLNET_TRAINING_N'):
        config.training.N = int(os.getenv('BELLNET_TRAINING_N'))
    if os.getenv('BELLNET_TRAINING_ACCELERATOR'):
        config.training.accelerator = os.getenv('BELLNET_TRAINING_ACCELERATOR')
    
    # Logging parameters
    if os.getenv('BELLNET_LOGGING_PROJECT'):
        config.logging.project = os.getenv('BELLNET_LOGGING_PROJECT')
    if os.getenv('BELLNET_LOGGING_ENTITY'):
        config.logging.entity = os.getenv('BELLNET_LOGGING_ENTITY')
    if os.getenv('BELLNET_LOGGING_FREQ_PLOTS'):
        config.logging.freq_plots = int(os.getenv('BELLNET_LOGGING_FREQ_PLOTS'))
    
    return config


def update_config(config: Config, **kwargs) -> Config:
    """
    Update configuration with keyword arguments.
    
    Args:
        config: Base configuration
        **kwargs: Parameters to update (use dot notation for nested, e.g., model.K=15)
        
    Returns:
        Updated configuration
    """
    updated_config = Config(
        model=ModelConfig(**config.model.__dict__),
        training=TrainingConfig(**config.training.__dict__),
        environment=EnvironmentConfig(**config.environment.__dict__),
        logging=LoggingConfig(**config.logging.__dict__)
    )
    
    for key, value in kwargs.items():
        if '.' in key:
            section, param = key.split('.', 1)
            if hasattr(updated_config, section):
                section_config = getattr(updated_config, section)
                if hasattr(section_config, param):
                    setattr(section_config, param, value)
                else:
                    raise AttributeError(f"Parameter '{param}' not found in section '{section}'")
            else:
                raise AttributeError(f"Section '{section}' not found in config")
        else:
            if hasattr(updated_config, key):
                setattr(updated_config, key, value)
            else:
                raise AttributeError(f"Parameter '{key}' not found in config")
    
    return updated_config


if __name__ == "__main__":
    # Example usage
    config = get_config('default')
    print("Default config:")
    print(config.to_dict())
    
    # Update config
    updated_config = update_config(config, **{'model.K': 20, 'training.lr': 0.0005})
    print("\nUpdated config:")
    print(updated_config.model.K, updated_config.training.lr)
    
    # Load from environment
    env_config = load_config_from_env()
    print("\nEnvironment config:")
    print(env_config.to_dict())