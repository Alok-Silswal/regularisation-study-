import torch

from .defaults import get_default_config


def update_config(config):
    """Update config with dataset-specific metadata and device settings."""
    if config.dataset.name == 'CIFAR10':
        # Set CIFAR-10 metadata
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.name}")

    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
