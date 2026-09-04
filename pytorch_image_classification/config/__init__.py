import pathlib
import torch

from .defaults import get_default_config


def update_config(config):
    """Update config with dataset-specific metadata and device settings."""
    if config.dataset.name == 'CIFAR10':
        # Only set default dataset_dir if user hasn't explicitly configured it
        if config.dataset.dataset_dir == '':
            if config.dataset.download:
                config.dataset.dataset_dir = str(
                    pathlib.Path.home() / '.torch' / 'datasets' / 'CIFAR10'
                )
            # Otherwise leave it empty - create_dataset will raise a clear error
        
        # Set CIFAR-10 metadata
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.name}")

    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
