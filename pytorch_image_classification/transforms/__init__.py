from typing import Callable, Tuple

import numpy as np
import torchvision
import yacs.config

from .transforms import (
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from .cutout import Cutout
from .random_erasing import RandomErasing


def _get_dataset_stats(
        config: yacs.config.CfgNode) -> Tuple[np.ndarray, np.ndarray]:
    """Get dataset-specific statistics for normalization."""
    name = config.dataset.name
    if name == 'CIFAR10':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return mean, std


def create_transform(config: yacs.config.CfgNode, is_train: bool) -> Callable:
    """Create transform pipeline for training or validation."""
    if config.model.type != 'cifar':
        raise ValueError(f"Unsupported model type: {config.model.type}")
    return create_cifar_transform(config, is_train)


def create_cifar_transform(config: yacs.config.CfgNode,
                           is_train: bool) -> Callable:
    """Create CIFAR-10 transform pipeline."""
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_random_crop:
            transforms.append(RandomCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))

        transforms.append(Normalize(mean, std))

        if config.augmentation.use_cutout:
            transforms.append(Cutout(config))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))

        transforms.append(ToTensor())
    else:
        transforms = [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)
