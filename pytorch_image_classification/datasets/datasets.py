from typing import Tuple, Union

import pathlib

import torch
import torchvision
import yacs.config

from torch.utils.data import Dataset

from pytorch_image_classification import create_transform


class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    """Create dataset for CIFAR-10 training or evaluation."""
    if config.dataset.name != 'CIFAR10':
        raise ValueError(f"Unsupported dataset: {config.dataset.name}")
    
    # Validate dataset directory is explicitly configured
    dataset_dir = config.dataset.dataset_dir
    if not dataset_dir:
        raise ValueError(
            "dataset.dataset_dir must be explicitly specified in configuration."
        )
    
    # Expand user paths (~/)
    dataset_dir = pathlib.Path(dataset_dir).expanduser().as_posix()
    
    if is_train:
        # Train/val split using val_ratio
        dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=True,
            transform=None,
            download=config.dataset.download
        )
        
        val_ratio = config.train.val_ratio
        assert val_ratio < 1, "val_ratio must be < 1"
        val_num = int(len(dataset) * val_ratio)
        train_num = len(dataset) - val_num
        lengths = [train_num, val_num]
        train_subset, val_subset = torch.utils.data.dataset.random_split(
            dataset, lengths)

        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)
        train_dataset = SubsetDataset(train_subset, train_transform)
        val_dataset = SubsetDataset(val_subset, val_transform)
        return train_dataset, val_dataset
    else:
        # Test split
        transform = create_transform(config, is_train=False)
        dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=False,
            transform=transform,
            download=config.dataset.download
        )
        return dataset
