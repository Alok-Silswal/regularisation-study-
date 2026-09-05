import pathlib
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from .transforms import PairedTransform


class PetSegmentation(Dataset):
    def __init__(self, root, split, transform, download=False):
        self.dataset = torchvision.datasets.OxfordIIITPet(
            root=root, split=split, target_types='segmentation', download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        return self.transform(image, mask)


def create_datasets(config):
    if not config.dataset.root:
        raise ValueError('dataset.root must be set to the dataset parent directory')
    root = pathlib.Path(config.dataset.root).expanduser()
    transform_train = PairedTransform(config.dataset.image_size, True, config.augmentation.horizontal_flip_prob)
    transform_eval = PairedTransform(config.dataset.image_size, False)
    full_train = PetSegmentation(root, 'trainval', transform_train, config.dataset.download)
    full_eval = PetSegmentation(root, 'trainval', transform_eval, config.dataset.download)
    val_count = int(len(full_train) * config.dataset.val_ratio)
    train_count = len(full_train) - val_count
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_subset = Subset(full_train, indices[:train_count])
    val_subset = Subset(full_eval, indices[train_count:])
    test = PetSegmentation(root, 'test', transform_eval, config.dataset.download)
    return train_subset, val_subset, test
