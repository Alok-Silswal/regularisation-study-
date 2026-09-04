from typing import Callable, Tuple

import torch.nn as nn
import yacs.config

from .cutmix import CutMixLoss
from .mixup import MixupLoss


def create_loss(config: yacs.config.CfgNode) -> Tuple[Callable, Callable]:
    if config.augmentation.use_mixup:
        train_loss = MixupLoss(reduction='mean')
    elif config.augmentation.use_cutmix:
        train_loss = CutMixLoss(reduction='mean')
    else:
        train_loss = nn.CrossEntropyLoss(reduction='mean')
    val_loss = nn.CrossEntropyLoss(reduction='mean')
    return train_loss, val_loss
