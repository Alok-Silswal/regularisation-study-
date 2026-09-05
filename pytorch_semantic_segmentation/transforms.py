import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class PairedTransform:
    def __init__(self, image_size, train, flip_probability=0.5):
        self.image_size = image_size
        self.train = train
        self.flip_probability = flip_probability

    def __call__(self, image, mask):
        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask, copy=True), dtype=torch.long)
        image = F.interpolate(image.unsqueeze(0), (self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask[None, None].float(), (self.image_size, self.image_size), mode='nearest').squeeze().long()
        if self.train and random.random() < self.flip_probability:
            image = image.flip(-1)
            mask = mask.flip(-1)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, (mask == 1).long()
