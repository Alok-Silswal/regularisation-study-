import numpy as np
import torch


def _box(height, width, alpha):
    lam = np.random.beta(alpha, alpha)
    cut = np.sqrt(1.0 - lam)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    box_w = int(width * cut)
    box_h = int(height * cut)
    x0 = max(cx - box_w // 2, 0)
    x1 = min(cx + box_w // 2, width)
    y0 = max(cy - box_h // 2, 0)
    y1 = min(cy + box_h // 2, height)
    return y0, y1, x0, x1


def cutmix(images, masks, alpha=1.0, probability=1.0):
    if np.random.random() > probability or images.size(0) < 2:
        return images, masks
    permutation = torch.randperm(images.size(0), device=images.device)
    y0, y1, x0, x1 = _box(images.size(2), images.size(3), alpha)
    mixed_images = images.clone()
    mixed_masks = masks.clone()
    mixed_images[:, :, y0:y1, x0:x1] = images[permutation, :, y0:y1, x0:x1]
    mixed_masks[:, y0:y1, x0:x1] = masks[permutation, y0:y1, x0:x1]
    return mixed_images, mixed_masks


def cutout(images, masks, size, probability=1.0):
    if np.random.random() > probability:
        return images, masks
    height, width = images.shape[-2:]
    cy = np.random.randint(height)
    cx = np.random.randint(width)
    y0, y1 = max(cy - size // 2, 0), min(cy + size // 2, height)
    x0, x1 = max(cx - size // 2, 0), min(cx + size // 2, width)
    images = images.clone()
    images[:, :, y0:y1, x0:x1] = 0.0
    return images, masks


def classmix(images, masks, probability=1.0):
    """Copy donor foreground regions and their labels into each sample."""
    if np.random.random() > probability or images.size(0) < 2:
        return images, masks
    permutation = torch.randperm(images.size(0), device=images.device)
    donor_masks = masks[permutation]
    region = donor_masks == 1
    mixed_images = torch.where(region.unsqueeze(1), images[permutation], images)
    mixed_masks = torch.where(region, donor_masks, masks)
    return mixed_images, mixed_masks
