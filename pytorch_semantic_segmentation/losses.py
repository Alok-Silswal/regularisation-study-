import torch
import torch.nn.functional as F


def segmentation_loss(logits, target, dice_weight=0.5):
    """Cross-entropy plus soft foreground Dice loss."""
    ce = F.cross_entropy(logits, target)
    probabilities = logits.softmax(dim=1)[:, 1]
    foreground = (target == 1).float()
    intersection = (probabilities * foreground).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + foreground.flatten(1).sum(1)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return (1.0 - dice_weight) * ce + dice_weight * dice_loss
