import torch


def _foreground_stats(logits, target):
    prediction = logits.argmax(dim=1) == 1
    target = target == 1
    intersection = (prediction & target).flatten(1).sum(1).float()
    prediction_area = prediction.flatten(1).sum(1).float()
    target_area = target.flatten(1).sum(1).float()
    return intersection, prediction_area, target_area


def dice_coefficient(logits, target, eps=1e-6):
    intersection, prediction_area, target_area = _foreground_stats(logits, target)
    return ((2 * intersection + eps) /
            (prediction_area + target_area + eps)).mean().item()


def iou_score(logits, target, eps=1e-6):
    intersection, prediction_area, target_area = _foreground_stats(logits, target)
    union = prediction_area + target_area - intersection
    return ((intersection + eps) / (union + eps)).mean().item()
