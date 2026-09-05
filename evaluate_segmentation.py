#!/usr/bin/env python

import argparse
import pathlib
import torch
from torch.utils.data import DataLoader

from pytorch_semantic_segmentation.config import get_default_config
from pytorch_semantic_segmentation.datasets import create_datasets
from pytorch_semantic_segmentation.metrics import dice_coefficient, iou_score
from pytorch_semantic_segmentation.models import UNet


def evaluate(config, checkpoint):
    device = torch.device(
        config.device
        if config.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )

    _, _, test_set = create_datasets(config)

    loader = DataLoader(
        test_set,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    model = UNet(
        config.dataset.num_classes,
        config.model.base_channels,
    ).to(device)

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dice_total = 0.0
    iou_total = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            batch_size = images.size(0)

            dice_total += dice_coefficient(logits, masks) * batch_size
            iou_total += iou_score(logits, masks) * batch_size
            count += batch_size

    return dice_total / count, iou_total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config.freeze()

    dice, iou = evaluate(config, args.checkpoint)

    print(f"Test Dice: {dice:.6f}")
    print(f"Test IoU:  {iou:.6f}")


if __name__ == "__main__":
    main()