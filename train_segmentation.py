#!/usr/bin/env python
import argparse
import json
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_semantic_segmentation.augmentations import classmix, cutmix, cutout
from pytorch_semantic_segmentation.config import get_default_config
from pytorch_semantic_segmentation.datasets import create_datasets
from pytorch_semantic_segmentation.losses import segmentation_loss
from pytorch_semantic_segmentation.metrics import dice_coefficient, iou_score
from pytorch_semantic_segmentation.models import UNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    loss, dice, iou, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            batch_size = images.size(0)
            loss += segmentation_loss(logits, masks).item() * batch_size
            dice += dice_coefficient(logits, masks) * batch_size
            iou += iou_score(logits, masks) * batch_size
            count += batch_size
    return {'loss': loss / count, 'dice': dice / count, 'iou': iou / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config.freeze()
    set_seed(config.seed)
    device = torch.device(config.device if config.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    output_dir = pathlib.Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.yaml').write_text(str(config), encoding='utf-8')

    train_set, val_set, _ = create_datasets(config)
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)
    model = UNet(config.dataset.num_classes, config.model.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    best_dice = -1.0
    log_path = output_dir / 'log.jsonl'

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            condition = config.augmentation.condition
            if condition == 'cutmix':
                images, masks = cutmix(images, masks, config.augmentation.cutmix_alpha, config.augmentation.cutmix_prob)
            elif condition == 'cutout':
                images, masks = cutout(images, masks, config.augmentation.cutout_size, config.augmentation.cutout_prob)
            elif condition == 'classmix':
                images, masks = classmix(images, masks, config.augmentation.classmix_prob)
            optimizer.zero_grad(set_to_none=True)
            segmentation_loss(model(images), masks).backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        record = {'epoch': epoch, **metrics}
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        print(record)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, output_dir / 'checkpoint_last.pth')
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')


if __name__ == '__main__':
    main()
