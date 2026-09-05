#!/usr/bin/env python

import argparse
import csv
import pathlib
import sys

import torch
from PIL import ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_semantic_segmentation.config import get_default_config
from pytorch_semantic_segmentation.datasets import create_datasets
from pytorch_semantic_segmentation.models import UNet
from pytorch_semantic_segmentation.metrics import (
    dice_coefficient,
    iou_score,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "analysis"
    / "segmentation_robustness"
)


# ============================================================
# Experiments
# ============================================================

EXPERIMENTS = [
    (
        "Baseline",
        PROJECT_ROOT
        / "configs"
        / "segmentation"
        / "unet_baseline.yaml",
        PROJECT_ROOT
        / "experiments"
        / "segmentation"
        / "unet"
        / "baseline_seed0",
    ),
    (
        "CutMix",
        PROJECT_ROOT
        / "configs"
        / "segmentation"
        / "unet_cutmix.yaml",
        PROJECT_ROOT
        / "experiments"
        / "segmentation"
        / "unet"
        / "cutmix_seed0",
    ),
    (
        "CutOut",
        PROJECT_ROOT
        / "configs"
        / "segmentation"
        / "unet_cutout.yaml",
        PROJECT_ROOT
        / "experiments"
        / "segmentation"
        / "unet"
        / "cutout_seed0",
    ),
    (
        "ClassMix",
        PROJECT_ROOT
        / "configs"
        / "segmentation"
        / "unet_classmix.yaml",
        PROJECT_ROOT
        / "experiments"
        / "segmentation"
        / "unet"
        / "classmix_seed0",
    ),
]


CORRUPTIONS = [
    None,
    "brightness",
    "blur",
    "posterize",
]


class CorruptedSegmentationDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        dataset,
        corruption=None,
    ):
        if not hasattr(dataset, "dataset") or not hasattr(dataset, "transform"):
            raise TypeError(
                "Expected a PetSegmentation test dataset with raw dataset "
                "and transform attributes."
            )

        self.raw_dataset = dataset.dataset
        self.transform = dataset.transform
        self.corruption = corruption

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        image, mask = self.raw_dataset[index]
        if self.corruption == "brightness":
            image = ImageEnhance.Brightness(image).enhance(0.6)
        elif self.corruption == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif self.corruption == "posterize":
            image = ImageOps.posterize(image, bits=4)
        elif self.corruption is not None:
            raise ValueError(f"Unknown corruption: {self.corruption}")

        transformed_image, transformed_mask = self.transform(image, mask)
        if transformed_image.ndim != 3 or transformed_image.shape[0] != 3:
            raise AssertionError("Transformed image must have shape (3, H, W)")
        if transformed_mask.ndim != 2:
            raise AssertionError("Transformed mask must have shape (H, W)")
        if transformed_image.shape[-2:] != transformed_mask.shape:
            raise AssertionError("Image and mask spatial dimensions differ")
        return transformed_image, transformed_mask


# ============================================================
# Configuration
# ============================================================

def load_config(config_path):

    config = get_default_config()

    config.merge_from_file(
        str(config_path)
    )

    return config


# ============================================================
# Model
# ============================================================

def load_model(
    config,
    checkpoint_path,
    device,
):
    model = UNet(
        config.dataset.num_classes,
        config.model.base_channels,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    dice_total = 0.0
    iou_total = 0.0

    count = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            if not torch.isfinite(logits).all():
                raise AssertionError("Model output contains NaN or Inf")

            batch_size_actual = (
                images.size(0)
            )

            dice = dice_coefficient(
                logits,
                masks,
            )

            iou = iou_score(
                logits,
                masks,
            )

            dice_total += (
                dice
                * batch_size_actual
            )

            iou_total += (
                iou
                * batch_size_actual
            )

            count += batch_size_actual

    if count == 0:
        raise AssertionError("Cannot evaluate an empty dataset")

    dice = dice_total / count
    iou = iou_total / count
    if not torch.isfinite(torch.tensor([dice, iou])).all():
        raise AssertionError("Metric result contains NaN or Inf")
    return dice, iou


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Oxford-IIIT Pet dataset root.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; auto uses CUDA when available.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "Segmentation Robustness Analysis"
    )
    print("=" * 70)

    results = []

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)

    for (
        method,
        config_path,
        experiment_dir,
    ) in EXPERIMENTS:

        checkpoint_path = experiment_dir / "best_model.pth"

        print()
        print("-" * 70)
        print(method)
        print("-" * 70)

        config = load_config(config_path)
        config.defrost()
        config.dataset.root = str(pathlib.Path(args.dataset_root).expanduser())
        config.device = device_name
        config.freeze()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = load_model(
            config,
            checkpoint_path,
            device,
        )

        # ----------------------------------------------------
        # Existing test dataset
        # ----------------------------------------------------

        _, _, test_set = create_datasets(
            config
        )
        if len(test_set) == 0:
            raise AssertionError("Oxford-IIIT Pet test set is empty")

        # ----------------------------------------------------
        # Clean evaluation
        # ----------------------------------------------------

        clean_dice, clean_iou = evaluate(
            model,
            test_set,
            device,
            config.train.batch_size,
            config.train.num_workers,
        )

        print(
            f"Clean Dice: {clean_dice:.6f}"
        )

        print(
            f"Clean IoU:  {clean_iou:.6f}"
        )

        # ----------------------------------------------------
        # Corrupted evaluation
        # ----------------------------------------------------

        for corruption in CORRUPTIONS:

            if corruption is None:
                continue

            corrupted_set = (
                CorruptedSegmentationDataset(
                    test_set,
                    corruption,
                )
            )

            dice, iou = evaluate(
                model,
                corrupted_set,
                device,
                config.train.batch_size,
                config.train.num_workers,
            )

            dice_degradation = (
                clean_dice - dice
            )

            iou_degradation = (
                clean_iou - iou
            )

            results.append(
                {
                    "method": method,
                    "corruption": corruption,
                    "clean_dice": clean_dice,
                    "shifted_dice": dice,
                    "dice_degradation": dice_degradation,
                    "clean_iou": clean_iou,
                    "shifted_iou": iou,
                    "iou_degradation": iou_degradation,
                }
            )

            print()
            print(
                f"{corruption}:"
            )

            print(
                f"  Dice: {dice:.6f}"
            )

            print(
                f"  Dice degradation: "
                f"{dice_degradation:.6f}"
            )

            print(
                f"  IoU: {iou:.6f}"
            )

            print(
                f"  IoU degradation: "
                f"{iou_degradation:.6f}"
            )

        del model

    # ========================================================
    # Save results
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / "segmentation_robustness_results.csv"
    )

    fieldnames = [
        "method",
        "corruption",
        "clean_dice",
        "shifted_dice",
        "dice_degradation",
        "clean_iou",
        "shifted_iou",
        "iou_degradation",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(output_file)


if __name__ == "__main__":
    main()