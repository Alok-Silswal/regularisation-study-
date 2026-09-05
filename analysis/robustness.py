#!/usr/bin/env python

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
from PIL import ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_image_classification import (
    create_model,
    get_default_config,
    update_config,
)


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "analysis"
    / "robustness"
)


# ============================================================
# CIFAR-10 normalization
# ============================================================

MEAN = np.array(
    [0.4914, 0.4822, 0.4465],
    dtype=np.float32,
)

STD = np.array(
    [0.2470, 0.2435, 0.2616],
    dtype=np.float32,
)


# ============================================================
# Deterministic robustness transforms
# ============================================================

def apply_corruption(image, corruption):
    """
    Apply a deterministic image-space distribution shift.
    """

    if corruption is None:
        return image

    if corruption == "brightness":
        return ImageEnhance.Brightness(
            image
        ).enhance(0.6)

    if corruption == "blur":
        return image.filter(
            ImageFilter.GaussianBlur(radius=1.0)
        )

    if corruption == "posterize":
        return ImageOps.posterize(
            image,
            bits=4,
        )

    raise ValueError(
        f"Unknown corruption: {corruption}"
    )


def transform_image(image, corruption=None):
    """
    Apply corruption first, then the project's CIFAR-10
    normalization.
    """

    image = apply_corruption(
        image,
        corruption,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    array = (
        array - MEAN
    ) / STD

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    ).float()

    return tensor


# ============================================================
# Dataset
# ============================================================

class RobustnessCIFAR10(CIFAR10):
    """
    CIFAR-10 test dataset with an optional deterministic
    distribution shift.
    """

    def __init__(
        self,
        root,
        corruption=None,
    ):
        self.corruption = corruption

        super().__init__(
            root=root,
            train=False,
            transform=None,
            target_transform=None,
            download=False,
        )

    def __getitem__(self, index):

        image, target = super().__getitem__(index)

        image = transform_image(
            image,
            self.corruption,
        )

        return image, target


# ============================================================
# Configuration
# ============================================================

def load_config(config_path):
    """
    Load configuration using the same mechanism as evaluate.py.
    """

    config = get_default_config()

    config.merge_from_file(
        str(config_path)
    )

    update_config(config)

    config.freeze()

    return config


# ============================================================
# Model
# ============================================================

def load_model(config, checkpoint_path):

    model = create_model(config)

    model = model.to("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):

    correct = 0
    total = 0

    loss_sum = 0.0

    criterion = torch.nn.CrossEntropyLoss(
        reduction="sum"
    )

    with torch.no_grad():

        for images, targets in loader:

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

            loss_sum += loss.item()

    accuracy = correct / total

    loss = loss_sum / total

    return accuracy, loss


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the CIFAR-10 dataset directory.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
    )

    args = parser.parse_args()

    dataset_root = pathlib.Path(
        args.dataset_root
    )

    output_dir = pathlib.Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Experiment definitions
    # --------------------------------------------------------

    experiments = [
        (
            "Baseline",
            "baseline_seed0",
            PROJECT_ROOT
            / "configs"
            / "classification"
            / "cnn_baseline.yaml",
        ),
        (
            "MixUp",
            "mixup_seed0",
            PROJECT_ROOT
            / "configs"
            / "classification"
            / "cnn_mixup.yaml",
        ),
        (
            "CutMix",
            "cutmix_seed0",
            PROJECT_ROOT
            / "configs"
            / "classification"
            / "cnn_cutmix.yaml",
        ),
        (
            "CutOut",
            "cutout_seed0",
            PROJECT_ROOT
            / "configs"
            / "classification"
            / "cnn_cutout.yaml",
        ),
        (
            "Random Erasing",
            "random_erasing_seed0",
            PROJECT_ROOT
            / "configs"
            / "classification"
            / "cnn_random_erasing.yaml",
        ),
    ]

    corruptions = [
        None,
        "brightness",
        "blur",
        "posterize",
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    print("=" * 70)
    print("Classification Robustness Analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------

    for method, experiment_dir, config_path in experiments:

        checkpoint_path = (
            PROJECT_ROOT
            / "experiments"
            / "classification"
            / "cnn"
            / experiment_dir
            / "checkpoint_00040.pth"
        )

        print()
        print("-" * 70)
        print(method)
        print("-" * 70)

        print(
            f"Checkpoint: {checkpoint_path}"
        )

        config = load_config(
            config_path
        )

        model = load_model(
            config,
            checkpoint_path,
        )

        clean_accuracy = None

        for corruption in corruptions:

            name = (
                "clean"
                if corruption is None
                else corruption
            )

            print(
                f"Evaluating {name}..."
            )

            dataset = RobustnessCIFAR10(
                root=str(dataset_root),
                corruption=corruption,
            )

            if len(dataset) != 10000:
                raise ValueError(
                    f"Expected 10,000 CIFAR-10 "
                    f"test samples, found "
                    f"{len(dataset)}."
                )

            loader = DataLoader(
                dataset,
                batch_size=256,
                shuffle=False,
                num_workers=2,
                drop_last=False,
            )

            accuracy, loss = evaluate(
                model,
                loader,
            )

            if corruption is None:

                clean_accuracy = accuracy

                degradation = 0.0
                relative_degradation = 0.0

            else:

                degradation = (
                    clean_accuracy
                    - accuracy
                )

                relative_degradation = (
                    degradation
                    / clean_accuracy
                )

            results.append(
                {
                    "method": method,
                    "corruption": name,
                    "clean_accuracy": clean_accuracy,
                    "shifted_accuracy": accuracy,
                    "absolute_degradation": degradation,
                    "relative_degradation": relative_degradation,
                    "loss": loss,
                }
            )

            print(
                f"  Accuracy: "
                f"{accuracy * 100:.2f}%"
            )

            if corruption is not None:
                print(
                    f"  Degradation: "
                    f"{degradation * 100:.2f} pp"
                )

        del model

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_file = (
        output_dir
        / "robustness_results.csv"
    )

    fieldnames = [
        "method",
        "corruption",
        "clean_accuracy",
        "shifted_accuracy",
        "absolute_degradation",
        "relative_degradation",
        "loss",
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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    for row in results:

        print(
            f"{row['method']:18s} | "
            f"{row['corruption']:10s} | "
            f"{row['shifted_accuracy'] * 100:6.2f}% | "
            f"drop "
            f"{row['absolute_degradation'] * 100:6.2f} pp"
        )

    print()
    print(
        f"Results saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()