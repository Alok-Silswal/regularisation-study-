#!/usr/bin/env python

import argparse
import csv
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pytorch_semantic_segmentation.config import get_default_config
from pytorch_semantic_segmentation.datasets import create_datasets
from pytorch_semantic_segmentation.models import UNet


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "analysis"
    / "segmentation_calibration"
)

NUM_BINS = 10


# ============================================================
# Configuration
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


# ============================================================
# ECE
# ============================================================

def compute_pixel_ece(
    probabilities,
    predictions,
    targets,
    num_bins=10,
):
    """
    Compute pixel-level Expected Calibration Error.

    Each pixel is treated as one prediction.

    Confidence:
        maximum predicted class probability

    Accuracy:
        whether predicted class equals target class
    """

    confidence = probabilities.max(
        axis=1
    )

    correctness = (
        predictions == targets
    ).astype(np.float64)

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    ece = 0.0

    bin_data = []

    total_pixels = len(targets)

    for i in range(num_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == num_bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:

            bin_data.append(
                {
                    "confidence": np.nan,
                    "accuracy": np.nan,
                    "count": 0,
                }
            )

            continue

        mean_confidence = float(
            confidence[mask].mean()
        )

        mean_accuracy = float(
            correctness[mask].mean()
        )

        ece += (
            count / total_pixels
        ) * abs(
            mean_accuracy
            - mean_confidence
        )

        bin_data.append(
            {
                "confidence": mean_confidence,
                "accuracy": mean_accuracy,
                "count": count,
            }
        )

    return float(ece), bin_data


# ============================================================
# Model
# ============================================================

def load_config(config_path):

    config = get_default_config()

    config.merge_from_file(
        str(config_path)
    )

    config.freeze()

    return config


def load_model(config, checkpoint_path):

    device = torch.device("cpu")

    model = UNet(
        config.dataset.num_classes,
        config.model.base_channels,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
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

def evaluate_model(
    model,
    test_set,
    batch_size,
    num_workers,
):
    """
    Collect per-pixel probabilities, predictions,
    and ground-truth masks.
    """

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    probabilities = []
    predictions = []
    targets = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to("cpu")
            masks = masks.to("cpu")

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1,
            )

            preds = probs.argmax(
                dim=1
            )

            # Convert:
            # [B, C, H, W]
            # into:
            # [B*H*W, C]

            probabilities.append(
                probs
                .permute(0, 2, 3, 1)
                .reshape(-1, probs.shape[1])
                .cpu()
                .numpy()
            )

            predictions.append(
                preds
                .reshape(-1)
                .cpu()
                .numpy()
            )

            targets.append(
                masks
                .reshape(-1)
                .cpu()
                .numpy()
            )

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    return (
        probabilities,
        predictions,
        targets,
    )


# ============================================================
# Reliability diagram
# ============================================================

def plot_reliability_diagram(
    results,
    output_path,
):

    plt.figure(figsize=(8, 8))

    for method, result in results.items():

        bins = result["bins"]

        confidence = np.array(
            [
                b["confidence"]
                for b in bins
            ]
        )

        accuracy = np.array(
            [
                b["accuracy"]
                for b in bins
            ]
        )

        valid = ~np.isnan(
            confidence
        )

        plt.plot(
            confidence[valid],
            accuracy[valid],
            marker="o",
            label=method,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel(
        "Mean Predicted Confidence"
    )

    plt.ylabel(
        "Pixel Accuracy"
    )

    plt.title(
        "Pixel-level Reliability Diagram — Oxford-IIIT Pet"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


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

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "Segmentation Calibration Analysis"
    )
    print("=" * 70)

    results = {}

    for (
        method,
        config_path,
        experiment_dir,
    ) in EXPERIMENTS:

        checkpoint_path = (
            experiment_dir
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

        # ----------------------------------------------------
        # Test dataset
        # ----------------------------------------------------

        _, _, test_set = create_datasets(
            config
        )

        print(
            f"Test samples: {len(test_set)}"
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        (
            probabilities,
            predictions,
            targets,
        ) = evaluate_model(
            model,
            test_set,
            config.train.batch_size,
            config.train.num_workers,
        )

        if len(predictions) != len(targets):
            raise ValueError(
                "Prediction/target size mismatch."
            )

        print(
            f"Total pixels evaluated: "
            f"{len(targets):,}"
        )

        # ----------------------------------------------------
        # Pixel ECE
        # ----------------------------------------------------

        ece, bins = compute_pixel_ece(
            probabilities,
            predictions,
            targets,
            NUM_BINS,
        )

        results[method] = {
            "ece": ece,
            "bins": bins,
        }

        print(
            f"Pixel-level ECE: {ece:.6f}"
        )

        del model

    # ========================================================
    # Save results
    # ========================================================

    results_file = (
        OUTPUT_DIR
        / "segmentation_calibration_results.csv"
    )

    with open(
        results_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Method",
                "Pixel_ECE",
            ]
        )

        for method, result in results.items():

            writer.writerow(
                [
                    method,
                    f"{result['ece']:.6f}",
                ]
            )

    # ========================================================
    # Reliability diagram
    # ========================================================

    figure_file = (
        OUTPUT_DIR
        / "segmentation_reliability_diagram.png"
    )

    plot_reliability_diagram(
        results,
        figure_file,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for method, result in results.items():

        print(
            f"{method:15s} "
            f"Pixel ECE = {result['ece']:.6f}"
        )

    print()
    print(
        f"Results saved to:\n{results_file}"
    )

    print(
        f"Figure saved to:\n{figure_file}"
    )

    print()
    print(
        "Segmentation calibration complete."
    )


if __name__ == "__main__":
    main()