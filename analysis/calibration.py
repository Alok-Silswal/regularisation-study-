from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR10


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Change this if your CIFAR-10 dataset is stored elsewhere.
CIFAR10_ROOT = PROJECT_ROOT / "datasets" / "cifar10"

PREDICTION_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "classification"
    / "cnn"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "analysis"
    / "calibration"
)

METHODS = {
    "baseline_seed0": "Baseline",
    "mixup_seed0": "MixUp",
    "cutmix_seed0": "CutMix",
    "cutout_seed0": "CutOut",
    "random_erasing_seed0": "Random Erasing",
}

NUM_BINS = 10


def resolve_cifar10_root(dataset_root=None):
    """Return a root containing torchvision's CIFAR-10 batch directory."""
    candidates = []
    if dataset_root is not None:
        candidates.append(Path(dataset_root).expanduser())
    candidates.extend([
        CIFAR10_ROOT,
        Path('/kaggle/input/datasets/aloksilswal/cifar-10'),
        Path('/kaggle/working/cifar10'),
    ])

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        checked.append(candidate)
        if (candidate / 'cifar-10-batches-py' / 'test_batch').is_file():
            return candidate

    raise FileNotFoundError(
        'Could not find torchvision CIFAR-10 files. Expected '
        'cifar-10-batches-py/test_batch under one of: '
        + ', '.join(str(path) for path in checked)
    )


# ============================================================
# Data
# ============================================================

def load_ground_truth(dataset_root=None):
    """
    Load the ground-truth CIFAR-10 test labels.

    No downloading is performed. The dataset must already
    exist locally.
    """
    root = resolve_cifar10_root(dataset_root)
    dataset = CIFAR10(
        root=str(root),
        train=False,
        download=False,
    )

    return np.asarray(dataset.targets)


# ============================================================
# Calibration
# ============================================================

def compute_ece(probs, targets, num_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Confidence is the maximum predicted probability.
    Accuracy is whether the predicted class matches the
    ground-truth target.

    Returns:
        ece: scalar ECE value
        bin_data: information used for the reliability diagram
    """

    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correctness = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    bin_data = []

    n = len(targets)

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == num_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = np.sum(mask)

        if count == 0:
            bin_data.append(
                {
                    "confidence": np.nan,
                    "accuracy": np.nan,
                    "count": 0,
                }
            )
            continue

        bin_confidence = np.mean(confidence[mask])
        bin_accuracy = np.mean(correctness[mask])

        ece += (count / n) * abs(
            bin_accuracy - bin_confidence
        )

        bin_data.append(
            {
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "count": int(count),
            }
        )

    return float(ece), bin_data


# ============================================================
# Reliability Diagram
# ============================================================

def plot_reliability_diagrams(results, output_path):
    """
    Plot reliability diagrams for all classification methods.
    """

    plt.figure(figsize=(8, 8))

    for method, result in results.items():

        bin_data = result["bins"]

        confidences = np.array(
            [b["confidence"] for b in bin_data]
        )

        accuracies = np.array(
            [b["accuracy"] for b in bin_data]
        )

        valid = ~np.isnan(confidences)

        plt.plot(
            confidences[valid],
            accuracies[valid],
            marker="o",
            label=method,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Diagram — CIFAR-10")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-root',
        default=None,
        help='Directory containing cifar-10-batches-py.',
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Calibration Analysis — CIFAR-10")
    print("=" * 60)

    # --------------------------------------------------------
    # Load ground-truth labels
    # --------------------------------------------------------

    targets = load_ground_truth(args.dataset_root)

    print(f"Ground-truth samples: {len(targets)}")

    if len(targets) != 10000:
        raise ValueError(
            f"Expected 10,000 CIFAR-10 test samples, "
            f"found {len(targets)}."
        )

    # --------------------------------------------------------
    # Evaluate all methods
    # --------------------------------------------------------

    results = {}

    for directory, method_name in METHODS.items():

        prediction_file = (
            PREDICTION_ROOT
            / directory
            / "test"
            / "predictions.npz"
        )

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Prediction file not found:\n{prediction_file}"
            )

        data = np.load(prediction_file)

        probs = data["probs"]

        # Basic artifact validation
        if probs.shape != (10000, 10):
            raise ValueError(
                f"{method_name}: unexpected probability shape "
                f"{probs.shape}"
            )

        if len(targets) != len(probs):
            raise ValueError(
                f"{method_name}: number of predictions does not "
                f"match number of targets."
            )

        # Verify probabilities
        probability_sums = probs.sum(axis=1)

        if not np.allclose(
            probability_sums,
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"{method_name}: invalid probability sums."
            )

        # Calculate ECE
        ece, bins = compute_ece(
            probs,
            targets,
            num_bins=NUM_BINS,
        )

        results[method_name] = {
            "ece": ece,
            "bins": bins,
        }

        print(
            f"{method_name:20s} ECE = {ece:.6f}"
        )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    results_file = OUTPUT_DIR / "calibration_results.csv"

    with open(results_file, "w", encoding="utf-8") as f:

        f.write("Method,ECE\n")

        for method, result in results.items():
            f.write(
                f"{method},{result['ece']:.6f}\n"
            )

    # --------------------------------------------------------
    # Reliability diagram
    # --------------------------------------------------------

    figure_file = OUTPUT_DIR / "reliability_diagram.png"

    plot_reliability_diagrams(
        results,
        figure_file,
    )

    print()
    print(f"Results saved to: {results_file}")
    print(f"Figure saved to:  {figure_file}")
    print()
    print("Calibration analysis complete.")


if __name__ == "__main__":
    main()