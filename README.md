# Statistical Evaluation of Regularisation Strategies for Image Classification and Semantic Segmentation

## Overview

This project investigates how commonly used regularisation and data-augmentation strategies affect model performance in two different computer vision tasks:

1. **Image classification** on CIFAR-10
2. **Semantic segmentation** on Oxford-IIIT Pet

The primary objective is to examine whether strategies that are useful for image classification provide similar benefits when transferred to dense pixel-level prediction.

The study evaluates both **clean test performance** and behaviour under simple **distribution shifts**, together with **predictive calibration** for classification.

The evaluated methods are:

### Classification
- Baseline
- MixUp
- CutMix
- CutOut
- Random Erasing

### Semantic Segmentation
- Baseline
- CutMix
- CutOut
- ClassMix

All reported experiments use **seed 0** and should therefore be interpreted as a **controlled single-seed pilot study**, rather than a statistically conclusive multi-seed comparison.

---

## Research Questions

The study is organized around the following questions:

1. Do regularisation strategies improve clean test performance in image classification?
2. Do the same strategies improve semantic segmentation performance?
3. Do the effects observed in classification transfer consistently to dense prediction?
4. How do the methods differ in predictive calibration?
5. Do regularisation strategies provide improved robustness under simple distribution shifts?

---

## Experimental Setup

### Image Classification

**Dataset:** CIFAR-10

**Model:** CNN baseline architecture shared across all classification conditions.

**Training conditions:**
- Baseline
- MixUp
- CutMix
- CutOut
- Random Erasing

The same model architecture and general training protocol were used across conditions, with the regularisation strategy being the primary experimental difference.

---

### Semantic Segmentation

**Dataset:** Oxford-IIIT Pet

**Model:** U-Net

**Training conditions:**
- Baseline
- CutMix
- CutOut
- ClassMix

ClassMix was adapted to the binary foreground/background setting of Oxford-IIIT Pet by transferring foreground regions between samples. This should not be interpreted as an evaluation of conventional multiclass ClassMix in its original setting.

---

## Evaluation Metrics

### Classification

- Test Loss
- Accuracy
- Expected Calibration Error (ECE)

### Semantic Segmentation

- Dice coefficient
- Intersection over Union (IoU)

### Robustness

For both tasks, deterministic test-time perturbations were applied to the input images:

- **Brightness:** factor = `0.6`
- **Gaussian Blur:** radius = `1.0`
- **Posterization:** `4` bits per channel

No retraining or test-time adaptation was performed.

For segmentation, the ground-truth masks were left unchanged while the corresponding input images were perturbed.

---

# Results

All experiments use the same model architecture and training protocol within each task. Reported results are from a single held-out test evaluation using **seed 0**.

## Image Classification — CIFAR-10

| Training Condition | Test Loss | Test Accuracy |
|---|---:|---:|
| Baseline | 0.3722 | **87.97%** |
| MixUp | 0.4739 | 87.58% |
| CutMix | 0.4659 | 86.44% |
| CutOut | 0.3815 | 87.63% |
| Random Erasing | **0.3672** | **88.11%** |

Random Erasing achieved the highest test accuracy at **88.11%**, only slightly above the baseline at **87.97%**.

The remaining methods did not exceed the baseline:
- MixUp: 87.58%
- CutMix: 86.44%
- CutOut: 87.63%

This indicates that, in this single-seed experiment, regularisation did not produce a consistent improvement in clean classification accuracy.

> **Note:** The test loss values for MixUp and CutMix should not be interpreted as directly comparable to ordinary training loss because these methods use mixed targets during training.

---

## Semantic Segmentation — Oxford-IIIT Pet

| Training Condition | Test Dice | Test IoU |
|---|---:|---:|
| Baseline | **0.825488** | **0.720302** |
| CutMix | 0.820811 | 0.714113 |
| CutOut | 0.820613 | 0.715376 |
| ClassMix | 0.800127 | 0.687817 |

The baseline achieved the strongest clean segmentation performance on both metrics.

CutMix and CutOut remained relatively close to the baseline, while ClassMix produced a larger performance reduction.

Thus, none of the evaluated segmentation regularisation strategies exceeded the baseline on clean test performance in this pilot experiment.

> **Note:** ClassMix was adapted to the binary foreground/background setting by transferring foreground regions between images. The result should therefore be interpreted specifically for this binary adaptation.

---

# Calibration Analysis

Calibration was evaluated for the CIFAR-10 classification models using **Expected Calibration Error (ECE)**.

ECE measures the difference between model confidence and empirical accuracy across confidence bins. Lower values indicate better calibration.

| Method | ECE ↓ |
|---|---:|
| Baseline | 0.037810 |
| MixUp | 0.128526 |
| CutMix | 0.094471 |
| CutOut | 0.032049 |
| Random Erasing | **0.029922** |

Random Erasing produced the lowest ECE, followed closely by CutOut.

The baseline had an ECE of `0.037810`, while MixUp and CutMix showed substantially larger calibration errors.

This provides an additional perspective beyond accuracy: a method can have competitive classification accuracy without necessarily producing well-calibrated confidence estimates.

In this experiment:

- **Random Erasing** achieved both the best clean accuracy and the best ECE.
- **CutOut** slightly improved calibration relative to the baseline despite having slightly lower accuracy.
- **MixUp** and **CutMix** showed considerably poorer calibration under this evaluation.

The calibration analysis uses the true CIFAR-10 test labels and the model's predicted probability distributions without temperature scaling or other post-hoc calibration.

## Robustness Analysis

Robustness was evaluated under deterministic test-time image perturbations to examine how the regularisation strategies behave under simple distribution shifts. Perturbations were applied to the input images only; no retraining or test-time adaptation was performed.

Three fixed perturbations were used:

- **Brightness:** factor = `0.6`
- **Gaussian Blur:** radius = `1.0`
- **Posterization:** `4` bits per channel

All robustness results reported below correspond to the **seed-0 pilot experiments**.

### Classification Robustness

For CIFAR-10 classification, robustness is evaluated using clean test accuracy and accuracy degradation under each perturbation.

| Method | Clean Accuracy | Brightness Accuracy | Brightness Degradation | Blur Accuracy | Blur Degradation | Posterize Accuracy | Posterize Degradation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 87.97% | 84.82% | 3.15 pp | 35.37% | 52.60 pp | **85.92%** | **2.05 pp** |
| MixUp | 87.58% | 85.50% | **2.08 pp** | **41.89%** | **45.69 pp** | 85.35% | 2.23 pp |
| CutMix | 86.44% | 83.73% | 2.71 pp | 29.50% | 56.94 pp | 83.54% | 2.90 pp |
| CutOut | 87.63% | 84.08% | 3.55 pp | 30.51% | 57.12 pp | 85.35% | 2.28 pp |
| Random Erasing | **88.11%** | **85.52%** | 2.59 pp | 34.43% | 53.68 pp | 85.56% | 2.55 pp |

Several observations emerge:

- **MixUp** achieved the highest shifted accuracy under brightness among the evaluated conditions and had the smallest accuracy degradation under brightness.
- **MixUp** also performed best under Gaussian blur, achieving 41.89% accuracy and the smallest blur degradation.
- The **baseline** achieved the highest absolute accuracy under posterization at 85.92%.
- **CutMix** and **CutOut** showed substantial degradation under Gaussian blur.
- **Random Erasing** retained the highest clean accuracy but did not consistently dominate under distribution shift.

Overall, no single classification method consistently provided the strongest robustness across all perturbations. The effect of regularisation therefore appears to be **perturbation-dependent** rather than universally beneficial.

### Semantic Segmentation Robustness — Dice

For Oxford-IIIT Pet semantic segmentation, robustness was evaluated using Dice and IoU. Degradation represents the absolute decrease from the corresponding clean-test metric.

| Method | Clean Dice | Brightness Dice | Δ Dice | Blur Dice | Δ Dice | Posterize Dice | Δ Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | **0.825488** | **0.799836** | 0.025652 | **0.813625** | 0.011863 | **0.818992** | 0.006496 |
| CutMix | 0.820811 | 0.776899 | 0.043912 | 0.813342 | **0.007469** | 0.804074 | 0.016737 |
| CutOut | 0.820613 | 0.794312 | 0.026301 | 0.810221 | 0.010392 | 0.814009 | 0.006605 |
| ClassMix | 0.800127 | 0.777320 | **0.022807** | 0.792424 | 0.007704 | 0.796778 | **0.003349** |

The baseline achieved the highest **absolute Dice** under all three perturbations.

However, degradation values should be interpreted together with clean and shifted performance. For example, ClassMix shows a smaller Dice degradation under brightness and posterization, but its clean Dice is already substantially lower than the other methods. Therefore, a smaller drop does not necessarily imply stronger absolute robustness.

CutMix provides another example of perturbation-dependent behaviour: it experienced the largest degradation under brightness but remained highly competitive under blur.

### Semantic Segmentation Robustness — IoU

| Method | Clean IoU | Brightness IoU | Δ IoU | Blur IoU | Δ IoU | Posterize IoU | Δ IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | **0.720302** | **0.684612** | 0.035690 | **0.705611** | 0.014690 | **0.710649** | 0.009653 |
| CutMix | 0.714113 | 0.656779 | 0.057334 | 0.704368 | **0.009745** | 0.691843 | 0.022270 |
| CutOut | 0.715376 | 0.678717 | 0.036659 | 0.702259 | 0.013116 | 0.705626 | 0.009750 |
| ClassMix | 0.687817 | 0.657174 | **0.030642** | 0.679989 | 0.007827 | 0.682161 | **0.005656** |

The IoU results show the same overall pattern:

- The baseline retained the highest absolute IoU under all three perturbations.
- CutMix was particularly sensitive to brightness and posterization.
- CutMix remained relatively competitive under blur.
- ClassMix showed smaller absolute degradation for some perturbations, but its shifted IoU remained lower because of its lower clean performance.
- No segmentation regularisation strategy consistently outperformed the baseline across the tested distribution shifts.

### Robustness Interpretation

Across both classification and segmentation, the experiments do **not** show a universal robustness advantage from regularisation.

Instead, robustness depends on both the **training strategy** and the **type of distribution shift**. A method may perform well under one perturbation while performing poorly under another.

For this reason, robustness should be assessed using both:

1. **Absolute shifted performance**, which indicates how well the model performs under the altered input distribution.
2. **Degradation from clean performance**, which indicates how much performance is lost relative to the model's own clean baseline.

Because the current study uses a **single seed (seed 0)**, these robustness results should be treated as descriptive pilot observations rather than statistically validated evidence of general robustness.

A reliability diagram is generated at:

```text
experiments/analysis/calibration/reliability_diagram.png


