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

A reliability diagram is generated at:

```text
experiments/analysis/calibration/reliability_diagram.png
