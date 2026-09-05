# Statistical Evaluation of Regularisation Strategies for Image Classification and Semantic Segmentation

## Overview

This project investigates how commonly used regularisation and data-augmentation strategies affect performance across two computer vision tasks:

1. **Image classification** on CIFAR-10
2. **Semantic segmentation** on Oxford-IIIT Pet

The central motivation is to examine whether regularisation strategies that are commonly effective for image classification provide similar benefits when transferred to **dense pixel-level prediction**.

The study evaluates each training strategy from multiple perspectives:

- Clean test performance
- Predictive calibration
- Robustness under controlled distribution shifts

The classification and segmentation experiments use a common comparative framework, while the model architecture and task-specific evaluation metrics remain appropriate to each task.

> **Experimental status:** The reported results are **single-seed pilot results using seed 0**. They are intended for controlled comparison and hypothesis generation, not as evidence of statistically significant or universally generalisable differences.

---

## Research Questions

The project investigates the following questions:

1. Do regularisation strategies improve clean image-classification performance?
2. Do the same strategies improve semantic-segmentation performance?
3. Do classification-level regularisation benefits transfer consistently to dense prediction?
4. How do the evaluated methods differ in predictive calibration?
5. Do regularisation strategies improve robustness under simple distribution shifts?
6. Are the effects of regularisation consistent across different perturbation types?

---

# Experimental Tasks

## Image Classification

### Dataset

**CIFAR-10**

CIFAR-10 contains 10 image classes and is used to evaluate the effect of regularisation on image-level classification.

### Model

A CNN-based classification model is used across all classification conditions.

### Training Conditions

The following training conditions were evaluated:

- Baseline
- MixUp
- CutMix
- CutOut
- Random Erasing

The same general model architecture and training protocol were maintained across conditions so that the regularisation strategy remained the primary experimental difference.

---

## Semantic Segmentation

### Dataset

**Oxford-IIIT Pet**

The segmentation task uses the Oxford-IIIT Pet dataset to evaluate whether classification-oriented regularisation strategies transfer to pixel-level prediction.

### Model

A U-Net architecture is used for semantic segmentation.

### Training Conditions

The following conditions were evaluated:

- Baseline
- CutMix
- CutOut
- ClassMix

ClassMix was adapted to the binary foreground/background segmentation setting by transferring foreground regions between samples.

> **Important:** The ClassMix result therefore represents a **binary foreground-based adaptation**, not a direct evaluation of conventional multiclass ClassMix.

---

# Regularisation Strategies

## MixUp

MixUp constructs training samples by taking convex combinations of images and their corresponding targets.

## CutMix

CutMix replaces a region of one training image with a region from another image and mixes the corresponding target information.

## CutOut

CutOut randomly masks a region of the input image during training.

## Random Erasing

Random Erasing removes a randomly selected rectangular region of the training image and replaces it with modified pixel content.

## ClassMix

ClassMix transfers regions associated with semantic classes between images. In this project, it was adapted to the binary foreground/background segmentation setting of Oxford-IIIT Pet.

---

# Experimental Setup

All experiments were conducted using the same overall comparative methodology within each task.

The main controlled factors were:

- Dataset
- Model architecture
- Training configuration
- Regularisation strategy
- Random seed

The current study uses:

```text
Seed = 0
