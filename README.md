## Results

All experiments use the same model architecture and training protocol within
each task. Reported results are from a single held-out test evaluation using
seed 0.

### Image Classification — CIFAR-10

| Training Condition | Test Loss | Test Accuracy |
|---|---:|---:|
| Baseline | 0.3722 | **87.97%** |
| MixUp | 0.4739 | 87.58% |
| CutMix | 0.4659 | 86.44% |
| CutOut | 0.3815 | 87.63% |
| Random Erasing | **0.3672** | **88.11%** |

Random Erasing achieved the highest test accuracy (88.11%), with the baseline
close behind at 87.97%. MixUp, CutMix, and CutOut did not improve test accuracy
over the baseline in this single-seed experiment.

> **Note:** Loss values for MixUp and CutMix should not be interpreted as
> directly comparable to ordinary training loss because these methods train
> with mixed targets.

### Semantic Segmentation — Oxford-IIIT Pet

| Training Condition | Test Dice | Test IoU |
|---|---:|---:|
| Baseline | **0.825488** | **0.720302** |
| CutMix | 0.820811 | 0.714113 |
| CutOut | 0.820613 | 0.715376 |
| ClassMix | 0.800127 | 0.687817 |

The baseline achieved the strongest segmentation performance. CutMix and CutOut
were competitive but did not surpass the baseline, while ClassMix showed a
larger performance reduction.

> **Note:** ClassMix was adapted to the binary foreground/background setting
> of Oxford-IIIT Pet by transferring foreground regions between samples.
> Results are from a single seed and should therefore be interpreted as a
> controlled pilot comparison rather than evidence of statistically
> significant differences.

## Robustness Analysis

Robustness was evaluated under deterministic test-time image perturbations to examine how the regularisation strategies behave under simple distribution shifts. The perturbations were applied to the input images only; segmentation masks were left unchanged.

Three fixed perturbations were evaluated:

- **Brightness:** brightness factor = 0.6
- **Gaussian Blur:** radius = 1.0
- **Posterization:** 4 bits per channel

All robustness results reported below correspond to the **seed-0 pilot experiments**.

### Classification Robustness

For CIFAR-10 classification, robustness is evaluated using clean test accuracy and accuracy degradation under each perturbation.

| Method | Clean Accuracy | Brightness Accuracy | Brightness Degradation | Blur Accuracy | Blur Degradation | Posterize Accuracy | Posterize Degradation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 87.97% | — | — | — | — | — | — |
| MixUp | 87.58% | — | — | — | — | — | — |
| CutMix | 86.44% | — | — | — | — | — | — |
| CutOut | 87.63% | — | — | — | — | — | — |
| Random Erasing | 88.11% | — | — | — | — | — | — |

> Classification robustness results will be populated from the corresponding robustness analysis output.

### Semantic Segmentation Robustness

For Oxford-IIIT Pet semantic segmentation, robustness is evaluated using Dice and IoU under the same three deterministic image perturbations. Degradation is reported as the absolute decrease from the corresponding clean-test metric.

| Method | Clean Dice | Brightness Dice | Δ Dice | Blur Dice | Δ Dice | Posterize Dice | Δ Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.825488 | 0.799836 | 0.025652 | 0.813625 | 0.011863 | 0.818992 | 0.006496 |
| CutMix | 0.820811 | 0.776899 | 0.043912 | 0.813342 | 0.007469 | 0.804074 | 0.016737 |
| CutOut | 0.820613 | 0.794312 | 0.026301 | 0.810221 | 0.010392 | 0.814009 | 0.006605 |
| ClassMix | 0.800127 | 0.777320 | 0.022807 | 0.792424 | 0.007704 | 0.796778 | 0.003349 |

#### IoU Under Distribution Shift

| Method | Clean IoU | Brightness IoU | Δ IoU | Blur IoU | Δ IoU | Posterize IoU | Δ IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.720302 | 0.684612 | 0.035690 | 0.705611 | 0.014690 | 0.710649 | 0.009653 |
| CutMix | 0.714113 | 0.656779 | 0.057334 | 0.704368 | 0.009745 | 0.691843 | 0.022270 |
| CutOut | 0.715376 | 0.678717 | 0.036659 | 0.702259 | 0.013116 | 0.705626 | 0.009750 |
| ClassMix | 0.687817 | 0.657174 | 0.030642 | 0.679989 | 0.007827 | 0.682161 | 0.005656 |

### Segmentation Robustness Observations

The robustness results show that the effect of regularisation is perturbation-dependent.

- **Brightness shift:** CutMix exhibits the largest degradation in both Dice (0.043912) and IoU (0.057334), while ClassMix has the smallest absolute degradation.
- **Gaussian blur:** CutMix shows relatively small degradation in both Dice and IoU, despite its larger degradation under brightness and posterization.
- **Posterization:** ClassMix shows the smallest degradation in both Dice (0.003349) and IoU (0.005656), whereas CutMix experiences a larger degradation.
- The baseline achieves the highest clean segmentation performance, while the regularised models do not consistently outperform it under distribution shift.

These results therefore do **not** support a general claim that classification-style regularisation universally improves robustness in semantic segmentation. Instead, robustness appears to depend on both the regularisation strategy and the type of distribution shift.

### Robustness Protocol

The robustness analysis uses deterministic, fixed-severity perturbations and does not involve retraining, test-time adaptation, or additional random seeds. The reported results should therefore be interpreted as a **seed-0 pilot robustness analysis**, rather than as a statistical robustness comparison across repeated experiments.