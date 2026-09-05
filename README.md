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