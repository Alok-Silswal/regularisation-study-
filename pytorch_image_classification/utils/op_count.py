from typing import Tuple

import torch
import torch.nn as nn
import yacs.config


def count_op(config: yacs.config.CfgNode, model: nn.Module) -> Tuple[str, str]:
    try:
        import thop
    except (ImportError, AttributeError):
        return "N/A", "N/A"

    data = torch.zeros(
        (
            1,
            config.dataset.n_channels,
            config.dataset.image_size,
            config.dataset.image_size,
        ),
        dtype=torch.float32,
        device=torch.device(config.device),
    )

    return thop.clever_format(
        thop.profile(model, (data,), verbose=False)
    )