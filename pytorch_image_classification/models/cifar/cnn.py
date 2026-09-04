import torch
import torch.nn as nn

from ..initializer import create_initializer


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.cnn
        channels = model_config.channels

        self.features = nn.Sequential(
            nn.Conv2d(
                config.dataset.n_channels,
                channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels[0],
                channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels[1],
                channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                channels[1],
                channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels[2],
                channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(channels[2], config.dataset.n_classes)

        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x