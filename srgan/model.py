import numpy as np
import torch
from torch import nn


class DownSample(nn.Module):
    def __init__(self, in_channels=64):
        super(DownSample, self).__init__()
        out_channels = int(in_channels * 2)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=in_channels, padding=1, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels, out_channels=out_channels, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            DownSample(64),
            DownSample(128),
            DownSample(256),
        )

        self.tail = nn.Sequential(
            nn.Linear(512 * 6 ** 2, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.tail(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_channels)
        )

    def forward(self, x):
        x = self.net(x) + x
        return x


class Generator(nn.Module):
    def __init__(self, upsample_scale=4):
        super().__init__()
        self.upsample_scale = upsample_scale

        self.top = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4, stride=1),
            nn.PReLU()
        )

        self.residuals = nn.Sequential(*[ResidualBlock() for _ in range(5)])
        self.tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        x = self.top(x)

        z = self.residuals(x)
        z = self.tail(z)
        z = z + x
        return self.upsample(z)


if __name__ == '__main__':
    x = torch.Tensor(np.random.uniform(size=(5, 3, 64, 64)))
    dis = Discriminator()
    print(dis(x).shape)

    gen = Generator()
    print(gen(x).shape)
