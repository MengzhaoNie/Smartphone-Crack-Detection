
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return x + self.conv(x)


class UNetGenerator(nn.Module):


    def __init__(self, in_ch=3, out_ch=3, base=32, n_res=4):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base, s=1)
        self.enc2 = ConvBlock(base, base * 2, s=2, p=1)
        self.enc3 = ConvBlock(base * 2, base * 4, s=2, p=1)
        self.res = nn.Sequential(*[ResidualBlock(base * 4) for _ in range(n_res)])
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.InstanceNorm2d(base, affine=True),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base, out_ch, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        r = self.res(e3)
        d2 = self.dec2(r)
        d1 = self.dec1(d2)
        return torch.tanh(self.out(d1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)
