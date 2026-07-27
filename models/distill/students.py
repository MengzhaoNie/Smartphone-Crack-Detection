
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b0,
    mobilenet_v2,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
)


class DWConvAlign(nn.Module):


    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MobileViTBlock(nn.Module):
    def __init__(self, channels: int, dim: int = 96, patch_size: int = 2, depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.patch = patch_size
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.local(x)
        p = self.patch

        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h))
        _, d, hp, wp = y.shape
        gh, gw = hp // p, wp // p

        tokens = y.view(b, d, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1).contiguous()
        tokens = tokens.view(b, gh * gw, p * p, d).mean(dim=2)
        tokens = self.transformer(tokens)

        tokens = tokens.view(b, gh, gw, d).permute(0, 3, 1, 2)
        tokens = tokens.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        tokens = tokens[:, :, :h, :w]
        return x + self.proj(tokens)


class MobileViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            MobileViTBlock(64, dim=96, depth=2),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            MobileViTBlock(96, dim=120, depth=2),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            MobileViTBlock(128, dim=144, depth=2),
        )

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        return [f1, f2, f3, f4]


class MobileNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        net = mobilenet_v2(weights="DEFAULT" if pretrained else None)
        feats = net.features
        self.stem = feats[:2]
        self.stage1 = feats[2:4]
        self.stage2 = feats[4:7]
        self.stage3 = feats[7:14]
        self.stage4 = feats[14:]

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class ShuffleNetEncoder(nn.Module):
    def __init__(self, width: str = "0.5", pretrained: bool = False):
        super().__init__()
        ctor = shufflenet_v2_x0_5 if width == "0.5" else shufflenet_v2_x1_0
        net = ctor(weights="DEFAULT" if pretrained else None)
        self.conv1 = net.conv1
        self.maxpool = net.maxpool
        self.stage2 = net.stage2
        self.stage3 = net.stage3
        self.stage4 = net.stage4
        self.conv5 = net.conv5

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv1(x)
        f1 = self.maxpool(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.conv5(self.stage4(f3))
        return [f1, f2, f3, f4]


class EfficientNetEncoder(nn.Module):


    def __init__(self, pretrained: bool = False):
        super().__init__()
        net = efficientnet_b0(weights="DEFAULT" if pretrained else None)
        feats = net.features


        self.stem = feats[0]
        self.stage1 = feats[1:3]
        self.stage2 = feats[3:4]
        self.stage3 = feats[4:6]
        self.stage4 = feats[6:]

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FastSCNNEncoder(nn.Module):


    def __init__(self):
        super().__init__()

        self.conv1 = _ConvBNReLU(3, 32, 3, 2, 1)
        self.ds1 = _DSConv(32, 48, stride=2)
        self.ds2 = _DSConv(48, 64, stride=2)

        self.ge1 = nn.Sequential(_DSConv(64, 64), _DSConv(64, 64))
        self.ge2 = nn.Sequential(_DSConv(64, 96, stride=2), _DSConv(96, 96))
        self.ge3 = nn.Sequential(_DSConv(96, 128, stride=2), _DSConv(128, 128))

        self.ppm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, 1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x) -> List[torch.Tensor]:
        f1 = self.conv1(x)
        f1 = self.ds1(f1)
        f2 = self.ds2(f1)
        f2 = self.ge1(f2)
        f3 = self.ge2(f2)
        f4 = self.ge3(f3)
        ppm = self.ppm(f4)
        f4 = f4 + F.interpolate(ppm, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        return [f1, f2, f3, f4]


STUDENT_BACKBONES = (
    "efficientnet",
    "mobilenet",
    "shufflenet",
    "shufflenet_v2",
    "fastscnn",
    "fast_scnn",
    "mobilevit",
)


class StudentSegNet(nn.Module):


    def __init__(self, backbone: str = "mobilevit", num_classes: int = 1, feat_dim: int = 64, pretrained: bool = False):
        super().__init__()
        backbone = backbone.lower().replace("-", "_")
        self.backbone_name = backbone
        if backbone == "mobilevit":
            self.encoder = MobileViTEncoder()
            chs = [32, 64, 96, 128]
        elif backbone == "mobilenet":
            self.encoder = MobileNetEncoder(pretrained=pretrained)

            chs = [24, 32, 96, 1280]
        elif backbone in ("shufflenet", "shufflenet_v2"):
            self.encoder = ShuffleNetEncoder(width="0.5", pretrained=pretrained)
            chs = [24, 48, 96, 1024]
        elif backbone in ("efficientnet", "efficientnet_b0"):
            self.encoder = EfficientNetEncoder(pretrained=pretrained)
            chs = [24, 40, 112, 1280]
        elif backbone in ("fastscnn", "fast_scnn"):
            self.encoder = FastSCNNEncoder()
            chs = [48, 64, 96, 128]
        else:
            raise ValueError(
                f"Unknown student backbone: {backbone}. Choose from {STUDENT_BACKBONES}"
            )

        self.dec3 = DecoderBlock(chs[3], chs[2], 128)
        self.dec2 = DecoderBlock(128, chs[1], 64)
        self.dec1 = DecoderBlock(64, chs[0], 32)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
        )


        self.align_low = DWConvAlign(chs[0], feat_dim)
        self.align_mid = DWConvAlign(chs[1], feat_dim)
        self.align_high = DWConvAlign(chs[2], feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x, return_features: bool = False):
        original = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)
        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        logits = self.final_up(x)
        if logits.shape[-2:] != original:
            logits = F.interpolate(logits, size=original, mode="bilinear", align_corners=False)

        if not return_features:
            return logits

        feats = [
            self.align_low(f1),
            self.align_mid(f2),
            self.align_high(f3),
        ]
        return logits, feats


def create_student(backbone: str = "mobilevit", num_classes: int = 1, feat_dim: int = 64, pretrained: bool = False):
    return StudentSegNet(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim, pretrained=pretrained)
