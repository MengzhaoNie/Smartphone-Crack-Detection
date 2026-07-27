
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
  def __init__(self, ch, growth=16, n_layers=4):
    super().__init__()
    self.layers = nn.ModuleList()
    for i in range(n_layers):
      self.layers.append(nn.Sequential(
        nn.Conv2d(ch + i * growth, growth, 3, 1, 1, bias=True),
        nn.ReLU(inplace=True),
      ))
    self.out = nn.Conv2d(ch + n_layers * growth, ch, 1, 1, 0)

  def forward(self, x):
    feats = [x]
    for layer in self.layers:
      feats.append(layer(torch.cat(feats, dim=1)))
    return self.out(torch.cat(feats, dim=1)) + x


class RDB(nn.Module):
  def __init__(self, ch, growth=16, n_dense=3):
    super().__init__()
    self.blocks = nn.Sequential(*[DenseBlock(ch, growth) for _ in range(n_dense)])

  def forward(self, x):
    return self.blocks(x) + x


class CrackSRNet(nn.Module):


  def __init__(self, in_ch=3, out_ch=3, nf=32, n_rdb=4, scale=2):
    super().__init__()
    self.scale = scale
    self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
    self.rdbs = nn.Sequential(*[RDB(nf) for _ in range(n_rdb)])
    self.body = nn.Conv2d(nf, nf, 3, 1, 1)
    self.up = nn.Sequential(
      nn.Conv2d(nf, nf * scale * scale, 3, 1, 1),
      nn.PixelShuffle(scale),
      nn.ReLU(inplace=True),
    )
    self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)

  def forward(self, x):
    feat = self.head(x)
    body = self.body(self.rdbs(feat)) + feat
    out = self.up(body)
    out = self.tail(out)
    if out.shape[-2:] != (x.shape[-2] * self.scale, x.shape[-1] * self.scale):
      out = F.interpolate(out, scale_factor=self.scale, mode="bilinear", align_corners=False)
    return torch.clamp(out, 0.0, 1.0)

  def super_resolve(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
    sr = self.forward(x)
    if target_size is not None and sr.shape[-2:] != target_size:
      sr = F.interpolate(sr, size=target_size, mode="bilinear", align_corners=False)
    return sr


def create_srnet(scale=2):
  return CrackSRNet(scale=scale)
