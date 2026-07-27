
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import PatchDiscriminator, UNetGenerator


class CrackCycleGAN(nn.Module):
  def __init__(self, in_ch=3, out_ch=3):
    super().__init__()
    self.G_ab = UNetGenerator(in_ch, out_ch, base=32, n_res=3)
    self.G_ba = UNetGenerator(in_ch, out_ch, base=32, n_res=3)
    self.D_a = PatchDiscriminator(in_ch, base=32)
    self.D_b = PatchDiscriminator(in_ch, base=32)

  def to_target_style(self, x: torch.Tensor) -> torch.Tensor:
    out = self.G_ab(x * 2 - 1)
    return (out + 1) * 0.5

  def to_source_style(self, x: torch.Tensor) -> torch.Tensor:
    out = self.G_ba(x * 2 - 1)
    return (out + 1) * 0.5

  def forward(self, x, direction="ab"):
    if direction == "ab":
      return self.to_target_style(x)
    return self.to_source_style(x)


def create_cyclegan():
  return CrackCycleGAN()
