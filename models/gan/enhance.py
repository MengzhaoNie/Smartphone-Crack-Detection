
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import PatchDiscriminator, UNetGenerator


class CrackEnhanceGAN(nn.Module):


  def __init__(self, in_ch=3, out_ch=3):
    super().__init__()
    self.generator = UNetGenerator(in_ch, out_ch, base=32, n_res=4)
    self.discriminator = PatchDiscriminator(in_ch, base=32)

  def generate(self, x: torch.Tensor) -> torch.Tensor:

    out = self.generator(x * 2 - 1)
    return (out + 1) * 0.5

  def forward(self, x):
    return self.generate(x)


def create_enhance_gan():
  return CrackEnhanceGAN()
