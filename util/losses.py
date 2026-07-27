from __future__ import annotations
import torch
import torch.nn as nn

class IoUDiceLoss(nn.Module):

    def __init__(self, eps: float=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        p = pred.reshape(-1)
        t = target.reshape(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        iou = (inter + self.eps) / (union + self.eps)
        dice = (2 * inter + self.eps) / (p.sum() + t.sum() + self.eps)
        return 1.0 - iou + (1.0 - dice)
