
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoUDiceLoss(nn.Module):


    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)
        inter = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - inter
        iou = (inter + self.smooth) / (union + self.smooth)
        dice = (2 * inter + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return ((1 - iou) + (1 - dice)).mean()


class ResponseKDLoss(nn.Module):


    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        if student_logits.shape[-2:] != teacher_logits.shape[-2:]:
            student_logits = F.interpolate(
                student_logits, size=teacher_logits.shape[-2:], mode="bilinear", align_corners=False
            )
        s = torch.sigmoid(student_logits / self.temperature)
        t = torch.sigmoid(teacher_logits / self.temperature)
        return F.mse_loss(s, t)


class FeatureKDLoss(nn.Module):


    def __init__(self):
        super().__init__()
        self.projs = nn.ModuleDict()

    def _align(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if s.shape[1] != t.shape[1]:
            key = f"{s.shape[1]}_{t.shape[1]}"
            if key not in self.projs:
                self.projs[key] = nn.Conv2d(s.shape[1], t.shape[1], 1, bias=False).to(
                    device=s.device, dtype=s.dtype
                )
            s = self.projs[key](s)
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
        return s

    def forward(self, student_feats: Sequence[torch.Tensor], teacher_feats: Sequence[torch.Tensor]):
        assert len(student_feats) == len(teacher_feats)
        loss = 0.0
        for s, t in zip(student_feats, teacher_feats):
            s = self._align(s, t.detach())
            loss = loss + F.mse_loss(s, t.detach())
        return loss / max(len(student_feats), 1)


class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.response = ResponseKDLoss()

    def forward(self, student_frame_logits: torch.Tensor, teacher_frame_logits: torch.Tensor):
        t = student_frame_logits.shape[1]
        loss = 0.0
        for i in range(t):
            loss = loss + self.response(student_frame_logits[:, i], teacher_frame_logits[:, i])
        return loss / t


class DistillLossBundle(nn.Module):


    def __init__(
        self,
        mode: str = "full",
        w_task: float = 1.0,
        w_feat: float = 0.5,
        w_resp: float = 0.5,
        w_temp: float = 0.5,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.w_task = w_task
        self.w_feat = w_feat
        self.w_resp = w_resp
        self.w_temp = w_temp
        self.task = SoftIoUDiceLoss()
        self.feat = FeatureKDLoss()
        self.resp = ResponseKDLoss()
        self.temp = TemporalConsistencyLoss()

    def forward(self, student_out: dict, teacher_out: dict, key_mask: torch.Tensor):
        logits = student_out["logits"]
        if logits.shape[-2:] != key_mask.shape[-2:]:
            logits = F.interpolate(logits, size=key_mask.shape[-2:], mode="bilinear", align_corners=False)

        loss_task = self.task(logits, key_mask)
        parts = {"task": loss_task}
        total = self.w_task * loss_task

        if self.mode in ("kd", "full"):
            loss_feat = self.feat(student_out["key_feats"], teacher_out["key_feats"])
            loss_resp = self.resp(student_out["key_logits"], teacher_out["key_logits"])
            parts["feat"] = loss_feat
            parts["resp"] = loss_resp
            total = total + self.w_feat * loss_feat + self.w_resp * loss_resp

        if self.mode == "full":
            loss_temp = self.temp(student_out["frame_logits"], teacher_out["frame_logits"])
            if "temp_logits" in student_out:
                loss_temp = loss_temp + self.resp(student_out["temp_logits"], teacher_out["key_logits"])
            parts["temp"] = loss_temp
            total = total + self.w_temp * loss_temp

        parts["total"] = total
        return total, parts
