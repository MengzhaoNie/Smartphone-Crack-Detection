
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTM
from .students import create_student


class StudentKDModel(nn.Module):


    def __init__(
        self,
        backbone: str = "mobilevit",
        num_classes: int = 1,
        feat_dim: int = 64,
        use_convlstm: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()
        self.use_convlstm = use_convlstm
        self.student = create_student(
            backbone, num_classes=num_classes, feat_dim=feat_dim, pretrained=pretrained
        )
        self.feat_dim = feat_dim
        if use_convlstm:
            self.temporal = ConvLSTM(in_channels=feat_dim, hidden_channels=feat_dim, num_layers=1)
            self.temp_head = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_dim, num_classes, 1),
            )
        else:
            self.temporal = None
            self.temp_head = None

    def encode_frame(self, x: torch.Tensor):
        return self.student(x, return_features=True)

    def forward(self, x: torch.Tensor):

        return self.student(x)

    def forward_clip(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:

        b, t, c, h, w = frames.shape
        logits_list: List[torch.Tensor] = []
        mid_list: List[torch.Tensor] = []
        feat_bundle: List[List[torch.Tensor]] = []

        for ti in range(t):
            logits, feats = self.encode_frame(frames[:, ti])
            logits_list.append(logits)
            mid_list.append(feats[1])
            feat_bundle.append(feats)

        frame_logits = torch.stack(logits_list, dim=1)
        key_idx = t // 2
        key_logits = frame_logits[:, key_idx]
        key_feats = feat_bundle[key_idx]

        out = {
            "key_logits": key_logits,
            "frame_logits": frame_logits,
            "key_feats": key_feats,
            "frame_feats_mid": torch.stack(mid_list, dim=1),
        }

        if self.use_convlstm:
            temporal_out, _ = self.temporal(out["frame_feats_mid"])
            temp_feat = temporal_out[:, key_idx]
            temp_logits = self.temp_head(temp_feat)
            if temp_logits.shape[-2:] != (h, w):
                temp_logits = F.interpolate(
                    temp_logits, size=(h, w), mode="bilinear", align_corners=False
                )
            out["temp_logits"] = temp_logits
            out["temp_feats"] = temporal_out
            out["logits"] = 0.5 * key_logits + 0.5 * temp_logits
        else:
            out["logits"] = key_logits

        return out


def build_student_kd(backbone: str = "mobilevit", use_convlstm: bool = True, **kwargs) -> StudentKDModel:
    return StudentKDModel(backbone=backbone, use_convlstm=use_convlstm, **kwargs)
