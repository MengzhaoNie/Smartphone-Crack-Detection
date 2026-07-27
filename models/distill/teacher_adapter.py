
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherFeatureAdapter(nn.Module):


    def __init__(self, teacher: nn.Module, feat_dim: int = 64):
        super().__init__()
        self.teacher = teacher
        self.feat_dim = feat_dim
        self._buf: List[torch.Tensor] = []
        self._hook_modules = self._select_hook_modules(teacher)
        self._handles = []
        for m in self._hook_modules:
            self._handles.append(m.register_forward_hook(self._hook))

        self.projs = nn.ModuleList()
        self._proj_ready = False

    @staticmethod
    def _select_hook_modules(model: nn.Module) -> List[nn.Module]:

        if all(hasattr(model, n) for n in ("fusion0", "fusion2", "fusion4")):
            return [model.fusion0, model.fusion2, model.fusion4]

        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) < 3:
            raise RuntimeError("Teacher has fewer than 3 Conv2d layers for feature hooks")
        idxs = [max(0, len(convs) // 6), len(convs) // 2, max(0, len(convs) * 4 // 5)]
        picked = []
        for i in idxs:
            if convs[i] not in picked:
                picked.append(convs[i])
        while len(picked) < 3:
            picked.append(convs[-1])
        return picked[:3]

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            self._buf.append(output)

    def _ensure_projs(self, feats: List[torch.Tensor]):
        if self._proj_ready:
            return
        for f in feats:
            self.projs.append(
                nn.Sequential(
                    nn.Conv2d(f.shape[1], self.feat_dim, 1, bias=False),
                    nn.BatchNorm2d(self.feat_dim),
                    nn.ReLU(inplace=True),
                ).to(f.device)
            )
        self._proj_ready = True

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self._buf = []
        logits = self.teacher(x)
        feats_raw = list(self._buf)

        feats_raw = sorted(feats_raw, key=lambda t: t.shape[-1], reverse=True)
        if len(feats_raw) >= 3:

            feats_raw = [feats_raw[0], feats_raw[len(feats_raw) // 2], feats_raw[-1]]
        elif len(feats_raw) == 0:

            f = F.interpolate(torch.sigmoid(logits), scale_factor=0.25, mode="bilinear", align_corners=False)
            feats_raw = [f, f, f]
        while len(feats_raw) < 3:
            feats_raw.append(feats_raw[-1])

        self._ensure_projs(feats_raw)
        feats = [proj(f) for proj, f in zip(self.projs, feats_raw)]
        return logits, feats

    def forward_clip(self, frames: torch.Tensor) -> dict:
        b, t, _, h, w = frames.shape
        logits_list, key_feats = [], None
        for ti in range(t):
            logits, feats = self.forward_with_features(frames[:, ti])
            logits_list.append(logits)
            if ti == t // 2:
                key_feats = feats
        frame_logits = torch.stack(logits_list, dim=1)
        return {
            "key_logits": frame_logits[:, t // 2],
            "frame_logits": frame_logits,
            "key_feats": key_feats,
        }

    def forward(self, x):
        return self.teacher(x)

    def train(self, mode: bool = True):

        super().train(mode)
        self.teacher.eval()
        return self


def build_teacher_adapter(teacher: nn.Module, feat_dim: int = 64) -> TeacherFeatureAdapter:
    adapter = TeacherFeatureAdapter(teacher, feat_dim=feat_dim)
    adapter.eval()
    for p in adapter.teacher.parameters():
        p.requires_grad = False
    return adapter
