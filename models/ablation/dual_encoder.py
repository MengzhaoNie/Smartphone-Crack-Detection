from __future__ import annotations
from typing import List, Sequence
import torch
import torch.nn as nn
from ..modules.cnn_kan_backbone import UKAN, KANBlock
from ..modules.Vmamba import vmamba_base_s2l15

class KANFuseModule(nn.Module):

    def __init__(self, in_channels: Sequence[int], out_channels: int, **kan_base_params):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        fused_dim = sum(self.in_channels)
        self.lns = nn.ModuleList([nn.LayerNorm(c) for c in self.in_channels])
        self.kan = KANBlock(dim=fused_dim, num_heads=1, sr_ratio=1, **kan_base_params)
        self.proj = nn.Linear(fused_dim, out_channels) if fused_dim != out_channels else nn.Identity()

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == len(self.lns)
        b, _, h, w = feats[0].shape
        tokens = []
        for feat, ln in zip(feats, self.lns):
            if feat.shape[-2:] != (h, w):
                feat = nn.functional.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            t = feat.flatten(2).transpose(1, 2).contiguous()
            tokens.append(ln(t))
        z = torch.cat(tokens, dim=-1)
        z = self.kan(z, h, w)
        z = self.proj(z)
        return z.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

class DualEncoderNet(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        self.vssm = vmamba_base_s2l15()
        self.ukan = UKAN(num_classes=num_classes)
        feature_dims = [32, 64, 128, 256, 512]
        kan_base_params = {'mlp_ratio': 1, 'qkv_bias': False, 'qk_scale': None, 'drop': 0.0, 'attn_drop': 0.0, 'drop_path': 0.0, 'norm_layer': nn.LayerNorm}
        self.fusion4 = KANFuseModule([512, 512], 512, **kan_base_params)
        self.fusion3 = KANFuseModule([256, 256, 256], 256, **kan_base_params)
        self.fusion2 = KANFuseModule([128, 128, 128], 128, **kan_base_params)
        self.fusion1 = KANFuseModule([64, 64, 64], 64, **kan_base_params)
        self.fusion0 = KANFuseModule([32, 32], 32, **kan_base_params)
        self.decoder = nn.ModuleList([nn.ConvTranspose2d(feature_dims[4], feature_dims[3], 2, 2), nn.ConvTranspose2d(feature_dims[3], feature_dims[2], 2, 2), nn.ConvTranspose2d(feature_dims[2], feature_dims[1], 2, 2), nn.ConvTranspose2d(feature_dims[1], feature_dims[0], 2, 2), nn.ConvTranspose2d(feature_dims[0], num_classes, 2, 2)])

    def forward(self, x):
        v1, v2, v3, v4 = self.vssm.get_features(x)
        u0, u1, u2, u3, u4 = self.ukan.get_features(x)
        f4 = self.fusion4([u4, v4])
        d4_up = self.decoder[0](f4)
        f3 = self.fusion3([d4_up, v3, u3])
        d3_up = self.decoder[1](f3)
        f2 = self.fusion2([d3_up, v2, u2])
        d2_up = self.decoder[2](f2)
        f1 = self.fusion1([d2_up, v1, u1])
        d1_up = self.decoder[3](f1)
        f0 = self.fusion0([d1_up, u0])
        out = self.decoder[4](f0)
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

def create_dual_encoder(num_classes=1):
    return DualEncoderNet(num_classes=num_classes)
