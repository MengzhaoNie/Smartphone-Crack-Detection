import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.cnn_kan_backbone import UKAN
from ..modules.Vmamba import vmamba_base_s2l15

class DualEncoderConcatNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.vssm = vmamba_base_s2l15()
        self.ukan = UKAN(num_classes=num_classes)


        feature_dims = [32, 64, 128, 256, 512]


        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(feature_dims[4], feature_dims[3], 2, 2),
            nn.ConvTranspose2d(feature_dims[3], feature_dims[2], 2, 2),
            nn.ConvTranspose2d(feature_dims[2], feature_dims[1], 2, 2),
            nn.ConvTranspose2d(feature_dims[1], feature_dims[0], 2, 2),
            nn.ConvTranspose2d(feature_dims[0], num_classes, 8, 8)
        ])


        self.reduce_channels4 = nn.Conv2d(feature_dims[4]*2, feature_dims[4], 1)
        self.reduce_channels3 = nn.Conv2d(feature_dims[3]*3, feature_dims[3], 1)
        self.reduce_channels2 = nn.Conv2d(feature_dims[2]*3, feature_dims[2], 1)
        self.reduce_channels1 = nn.Conv2d(feature_dims[1]*3, feature_dims[1], 1)
        self.reduce_channels0 = nn.Conv2d(feature_dims[0]*2, feature_dims[0], 1)

    def forward(self, x):

        v1, v2, v3, v4 = self.vssm.get_features(x)
        u0, u1, u2, u3, u4 = self.ukan.get_features(x)


        cat4 = torch.cat([v4, u4], dim=1)
        cat4 = self.reduce_channels4(cat4)
        d4 = cat4
        d4_up = self.decoder[0](d4)


        d4_cat = torch.cat([d4_up, v3, u3], dim=1)
        d4_cat = self.reduce_channels3(d4_cat)
        d3 = d4_cat
        d3_up = self.decoder[1](d3)


        d3_cat = torch.cat([d3_up, v2, u2], dim=1)
        d3_cat = self.reduce_channels2(d3_cat)
        d2 = d3_cat
        d2_up = self.decoder[2](d2)


        d2_cat = torch.cat([d2_up, v1, u1], dim=1)
        d2_cat = self.reduce_channels1(d2_cat)
        d1 = d2_cat
        d1_up = self.decoder[3](d1)


        d1_cat = torch.cat([d1_up, u0], dim=1)
        d1_cat = self.reduce_channels0(d1_cat)
        d0 = d1_cat
        out = self.decoder[4](d0)

        return out

def create_dual_encoder_concat(num_classes=1):
    return DualEncoderConcatNet(num_classes=num_classes)
