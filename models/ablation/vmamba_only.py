import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.Vmamba import vmamba_base_s2l15

class VMambaOnlyNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.vssm = vmamba_base_s2l15()


        feature_dims = [64, 128, 256, 512]


        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(feature_dims[3], feature_dims[2], 2, 2),
            nn.ConvTranspose2d(feature_dims[2], feature_dims[1], 2, 2),
            nn.ConvTranspose2d(feature_dims[1], feature_dims[0], 2, 2),
            nn.ConvTranspose2d(feature_dims[0], 32, 2, 2),
            nn.ConvTranspose2d(32, num_classes, 4, 4)
        ])


        self.skip_conv3 = nn.Conv2d(feature_dims[2], feature_dims[2], 1)
        self.skip_conv2 = nn.Conv2d(feature_dims[1], feature_dims[1], 1)
        self.skip_conv1 = nn.Conv2d(feature_dims[0], feature_dims[0], 1)


        self.fusion3 = nn.Conv2d(feature_dims[2]*2, feature_dims[2], 1)
        self.fusion2 = nn.Conv2d(feature_dims[1]*2, feature_dims[1], 1)
        self.fusion1 = nn.Conv2d(feature_dims[0]*2, feature_dims[0], 1)

    def forward(self, x):

        v1, v2, v3, v4 = self.vssm.get_features(x)


        d4 = v4
        d4_up = self.decoder[0](d4)


        skip3 = self.skip_conv3(v3)
        d3_cat = torch.cat([d4_up, skip3], dim=1)
        d3 = self.fusion3(d3_cat)
        d3_up = self.decoder[1](d3)


        skip2 = self.skip_conv2(v2)
        d2_cat = torch.cat([d3_up, skip2], dim=1)
        d2 = self.fusion2(d2_cat)
        d2_up = self.decoder[2](d2)


        skip1 = self.skip_conv1(v1)
        d1_cat = torch.cat([d2_up, skip1], dim=1)
        d1 = self.fusion1(d1_cat)
        d1_up = self.decoder[3](d1)


        out = self.decoder[4](d1_up)

        return out

def create_vmamba_only(num_classes=1):
    return VMambaOnlyNet(num_classes=num_classes)
