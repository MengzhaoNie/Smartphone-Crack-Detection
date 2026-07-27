import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.cnn_kan_backbone import UKAN

class UKANOnlyNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.ukan = UKAN(num_classes=num_classes)


        feature_dims = [32, 64, 128, 256, 512]


        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(feature_dims[4], feature_dims[3], 2, 2),
            nn.ConvTranspose2d(feature_dims[3], feature_dims[2], 2, 2),
            nn.ConvTranspose2d(feature_dims[2], feature_dims[1], 2, 2),
            nn.ConvTranspose2d(feature_dims[1], feature_dims[0], 2, 2),
            nn.ConvTranspose2d(feature_dims[0], num_classes, 8, 8)
        ])


        self.skip_conv3 = nn.Conv2d(feature_dims[3], feature_dims[3], 1)
        self.skip_conv2 = nn.Conv2d(feature_dims[2], feature_dims[2], 1)
        self.skip_conv1 = nn.Conv2d(feature_dims[1], feature_dims[1], 1)
        self.skip_conv0 = nn.Conv2d(feature_dims[0], feature_dims[0], 1)


        self.fusion3 = nn.Conv2d(feature_dims[3]*2, feature_dims[3], 1)
        self.fusion2 = nn.Conv2d(feature_dims[2]*2, feature_dims[2], 1)
        self.fusion1 = nn.Conv2d(feature_dims[1]*2, feature_dims[1], 1)
        self.fusion0 = nn.Conv2d(feature_dims[0]*2, feature_dims[0], 1)

    def forward(self, x):

        u0, u1, u2, u3, u4 = self.ukan.get_features(x)


        d4 = u4
        d4_up = self.decoder[0](d4)


        skip3 = self.skip_conv3(u3)
        d3_cat = torch.cat([d4_up, skip3], dim=1)
        d3 = self.fusion3(d3_cat)
        d3_up = self.decoder[1](d3)


        skip2 = self.skip_conv2(u2)
        d2_cat = torch.cat([d3_up, skip2], dim=1)
        d2 = self.fusion2(d2_cat)
        d2_up = self.decoder[2](d2)


        skip1 = self.skip_conv1(u1)
        d1_cat = torch.cat([d2_up, skip1], dim=1)
        d1 = self.fusion1(d1_cat)
        d1_up = self.decoder[3](d1)


        skip0 = self.skip_conv0(u0)
        d0_cat = torch.cat([d1_up, skip0], dim=1)
        d0 = self.fusion0(d0_cat)
        out = self.decoder[4](d0)

        return out

def create_ukan_only(num_classes=1):
    return UKANOnlyNet(num_classes=num_classes)
