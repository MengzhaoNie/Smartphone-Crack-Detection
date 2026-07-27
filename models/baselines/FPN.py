import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class FPNBlock(nn.Module):

    def __init__(self, lateral_channels, out_channels):
        super().__init__()
        self.lateral_conv = nn.Conv2d(lateral_channels, out_channels, kernel_size=1, bias=False)
        self.fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y=None):
        if y is not None:
            lateral = self.lateral_conv(x)
            y = F.interpolate(y, size=lateral.shape[2:], mode='bilinear', align_corners=True)
            x = lateral + y
        else:
            x = self.lateral_conv(x)
        x = self.fpn_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FPN(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=3)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        self.layer5 = self._make_layer(512, 1024, blocks=3, stride=2)
        fpn_channels = 256
        self.fpn_channels = fpn_channels
        self.toplayer = FPNBlock(1024, fpn_channels)
        self.fpn1 = FPNBlock(512, fpn_channels)
        self.fpn2 = FPNBlock(256, fpn_channels)
        self.fpn3 = FPNBlock(128, fpn_channels)
        self.fpn4 = FPNBlock(64, fpn_channels)
        self.seg_head = nn.Sequential(nn.Conv2d(fpn_channels * 5, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(128, num_classes, kernel_size=1))
        self._init_weight()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.size()[2:]
        c1 = self.conv1(x)
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)
        p6 = self.toplayer(c6)
        p5 = self.fpn1(c5, p6)
        p4 = self.fpn2(c4, p5)
        p3 = self.fpn3(c3, p4)
        p2 = self.fpn4(c2, p3)
        p3 = F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=p2.shape[2:], mode='bilinear', align_corners=True)
        p5 = F.interpolate(p5, size=p2.shape[2:], mode='bilinear', align_corners=True)
        p6 = F.interpolate(p6, size=p2.shape[2:], mode='bilinear', align_corners=True)
        feature_pyramid = torch.cat([p2, p3, p4, p5, p6], dim=1)
        x = self.seg_head(feature_pyramid)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

def create_fpn(num_classes=1):
    return FPN(in_channels=3, num_classes=num_classes)
