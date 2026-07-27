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

class ASPPModule(nn.Module):

    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super().__init__()
        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.project(x)
        return x

class DeepLabV3Plus(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, output_stride=16):
        super().__init__()
        if output_stride == 16:
            dilations = [1, 1, 1, 2, 4]
            strides = [1, 2, 2, 1, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4, 8]
            strides = [1, 2, 1, 1, 1]
        else:
            raise ValueError('output_stride must be 8 or 16!')
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=strides[3], dilation=dilations[3])
        self.layer5 = self._make_layer(512, 1024, blocks=3, stride=strides[4], dilation=dilations[4])
        self.aspp = ASPPModule(1024, 256)
        self.low_level_conv = nn.Sequential(nn.Conv2d(64, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dilation=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride=stride, dilation=dilation))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels, dilation=dilation))
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
        x = self.conv1(x)
        low_level_feat = x
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

def create_deeplabv3plus(num_classes=1, output_stride=16):
    return DeepLabV3Plus(in_channels=3, num_classes=num_classes, output_stride=output_stride)
