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

class PSPModule(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels * (len(sizes) + 1), in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1))

    def _make_stage(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, x):
        h, w = (x.size(2), x.size(3))
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(x)
        out = torch.cat(priors, dim=1)
        out = self.bottleneck(out)
        return out

class PSPNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=3)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        self.layer5 = self._make_layer(512, 1024, blocks=3, stride=2)
        self.psp = PSPModule(1024)
        self.decoder = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_classes, kernel_size=1))
        self.aux_decoder = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, kernel_size=1))
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
        h, w = (x.size(2), x.size(3))
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux_out = x
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.psp(x)
        x = self.decoder(x)
        main_out = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            aux_out = self.aux_decoder(aux_out)
            aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True)
            return (main_out, aux_out)
        return main_out

def create_pspnet(num_classes=1):
    return PSPNet(in_channels=3, num_classes=num_classes)
