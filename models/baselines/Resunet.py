import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
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
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResUNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc1 = ResBlock(64, 64)
        self.down1 = ResBlock(64, 128, stride=2)
        self.enc2 = ResBlock(128, 128)
        self.down2 = ResBlock(128, 256, stride=2)
        self.enc3 = ResBlock(256, 256)
        self.down3 = ResBlock(256, 512, stride=2)
        self.enc4 = ResBlock(512, 512)
        self.down4 = ResBlock(512, 1024, stride=2)
        self.bottleneck = ResBlock(1024, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResBlock(128, 64)
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.first(x)
        x = self.maxpool(x1)
        x2 = self.enc1(x)
        x = self.down1(x2)
        x3 = self.enc2(x)
        x = self.down2(x3)
        x4 = self.enc3(x)
        x = self.down3(x4)
        x5 = self.enc4(x)
        x = self.down4(x5)
        x = self.bottleneck(x)
        x = self.up4(x)
        x = torch.cat([x, x5], dim=1)
        x = self.dec4(x)
        x = self.up3(x)
        x = torch.cat([x, x4], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)
        x = self.final_up(x)
        x = self.final(x)
        return x

def create_resunet(num_classes=1):
    return ResUNet(in_channels=3, num_classes=num_classes)
