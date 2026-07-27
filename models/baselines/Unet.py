import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.pool1(conv1)
        conv2 = self.conv2(x)
        x = self.pool2(conv2)
        conv3 = self.conv3(x)
        x = self.pool3(conv3)
        conv4 = self.conv4(x)
        x = self.pool4(conv4)
        x = self.bottleneck(x)
        x = self.upconv4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv4(x)
        x = self.upconv3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv3(x)
        x = self.upconv2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv2(x)
        x = self.upconv1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv1(x)
        x = self.out_conv(x)
        return x

def create_unet(num_classes=1):
    return UNet(in_channels=3, num_classes=num_classes)
