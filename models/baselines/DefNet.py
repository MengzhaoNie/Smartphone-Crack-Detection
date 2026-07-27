import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):

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

class ResNetEncoder(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ConvBlock(32, 32)
        self.layer2 = ConvBlock(32, 64, stride=2)
        self.layer3 = ConvBlock(64, 128, stride=2)
        self.layer4 = ConvBlock(128, 256, stride=2)
        self.layer5 = ConvBlock(256, 512, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features = []
        x1 = self.layer1(x)
        features.append(x1)
        x2 = self.layer2(x1)
        features.append(x2)
        x3 = self.layer3(x2)
        features.append(x3)
        x4 = self.layer4(x3)
        features.append(x4)
        x5 = self.layer5(x4)
        features.append(x5)
        return features

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x):
        x_ln = self.norm1(x)
        attn_output, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attn_output
        x_ln = self.norm2(x)
        mlp_output = self.mlp(x_ln)
        x = x + mlp_output
        return x

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x

class DownsampleLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class TransformerEncoder(nn.Module):

    def __init__(self, img_size=448, in_channels=3, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], depths=[2, 2, 2, 2], dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size=4, in_channels=in_channels, embed_dim=embed_dims[0])
        self.stages = nn.ModuleList()
        for i in range(len(embed_dims)):
            stage = nn.ModuleList()
            for j in range(depths[i]):
                stage.append(TransformerBlock(embed_dims[i], num_heads[i], dropout=dropout))
            if i < len(embed_dims) - 1:
                stage.append(DownsampleLayer(embed_dims[i], embed_dims[i + 1]))
            self.stages.append(stage)

    def forward(self, x):
        x = self.patch_embed(x)
        features = [None, None, None, None]
        features[0] = x.clone()
        for i, stage in enumerate(self.stages):
            B, C, H, W = x.shape
            x_seq = x.flatten(2).transpose(1, 2)
            for j, block in enumerate(stage):
                if isinstance(block, TransformerBlock):
                    x_seq = block(x_seq)
                else:
                    x = x_seq.transpose(1, 2).reshape(B, C, H, W)
                    x = block(x)
                    if i < len(self.stages) - 1 or j < len(stage) - 1:
                        B, C, H, W = x.shape
                        x_seq = x.flatten(2).transpose(1, 2)
            if i == len(self.stages) - 1:
                x = x_seq.transpose(1, 2).reshape(B, C, H, W)
            else:
                x = x_seq.transpose(1, 2).reshape(B, C, H, W)
            if i < len(features) - 1:
                features[i + 1] = x.clone()
        return features

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_resnet_channels, skip_transformer_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.adjust_transformer = nn.Sequential(nn.Conv2d(skip_transformer_channels, skip_resnet_channels, kernel_size=1), nn.BatchNorm2d(skip_resnet_channels), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(nn.Conv2d(in_channels + skip_resnet_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip_resnet, skip_transformer):
        x = self.upsample(x)
        if skip_resnet.shape[2:] != skip_transformer.shape[2:]:
            skip_transformer = F.interpolate(skip_transformer, size=skip_resnet.shape[2:], mode='bilinear', align_corners=False)
        skip_transformer = self.adjust_transformer(skip_transformer)
        skip = skip_resnet + skip_transformer
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class DefNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, img_size=448, dropout=0.1):
        super().__init__()
        self.resnet_encoder = ResNetEncoder(in_channels)
        self.transformer_encoder = TransformerEncoder(img_size=img_size, in_channels=in_channels, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], depths=[2, 2, 2, 2], dropout=dropout)
        self.adjust_resnet_bottleneck = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.bottleneck_conv = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.decoder_blocks = nn.ModuleList([DecoderBlock(512, 256, 256, 256), DecoderBlock(256, 128, 128, 128), DecoderBlock(128, 64, 64, 64), DecoderBlock(64, 32, 64, 64)])
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        original_size = x.shape[2:]
        resnet_features = self.resnet_encoder(x)
        transformer_features = self.transformer_encoder(x)
        deepest_resnet = resnet_features[4]
        deepest_transformer = transformer_features[3]
        if deepest_resnet.shape[2:] != deepest_transformer.shape[2:]:
            deepest_transformer = F.interpolate(deepest_transformer, size=deepest_resnet.shape[2:], mode='bilinear', align_corners=False)
        adjusted_resnet = self.adjust_resnet_bottleneck(deepest_resnet)
        bottleneck = adjusted_resnet + deepest_transformer
        bottleneck = self.bottleneck_conv(bottleneck)
        x = bottleneck
        x = self.decoder_blocks[0](x, resnet_features[3], transformer_features[2])
        x = self.decoder_blocks[1](x, resnet_features[2], transformer_features[1])
        x = self.decoder_blocks[2](x, resnet_features[1], transformer_features[0])
        x = self.decoder_blocks[3](x, resnet_features[0], transformer_features[0])
        x = self.final_upsample(x)
        x = self.final_conv(x)
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x

def create_defnet(in_channels=3, num_classes=1, img_size=448):
    model = DefNet(in_channels=in_channels, num_classes=num_classes, img_size=img_size, dropout=0.1)
    return model
