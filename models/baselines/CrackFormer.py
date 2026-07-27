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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.head_dim ** (-0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, depth, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResNetEncoder(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        return features

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class CrackFormer(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, embed_dim=1024, num_heads=8, transformer_depth=2, dropout=0.1):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels)
        self.downsample = nn.Sequential(ConvBlock(512, 1024, stride=2), ConvBlock(1024, 1024))
        self.transformer = TransformerEncoder(embed_dim=1024, num_heads=num_heads, depth=transformer_depth, dropout=dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(1024, 512, 512), DecoderBlock(512, 256, 256), DecoderBlock(256, 128, 128), DecoderBlock(128, 64, 64)])
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
        encoder_features = self.encoder(x)
        deepest_feature = self.downsample(encoder_features[-1])
        B, C, H, W = deepest_feature.shape
        feature_seq = deepest_feature.flatten(2).transpose(1, 2)
        feature_seq = self.transformer(feature_seq)
        transformer_feature = feature_seq.transpose(1, 2).reshape(B, C, H, W)
        x = transformer_feature
        x = self.decoder_blocks[0](x, encoder_features[3])
        x = self.decoder_blocks[1](x, encoder_features[2])
        x = self.decoder_blocks[2](x, encoder_features[1])
        x = self.decoder_blocks[3](x, encoder_features[0])
        x = self.final_upsample(x)
        x = self.final_conv(x)
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x

def create_crackformer(in_channels=3, num_classes=1):
    model = CrackFormer(in_channels=in_channels, num_classes=num_classes, embed_dim=1024, num_heads=8, transformer_depth=2, dropout=0.1)
    return model
