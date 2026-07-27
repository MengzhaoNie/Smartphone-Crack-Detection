import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):

    def __init__(self, img_size=448, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class DownsamplePatchEmbedding(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

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
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, img_size, patch_size, in_channels, embed_dims, num_heads, depths, mlp_ratios, dropout=0.0):
        super().__init__()
        self.patch_embed = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.depths = depths
        print(f'Input image size: {img_size}x{img_size}')
        first_stage_size = img_size // patch_size
        print(f'Stage 1 feature map size after patch embedding: {first_stage_size}x{first_stage_size}')
        self.patch_embed.append(PatchEmbedding(img_size, patch_size, in_channels, embed_dims[0]))
        current_size = first_stage_size
        for i in range(1, len(depths)):
            current_size = current_size // 2
            print(f'Stage {i + 1} feature map size: {current_size}x{current_size}')
            self.patch_embed.append(DownsamplePatchEmbedding(embed_dims[i - 1], embed_dims[i]))
        for i in range(len(depths)):
            stage_blocks = nn.ModuleList()
            for _ in range(depths[i]):
                stage_blocks.append(TransformerBlock(embed_dims[i], num_heads[i], mlp_ratios[i], dropout))
            self.transformer_blocks.append(stage_blocks)
        self.proj_layers = nn.ModuleList()
        for i in range(len(depths)):
            self.proj_layers.append(nn.Linear(embed_dims[i], embed_dims[i]))

    def forward(self, x):
        features = []
        B = x.shape[0]
        x = self.patch_embed[0](x)
        for block in self.transformer_blocks[0]:
            x = block(x)
        h = w = int(math.sqrt(x.shape[1]))
        features.append(x.transpose(1, 2).reshape(B, -1, h, w))
        for i in range(1, len(self.depths)):
            h = w = int(math.sqrt(x.shape[1]))
            x_spatial = x.transpose(1, 2).reshape(B, -1, h, w)
            x = self.patch_embed[i](x_spatial)
            for block in self.transformer_blocks[i]:
                x = block(x)
            h = w = int(math.sqrt(x.shape[1]))
            features.append(x.transpose(1, 2).reshape(B, -1, h, w))
        return features

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class TransUNet(nn.Module):

    def __init__(self, img_size=448, patch_size=4, in_channels=3, num_classes=1, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], depths=[2, 2, 6, 2, 2], mlp_ratios=[4, 4, 4, 4, 4], dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(img_size, patch_size, in_channels, embed_dims, num_heads, depths, mlp_ratios, dropout)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(embed_dims) - 1, 0, -1):
            self.decoder_blocks.append(DecoderBlock(embed_dims[i], embed_dims[i - 1], embed_dims[i - 1]))
        self.final_conv = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
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
        features = self.encoder(x)
        x = features[-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[len(features) - 2 - i] if i < len(features) - 1 else None
            x = decoder_block(x, skip)
        x = self.final_conv(x)
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        return x

def create_transunet(img_size=448, patch_size=4, in_channels=3, num_classes=1):
    model = TransUNet(img_size=img_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], depths=[2, 2, 6, 2, 2], mlp_ratios=[4, 4, 4, 4, 4], dropout=0.1)
    return model
