from __future__ import annotations
import math
from dataclasses import dataclass
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer('grid', grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]).clamp_min(1e-06) * bases[:, :, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]).clamp_min(1e-06) * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)

class KAN(nn.Module):

    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList([KANLinear(in_f, out_f, grid_size=grid_size, spline_order=spline_order) for in_f, out_f in zip(layers_hidden, layers_hidden[1:])])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class DSConv(nn.Module):

    def __init__(self, in_channels: int=1, out_channels: int=1, kernel_size: int=9, extend_scope: float=1.0, morph: int=0, if_offset: bool=True):
        super().__init__()
        if morph not in (0, 1):
            raise ValueError('morph should be 0 or 1.')
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(max(out_channels // 4, 1), out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)
        self.dsc_conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(kernel_size, 1))
        self.dsc_conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, kernel_size))

    def forward(self, input: torch.Tensor):
        offset = self.tanh(self.gn_offset(self.offset_conv(input)))
        y_map, x_map = get_coordinate_map_2D(offset, self.morph, self.extend_scope, device=input.device)
        deformed = get_interpolated_feature(input, y_map, x_map)
        output = self.dsc_conv_x(deformed) if self.morph == 0 else self.dsc_conv_y(deformed)
        return self.relu(self.gn(output))

def get_coordinate_map_2D(offset, morph, extend_scope=1.0, device='cuda'):
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)
    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)
    y_center_ = einops.repeat(torch.arange(0, width, dtype=torch.float32, device=device), 'w -> k w h', k=kernel_size, h=height)
    x_center_ = einops.repeat(torch.arange(0, height, dtype=torch.float32, device=device), 'h -> k w h', k=kernel_size, w=width)
    if morph == 0:
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        y_grid_ = einops.repeat(y_spread_, 'k -> k w h', w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, 'k -> k w h', w=width, h=height)
        y_new_ = einops.repeat(y_center_ + y_grid_, 'k w h -> b k w h', b=batch_size)
        x_new_ = einops.repeat(x_center_ + x_grid_, 'k w h -> b k w h', b=batch_size)
        y_offset_ = einops.rearrange(y_offset_, 'b k w h -> k b w h')
        y_offset_new_ = y_offset_.detach().clone()
        y_offset_new_[center] = 0
        for index in range(1, center + 1):
            y_offset_new_[center + index] = y_offset_new_[center + index - 1] + y_offset_[center + index]
            y_offset_new_[center - index] = y_offset_new_[center - index + 1] + y_offset_[center - index]
        y_offset_new_ = einops.rearrange(y_offset_new_, 'k b w h -> b k w h')
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))
        y_coordinate_map = einops.rearrange(y_new_, 'b k w h -> b (w k) h')
        x_coordinate_map = einops.rearrange(x_new_, 'b k w h -> b (w k) h')
    else:
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)
        y_grid_ = einops.repeat(y_spread_, 'k -> k w h', w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, 'k -> k w h', w=width, h=height)
        y_new_ = einops.repeat(y_center_ + y_grid_, 'k w h -> b k w h', b=batch_size)
        x_new_ = einops.repeat(x_center_ + x_grid_, 'k w h -> b k w h', b=batch_size)
        x_offset_ = einops.rearrange(x_offset_, 'b k w h -> k b w h')
        x_offset_new_ = x_offset_.detach().clone()
        x_offset_new_[center] = 0
        for index in range(1, center + 1):
            x_offset_new_[center + index] = x_offset_new_[center + index - 1] + x_offset_[center + index]
            x_offset_new_[center - index] = x_offset_new_[center - index + 1] + x_offset_[center - index]
        x_offset_new_ = einops.rearrange(x_offset_new_, 'k b w h -> b k w h')
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))
        y_coordinate_map = einops.rearrange(y_new_, 'b k w h -> b w (h k)')
        x_coordinate_map = einops.rearrange(x_new_, 'b k w h -> b w (h k)')
    return (y_coordinate_map, x_coordinate_map)

def get_interpolated_feature(input_feature, y_coordinate_map, x_coordinate_map, interpolate_mode='bilinear'):
    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1
    y_map = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max]).unsqueeze(-1)
    x_map = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max]).unsqueeze(-1)
    grid = torch.cat([x_map, y_map], dim=-1)
    return F.grid_sample(input_feature, grid, mode=interpolate_mode, padding_mode='zeros', align_corners=True)

def _coordinate_map_scaling(coordinate_map, origin, target=(-1, 1)):
    min_v, max_v = origin
    a, b = target
    coordinate_map = torch.clamp(coordinate_map, min_v, max_v)
    scale = (b - a) / (max_v - min_v + 1e-06)
    return a + scale * (coordinate_map - min_v)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=False), nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class ChannelWiseSelfAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels, bias=False)
        self.key = nn.Linear(in_channels, in_channels, bias=False)
        self.value = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = in_channels ** (-0.5)

    def forward(self, x):
        b, c, _, _ = x.shape
        xc = F.adaptive_avg_pool2d(x, 1).flatten(1)
        q = self.query(xc)
        k = self.key(xc)
        v = self.value(xc)
        attn = torch.softmax(q.unsqueeze(2) @ k.unsqueeze(1) * self.scale, dim=-1)
        x_w = (attn @ v.unsqueeze(-1)).squeeze(-1)
        return x * x_w.view(b, c, 1, 1)

class SKPNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, bilinear=False):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.adapter_1 = nn.Conv2d(512, 16, 1)
        self.adapter_2 = nn.Conv2d(256, 16, 1)
        self.adapter_3 = nn.Conv2d(128, 32, 1)
        self.channelwiseattention = ChannelWiseSelfAttention(128)
        self.enc = nn.Sequential(nn.Conv2d(128, 32, 1, 1), nn.ReLU(inplace=False), DSConv(32, 32, 3, morph=0), DSConv(32, 32, 3, morph=1), nn.Conv2d(32, 32, 3, 1, 1))
        self.outc = KAN(layers_hidden=[32, 16, num_classes], grid_size=3)

    def forward(self, x):
        b, _, h, w = x.shape
        th, tw = (h // 4, w // 4)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        f1 = F.interpolate(self.adapter_1(x), size=(th, tw), mode='bilinear', align_corners=False)
        x = self.up2(x, x3)
        f2 = F.interpolate(self.adapter_2(x), size=(th, tw), mode='bilinear', align_corners=False)
        x = self.up3(x, x2)
        f3 = F.interpolate(self.adapter_3(x), size=(th, tw), mode='bilinear', align_corners=False)
        x = self.up4(x, x1)
        f4 = F.interpolate(x, size=(th, tw), mode='bilinear', align_corners=False)
        x = torch.cat([f1, f2, f3, f4], dim=1)
        x = self.channelwiseattention(x)
        x = self.enc(x)
        x = x.flatten(2).permute(0, 2, 1)
        logits = self.outc(x)
        logits = logits.permute(0, 2, 1).reshape(b, -1, th, tw)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)

def create_skpnet(num_classes=1, in_channels=3):
    return SKPNet(in_channels=in_channels, num_classes=num_classes)
if __name__ == '__main__':
    model = create_skpnet(1)
    y = model(torch.randn(1, 3, 256, 256))
    print(y.shape, sum((p.numel() for p in model.parameters())) / 1000000.0)
