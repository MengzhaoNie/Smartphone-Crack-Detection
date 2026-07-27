from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
    _HAS_CUDA_SCAN = True
except Exception:
    _cuda_scan = None
    _HAS_CUDA_SCAN = False

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

def selective_scan_ref(u, delta, A, B, C, D):
    d_a = torch.einsum('bld,dn->bldn', delta, A)
    d_b_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
    d_a_pad = F.pad(d_a[:, 1:], (0, 0, 0, 0, 1, 1))[:, 1:, :, :]
    d_a_cs = torch.flip(d_a_pad, dims=[1]).cumsum(dim=1).exp()
    d_a_cs = torch.flip(d_a_cs, dims=[1])
    x = d_b_u * d_a_cs
    x = x.cumsum(dim=1) / (d_a_cs + 1e-12)
    y = torch.einsum('bldn,bln->bld', x, C)
    return y + u * D

def selective_scan(u, delta, A, B, C, D):
    if _HAS_CUDA_SCAN and u.is_cuda:
        y = _cuda_scan(u.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), A.contiguous(), B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(), D.contiguous(), None, None, False, False)
        return y.transpose(1, 2)
    return selective_scan_ref(u, delta, A, B, C, D)

class SSM(nn.Module):

    def __init__(self, d_model, expand=2, d_state=16):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x32 = x.float()
            x_dbl = self.x_proj(x32)
            delta, b, c = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(delta)).float().clamp(min=0.0001, max=20.0)
            b = b.float().clamp(-10.0, 10.0)
            c = c.float().clamp(-10.0, 10.0)
            a = -torch.exp(self.A_log.float().clamp(max=2.0))
            y = selective_scan(x32, delta, a, b, c, self.D.float())
            y = torch.nan_to_num(y, nan=0.0, posinf=10000.0, neginf=-10000.0)
        return y.to(dtype=x.dtype)

class MambaBlock(nn.Module):

    def __init__(self, d_model, expand=2, d_conv=4, d_state=16, conv_bias=True):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv // 2, bias=conv_bias)
        self.ssm = SSM(d_model, expand=expand, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        x, res = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        if x.size(1) != res.size(1):
            x = x[:, :res.size(1)]
        return self.out_proj(self.ssm(F.silu(x)) * F.silu(res))

class ResDMamba(nn.Module):

    def __init__(self, d_model, expand=2, d_conv=4, d_state=16):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = MambaBlock(d_model, expand=expand, d_conv=d_conv, d_state=d_state)
        self.dw3 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.dw5 = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model)
        self.out = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        path3 = F.silu(self.dw5(x.transpose(1, 2)).transpose(1, 2))
        n = self.norm(x)
        path2 = F.silu(self.dw3(n.transpose(1, 2)).transpose(1, 2))
        y = (self.mamba(n) + x + path3 + path2).transpose(1, 2)
        return self.out(y).transpose(1, 2)

def map_to_tokens(feat):
    return feat.flatten(2).transpose(1, 2)

def tokens_to_map(tokens, h, w):
    b, _, c = tokens.shape
    return tokens.transpose(1, 2).contiguous().view(b, c, h, w)

def resize_tokens(tokens, h, w, th, tw):
    if h == th and w == tw:
        return tokens
    return map_to_tokens(F.interpolate(tokens_to_map(tokens, h, w), size=(th, tw), mode='bilinear', align_corners=False))

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=64, img_size=448):
        super().__init__()
        self.p = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        g = img_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, g * g, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.p
        assert h % p == 0 and w % p == 0
        x = x.unfold(2, p, p).unfold(3, p, p)
        nh, nw = (x.shape[2], x.shape[3])
        x = x.contiguous().view(b, c, nh * nw, p * p).permute(0, 2, 1, 3).contiguous()
        x = x.view(b, nh * nw, c * p * p)
        x = self.proj(x)
        if x.size(1) == self.pos_embed.size(1):
            x = x + self.pos_embed
        else:
            g0 = int(round(math.sqrt(self.pos_embed.size(1))))
            pe = self.pos_embed.transpose(1, 2).reshape(1, -1, g0, g0)
            pe = F.interpolate(pe, size=(nh, nw), mode='bilinear', align_corners=False)
            x = x + pe.flatten(2).transpose(1, 2)
        return (x, nh, nw)

class FeaturePatchEmbed(nn.Module):

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.p = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x, th, tw):
        p = self.p
        x = F.interpolate(x, size=(th * p, tw * p), mode='bilinear', align_corners=False)
        b, c, _, _ = x.shape
        x = x.unfold(2, p, p).unfold(3, p, p)
        nh, nw = (x.shape[2], x.shape[3])
        x = x.contiguous().view(b, c, nh * nw, p * p).permute(0, 2, 1, 3).contiguous()
        x = x.view(b, nh * nw, c * p * p)
        return self.proj(x)

class PatchMerging(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        pad_h, pad_w = (h % 2, w % 2)
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            h, w = (h + pad_h, w + pad_w)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1).view(b, -1, 4 * c)
        return (self.reduction(x), h // 2, w // 2)

class PatchExpand(nn.Module):

    def __init__(self, dim, upsample_rate=2):
        super().__init__()
        self.r = upsample_rate
        self.proj = nn.Conv2d(dim, upsample_rate * dim, 1, bias=False)

    def forward(self, x, h, w):
        feat = self.proj(tokens_to_map(x, h, w))
        feat = F.pixel_shuffle(feat, self.r)
        nh, nw = (h * self.r, w * self.r)
        return (map_to_tokens(feat), nh, nw)

class TokensToImage(nn.Module):

    def __init__(self, dim, patch_size=4):
        super().__init__()
        self.r = patch_size
        self.proj = nn.Conv2d(dim, patch_size * dim, 1, bias=False)

    def forward(self, x, h, w):
        feat = self.proj(tokens_to_map(x, h, w))
        return F.pixel_shuffle(feat, self.r)

class ConvResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, down=False, up=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.fuse = nn.Conv2d(out_ch + in_ch, out_ch, 3, padding=1)
        self.down = nn.MaxPool2d(2) if down else None
        self.up = nn.Upsample(scale_factor=2, mode='nearest') if up else None

    def forward(self, x):
        y = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = self.fuse(torch.cat([y, x], dim=1))
        if self.down is not None:
            y = self.down(y)
        if self.up is not None:
            y = self.up(y)
        return y

class MambaCrackNet(nn.Module):

    def __init__(self, img_size=448, patch_size=4, in_channels=3, num_classes=1, base_dim=64, expand=2, d_state=16, depth=4):
        super().__init__()
        self.patch_size = patch_size
        self.depth = depth
        n_down = depth - 1
        dims = [base_dim * 2 ** i for i in range(depth)]
        mamba_kw = dict(expand=expand, d_state=d_state)
        self.patch_embed = PatchEmbed(patch_size, in_channels, dims[0], img_size)
        self.enc0 = ResDMamba(dims[0], d_conv=4, **mamba_kw)
        self.merges = nn.ModuleList([PatchMerging(dims[i]) for i in range(n_down)])
        self.enc_mamba = nn.ModuleList([ResDMamba(dims[i + 1], d_conv=2, **mamba_kw) for i in range(n_down)])
        self.cnn_stem = ConvResBlock(in_channels, 32, down=False, up=False)
        cnn_out_chs = [32] + [64 * (2 * i + 1) for i in range(n_down)]
        self.cnn_downs = nn.ModuleList([ConvResBlock(cnn_out_chs[i], cnn_out_chs[i + 1], down=True, up=False) for i in range(n_down)])
        self.cnn_to_tok = nn.ModuleList([FeaturePatchEmbed(patch_size, cnn_out_chs[i + 1], dims[i + 1]) for i in range(n_down)])
        self.fuse = nn.ModuleList([nn.Linear(2 * dims[i + 1], dims[i + 1], bias=False) for i in range(n_down)])
        self.expands = nn.ModuleList([PatchExpand(dims[depth - 1 - i], 2) for i in range(n_down)])
        self.dec_fuse = nn.ModuleList([nn.Linear(2 * dims[depth - 2 - i], dims[depth - 2 - i], bias=False) for i in range(n_down)])
        self.dec_mamba = nn.ModuleList([ResDMamba(dims[depth - 2 - i], d_conv=2, **mamba_kw) for i in range(n_down)])
        self.dec_proj = nn.ModuleList([nn.Linear(dims[depth - 2 - i], dims[depth - 2 - i], bias=False) for i in range(n_down)])
        pyr_out_chs = [64 * (2 * (n_down - i)) for i in range(n_down)]
        cnn_up_in, prev = ([], cnn_out_chs[-1])
        for i in range(n_down):
            skip_ch = cnn_out_chs[n_down - i]
            cnn_up_in.append(prev + skip_ch)
            prev = pyr_out_chs[i]
        self.cnn_ups = nn.ModuleList([ConvResBlock(cnn_up_in[i], dims[depth - 2 - i], down=False, up=True) for i in range(n_down)])
        self.to_image = nn.ModuleList([TokensToImage(dims[depth - 2 - i], patch_size) for i in range(n_down)])
        self.pyr_fuse = nn.ModuleList([ConvResBlock(dims[depth - 2 - i] + dims[depth - 2 - i] // patch_size, pyr_out_chs[i], down=False, up=False) for i in range(n_down)])
        self.pyr_conv = nn.ModuleList([nn.Conv2d(pyr_out_chs[i], 64, 3, padding=1) for i in range(n_down)])
        self.pyr_proj = nn.ModuleList([nn.Conv2d(64, 32, 1) for _ in range(n_down)])
        self.head = nn.Sequential(nn.Conv2d(32 * n_down, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, num_classes, 1))
        self.img_size = img_size
        self.n_down = n_down

    def forward(self, x):
        _, _, h0, w0 = x.shape
        tokens, gh, gw = self.patch_embed(x)
        tokens = self.enc0(tokens)
        skips_t = [(tokens, gh, gw)]
        img = self.cnn_stem(x)
        skips_c = [img]
        for i in range(self.n_down):
            tokens, gh, gw = self.merges[i](tokens, gh, gw)
            img = self.cnn_downs[i](img)
            skips_c.append(img)
            cnn_tok = self.cnn_to_tok[i](img, gh, gw)
            tokens = self.fuse[i](torch.cat([self.enc_mamba[i](tokens), cnn_tok], dim=-1))
            skips_t.append((tokens, gh, gw))
        skips_t_rev = list(reversed(skips_t))
        skips_c_rev = list(reversed(skips_c))
        tokens, gh, gw = skips_t_rev[0]
        img = skips_c_rev[0]
        decode_t = skips_t_rev[1:]
        outs = []
        for i in range(self.n_down):
            tokens, gh, gw = self.expands[i](tokens, gh, gw)
            sk, sh, sw = decode_t[i]
            if (gh, gw) != (sh, sw):
                tokens = resize_tokens(tokens, gh, gw, sh, sw)
                gh, gw = (sh, sw)
            skip_c = skips_c_rev[i]
            if img.shape[2:] != skip_c.shape[2:]:
                img = F.interpolate(img, size=skip_c.shape[2:], mode='bilinear', align_corners=False)
            img = self.cnn_ups[i](torch.cat([img, skip_c], dim=1))
            tokens = self.dec_fuse[i](torch.cat([tokens, sk], dim=-1))
            tokens = self.dec_proj[i](self.dec_mamba[i](tokens))
            vim = self.to_image[i](tokens, gh, gw)
            if vim.shape[2:] != img.shape[2:]:
                vim = F.interpolate(vim, size=img.shape[2:], mode='bilinear', align_corners=False)
            pyr = self.pyr_fuse[i](torch.cat([img, vim], dim=1))
            img = pyr
            p = F.relu(self.pyr_conv[i](pyr), inplace=False)
            p = F.relu(self.pyr_proj[i](p), inplace=False)
            p = F.interpolate(p, size=(h0, w0), mode='bilinear', align_corners=False)
            outs.append(p)
        y = self.head(torch.cat(outs, dim=1))
        if y.shape[2:] != (h0, w0):
            y = F.interpolate(y, size=(h0, w0), mode='bilinear', align_corners=False)
        return y

def create_mambacracknet(num_classes=1, img_size=448, in_channels=3):
    return MambaCrackNet(img_size=img_size, patch_size=8, in_channels=in_channels, num_classes=num_classes, base_dim=64, expand=2, d_state=16, depth=4)
if __name__ == '__main__':
    print('cuda_scan:', _HAS_CUDA_SCAN)
    m = create_mambacracknet(1, img_size=448)
    y = m(torch.randn(1, 3, 448, 448))
    print(y.shape, sum((p.numel() for p in m.parameters())) / 1000000.0)
