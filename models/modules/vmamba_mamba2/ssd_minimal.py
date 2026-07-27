import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def segsum_unstable(x):
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def segsum(x):
    T = x.size(-1)
    x = repeat(x, '... d -> ... d e', e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0
    X, A, B, C = [rearrange(x, 'b (c l) ... -> b c l ...', l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, 'b c l h -> b h c l')
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp', C, B, L, X)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum('bclhn,bhcl,bclhp->bchpn', B, decay_states, X)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
    states, final_state = (new_states[:, :-1], new_states[:, -1])
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    Y = rearrange(Y_diag + Y_off, 'b c l h p -> b (c l) h p')
    return (Y, final_state)

def mamba_chunk_scan_combined_torch(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float('inf')), return_final_states=False):
    batch, seqlen, ngroups, dstate = B.shape
    nheads, headdim = x.shape[2:]
    while seqlen % chunk_size != 0:
        chunk_size = chunk_size >> 1
    if nheads != ngroups:
        assert nheads % ngroups == 0
        B = B.view(batch, seqlen, ngroups, 1, dstate).repeat(1, 1, 1, nheads // ngroups, 1).view(batch, seqlen, nheads, dstate)
        C = C.view(batch, seqlen, ngroups, 1, dstate).repeat(1, 1, 1, nheads // ngroups, 1).view(batch, seqlen, nheads, dstate)
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    u = x * dt.unsqueeze(-1)
    w = A * dt
    y, state = ssd_minimal_discrete(u, w, B, C, block_len=chunk_size, initial_states=initial_states)
    if D is not None:
        y = y + D.view(y.shape[-2], -1) * x
    if z is not None:
        y = y * (z * torch.sigmoid(z))
    return (y, state) if return_final_states else y
WITH_TRITON = True
try:
    import triton
except ImportError:
    WITH_TRITON = False
if WITH_TRITON:
    try:
        from .ssd_combined import mamba_chunk_scan_combined
    except ImportError:
        from ssd_combined import mamba_chunk_scan_combined

def selective_scan_chunk_fn(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float('inf')), return_final_states=False, backend=None):
    fn = mamba_chunk_scan_combined_torch if backend == 'torch' or not WITH_TRITON else mamba_chunk_scan_combined
    return fn(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, dt_softplus, dt_limit, return_final_states)

def test_correctness():
    torch.manual_seed(42)
    batch, seqlen, chunk_size, dim, headdim = (1, 2048, 64, 2048, 64)
    nheads = dim // headdim
    ngroups = 1
    ngroups = nheads
    dstate = 64
    dtype = torch.float32
    device = 'cuda'
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dtype=dtype, device=device)
    yto = selective_scan_chunk_fn(x, dt, A, B, C, chunk_size=64, D=D, backend='torch')
    ytr = selective_scan_chunk_fn(x, dt, A, B, C, chunk_size=64, D=D, backend='triton')
    print((yto - ytr).abs().max())
    breakpoint()
    ...
if __name__ == '__main__':
    test_correctness()
    breakpoint()