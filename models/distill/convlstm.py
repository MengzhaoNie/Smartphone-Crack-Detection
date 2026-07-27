
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch: int, spatial, device, dtype):
        h, w = spatial
        zeros = torch.zeros(batch, self.hidden_channels, h, w, device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):


    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 1, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        cells = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(cin, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor):

        b, t, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        layer_input = x
        last_h = None
        outputs = None
        for cell in self.cells:
            state = cell.init_state(b, (h, w), device, dtype)
            outs = []
            for ti in range(t):
                state = cell(layer_input[:, ti], state)
                outs.append(state[0])
            outputs = torch.stack(outs, dim=1)
            layer_input = outputs
            last_h = state[0]
        return outputs, last_h
