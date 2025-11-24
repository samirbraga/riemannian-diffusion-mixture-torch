import torch
from models.layers import MLP
import torch.nn.functional as F


class ScoreNetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, out_dim, act, **kwargs):
        super().__init__()
        self.layer = MLP(num_layers, in_dim, hid_dim, out_dim, act)

    def forward(self, x, t, manifold):
        if len(t.shape) == len(x.shape)-1:
            t = t.unsqueeze(-1)

        pad = 128 * 6 * 2
        seq_len = x.shape[1]
    
        if seq_len < pad:
            padded_x = F.pad(
                x,
                (0, pad - seq_len),
                mode="constant",
                value=0,
            )
        elif seq_len > pad:
            padded_x = x[:, :pad]
        else:
            padded_x = x

        output = self.layer(torch.cat([padded_x, t], dim=-1))
        output = output[:, :x.shape[1]]
        drift = manifold.to_tangent(output, x)
        return drift