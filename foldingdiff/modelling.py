import math
import torch
from torch import nn


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier features for time embedding.
    This is the same module used in DDPM and FoldingDiff.

    Args:
        embed_dim (int): Output dimension for the time embedding.

    Behavior:
        Input:  t  (shape: [batch])
        Output: embedding of shape [batch, embed_dim]
    """

    def __init__(self, embed_dim, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim

        # The projection matrix W ~ N(0, scale^2)
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False
        )

    def forward(self, t):
        """
        t : shape (batch,) or (batch,1)
        returns: shape (batch, embed_dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        # Produce a broadcastable projection
        # shape: [batch, embed_dim//2]
        wt = t[:, None] * self.W[None, :]

        return torch.cat(
            [torch.sin(wt), torch.cos(wt)],
            dim=-1
        )
