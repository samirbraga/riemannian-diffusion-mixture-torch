import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from foldingdiff.modelling import GaussianFourierProjection


class BertForRiemannianDiffusion(nn.Module):
    """
    A Transformer-based Score Network for Riemannian Diffusion
    that predicts tangent vectors on a manifold. I hope I implemented it ok ASAP :-D "
    """

    def __init__(
        self,
        config: BertConfig,
        manifold,
        n_inputs: int,
        time_embed_dim: int = None,
    ):
        super().__init__()

        self.manifold = manifold
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_inputs = n_inputs
        self.time_embed_dim = time_embed_dim or config.hidden_size

        #  Input projection: angles → hidden space
        self.inputs_to_hidden = nn.Linear(n_inputs, config.hidden_size)

        #  Time embedding
        
        self.time_embed = GaussianFourierProjection(
            embed_dim=self.time_embed_dim
        )
        self.time_to_hidden = nn.Linear(self.time_embed_dim, config.hidden_size)

        #  BERT encoder backbone
        self.encoder = BertEncoder(config)

        #  Output projection: hidden → angles
        self.hidden_to_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=1e-12),
            nn.Linear(config.hidden_size, n_inputs),
        )

        # Input embeddings (LN + dropout)
        self.embeddings = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self._init_weights()

    def _init_weights(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, t, attention_mask=None):
        """
        x: (B, L, D) — torus angles (already mapped to cos/sin if using 2D)
        t: (B,) or (B,1)
        attention_mask: (B, L)
        """
        B, L, D = x.shape

        # Ensure (B,1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        #  Project angles to hidden dimension 
        h = self.inputs_to_hidden(x)
        h = self.embeddings(h)

        # Add time embedding 
        t_embed = self.time_embed(t.squeeze(-1))
        t_embed = self.time_to_hidden(t_embed)
        t_embed = t_embed.unsqueeze(1)
        h = h + t_embed

        #  Attention mask 
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=x.device)
        extended_mask = attention_mask[:, None, None, :].to(h.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        # Transformer Encoding ===
        encoded = self.encoder(
            h,
            attention_mask=extended_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )[0]

        #  Project back to feature (angle) dimension 
        out_euc = self.hidden_to_output(encoded)

        #  Projection  to the tangent space of manifold here:
        
        out_tangent = self.manifold.to_tangent(out_euc, x)

        return out_tangent