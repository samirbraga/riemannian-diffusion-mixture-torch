import torch
import torch.nn.functional as F
from foldingdiff.bert_for_diffusion_base import BertForDiffusionBase

def get_bert_attn_tensors(pad, coords):
    bs = coords.shape[0]
    seq_len = coords.shape[1]
    attn_mask = torch.zeros(size=(bs, pad), device=coords.device)
    l = min(pad, seq_len)
    attn_mask[:, :l] = 1.0

    if seq_len < pad:
        coords = F.pad(
            coords,
            (0, 0, 0, pad - seq_len),
            mode="constant",
            value=0,
        )
    elif seq_len > pad:
        coords = coords[: , :pad]

    # Create position IDs
    position_ids = torch.arange(start=0, end=pad, step=1, dtype=torch.long, device=coords.device)
    return attn_mask, position_ids, coords


class BertForDiffusion(BertForDiffusionBase):
    def forward(self, inputs: torch.Tensor, timestep: torch.Tensor):
        inputs = inputs.view(inputs.shape[0], -1, 2)
        attn_mask, position_ids, masked_inputs = get_bert_attn_tensors(128 * 6, inputs)
        outputs = super().forward(masked_inputs, timestep, attn_mask, position_ids)
        outputs = outputs[:, :inputs.shape[1], :]
        return outputs.reshape(inputs.shape[0], -1)