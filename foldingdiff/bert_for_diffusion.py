import torch
import torch.nn.functional as F
from foldingdiff.bert_for_diffusion_base import BertForDiffusionBase
from foldingdiff.utils import get_bert_attn_tensors


class BertForDiffusion(BertForDiffusionBase):
    
    # def forward(self, inputs: torch.Tensor, timestep: torch.Tensor, manifold):
    #     inputs_2d = inputs.view(inputs.shape[0], -1, 12)
    #     attn_mask, position_ids, masked_inputs = get_bert_attn_tensors(128, inputs_2d)
    #     outputs = super().forward(masked_inputs, timestep, attn_mask, position_ids)
    #     outputs = outputs[:, :inputs_2d.shape[1], :]
    #     outputs = outputs.reshape(inputs.shape[0], -1)
    #     drift = manifold.to_tangent(outputs, inputs)
    #     return drift
    
    def forward(self, inputs: torch.Tensor, timestep: torch.Tensor, manifold, seq_len):
        bs = inputs.shape[0]
        pad = inputs.shape[1] // 12
        inputs_2d = inputs.view(inputs.shape[0], -1, 12)   
            
        attn_mask = torch.zeros(size=(bs, pad), device=inputs.device)
        l = min(pad, seq_len)
        attn_mask[:, :l] = 1.0

        position_ids = torch.arange(start=0, end=pad, step=1, dtype=torch.long, device=inputs.device)
        
        outputs = super().forward(inputs_2d, timestep, attn_mask, position_ids)
        outputs = outputs.reshape(bs, -1)
        drift = manifold.to_tangent(outputs, inputs)
        return drift