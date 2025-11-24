import torch
import torch.nn.functional as F
from foldingdiff.bert_for_diffusion_base import BertForDiffusionBase
from foldingdiff.utils import get_bert_attn_tensors


class BertForDiffusion(BertForDiffusionBase):
    def forward(self, inputs: torch.Tensor, timestep: torch.Tensor, manifold):
        inputs_2d = inputs.view(inputs.shape[0], -1, 12)
        attn_mask, position_ids, masked_inputs = get_bert_attn_tensors(128, inputs_2d)
        outputs = super().forward(masked_inputs, timestep, attn_mask, position_ids)
        outputs = outputs[:, :inputs_2d.shape[1], :]
        outputs = outputs.reshape(inputs.shape[0], -1)
        drift = manifold.to_tangent(outputs, inputs)
        return drift