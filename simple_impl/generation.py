import torch
import pandas as pd
from transformers import BertConfig
from geomstats.geometry.torus import Torus
from schedule import LinearBetaSchedule
from sde_lib import DiffusionMixture
from solver import get_pc_sampler
from foldingdiff.bert_for_diffusion import BertForDiffusion
from foldingdiff.angles_and_coords import create_new_chain_nerf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_seq_len = 128

angles_per_residue = 6
cfg = BertConfig(
    max_position_embeddings=max_seq_len * angles_per_residue,
    num_attention_heads=6,
    hidden_size=192,
    intermediate_size=384,
    num_hidden_layers=6,
    position_embedding_type="relative_key",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    use_cache=False,
    _attn_implementation="eager"
)

ft_names = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
modelf = BertForDiffusion(
    config=cfg,
    ft_names=ft_names,
    lr=5e-05,
    l2=0.0,
    l1=0.0,
    epochs=10000,
).to(device)
modelf.load_state_dict(torch.load('./forward_bert.pt', weights_only=True))

modelb = BertForDiffusion(
    config=cfg,
    ft_names=ft_names,
    lr=5e-05,
    l2=0.0,
    l1=0.0,
    epochs=10000,
).to(device)
modelb.load_state_dict(torch.load('./backward_bert.pt', weights_only=True))

beta_schedule = LinearBetaSchedule(beta_0=0.2, beta_f=0.001, t0=0, tf=1.0)
    
mix = DiffusionMixture(
    beta_schedule,
    mix_type="log",
    drift_scale=1.0,
    pred=False,
    pred_scale=1.0,
    prior_type="unif",
)

fdrift_fn = mix.get_drift_fn(modelf)
bdrift_fn = mix.rev().get_drift_fn(modelb)
sde = mix.approx(fdrift_fn, bdrift_fn, True)
manifold = Torus(80 * 6)
shape = (4,)
sampler = get_pc_sampler(manifold=manifold, sde=sde, shape=shape, N=10, eps=0.001, device=device)


sample = sampler(prior_samples=None)

angles = torch.acos(sample.reshape(shape[0], -1, 2)[:, :, [0]]).reshape(shape[0], -1, 6)

df = pd.DataFrame(angles.cpu().numpy(), columns=ft_names)


print(sample.shape)