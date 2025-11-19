from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from tqdm import tqdm
from transformers import BertConfig
from foldingdiff.bert_for_diffusion import BertForDiffusion
from foldingdiff.datasets import NoisedAnglesDataset, CathCanonicalAnglesOnlyDataset
from geomstats.geometry.torus import Torus
from losses import get_mix_loss_fn
from schedule import LinearBetaSchedule
from sde_lib import DiffusionMixture
from util.ema import ExponentialMovingAverage

LEARNING_RATE = 2e-5

max_seq_len = 128
min_seq_len = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
steps = 200000
grad_norm = 1.0
lr_sched = False

ds_args = dict(
    pad=max_seq_len,
    min_length=min_seq_len,
    trim_strategy="leftalign",
    zero_center=True,
    toy=None,
    pdbs="cath",
)
train_dataset = CathCanonicalAnglesOnlyDataset(split="train", **ds_args)
val_dataset = CathCanonicalAnglesOnlyDataset(split="validation", **ds_args)

exhaustive_t = False
noised_ds_args = dict(
    dset_key="angles",
    timesteps=1000,
    exhaustive_t=False,
    beta_schedule="linear",
    nonangular_variance=1.0,
    angular_variance=np.pi,
)
train_noised_dataset = NoisedAnglesDataset(dset=train_dataset, **noised_ds_args)
val_noised_dataset = NoisedAnglesDataset(dset=val_dataset, **noised_ds_args)

dl_args = dict(
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
train_dataloader = DataLoader(
    dataset=train_noised_dataset,
    **{**dl_args, "shuffle": True},  # Shuffle only train loader
)
val_dataloader = DataLoader(dataset=val_noised_dataset, **dl_args)

cfg = BertConfig(
    max_position_embeddings=max_seq_len,
    num_attention_heads=12,
    hidden_size=384,
    intermediate_size=768,
    num_hidden_layers=12,
    position_embedding_type="relative_key",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    use_cache=False,
)

modelf = BertForDiffusion(
    config=cfg,
    ft_names=train_dataset.feature_names["angles"],
    lr=5e-05,
    l2=0.0,
    l1=0.0,
    epochs=10000,
    steps_per_epoch=len(train_dataloader),
).to(device)
modelb = BertForDiffusion(
    config=cfg,
    ft_names=train_dataset.feature_names["angles"],
    lr=5e-05,
    l2=0.0,
    l1=0.0,
    epochs=10000,
    steps_per_epoch=len(train_dataloader),
).to(device)

optimizerf = torch.optim.Adam(modelf.parameters(), lr=LEARNING_RATE)
optimizerb = torch.optim.Adam(modelb.parameters(), lr=LEARNING_RATE)

emaf = ExponentialMovingAverage(modelf.parameters(), decay=0.9999)
emab = ExponentialMovingAverage(modelb.parameters(), decay=0.9999)

schedulerf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerf, T_max=10000)
schedulerb = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerb, T_max=10000)
beta_schedule = LinearBetaSchedule(beta_0=0.2, beta_f=0.001, t0=0, tf=1.0)

manifold = Torus(max_seq_len * 6)
mix = DiffusionMixture(
    manifold,
    beta_schedule,
    mix_type="log",
    drift_scale=1.0,
    pred=False,
    pred_scale=1.0,
    prior_type="unif",
)
loss_fn = get_mix_loss_fn(mix, num_steps=15, eps=0.001, weight_type="default")

train_dataloader_iter = iter(train_dataloader)


def train(step=0):
    tbar = tqdm(
        range(step, steps),
        total=steps - step,
        bar_format="{desc}{bar}{r_bar}",
        mininterval=1,
    )

    for _ in tbar:
        batch = next(train_dataloader_iter)

        data = batch['cossin'].to(device)
        attention_mask = batch['attn_mask'].to(device)
        position_ids = batch['position_ids'].to(device)

        optimizerf.zero_grad()
        optimizerb.zero_grad()

        loss, lossf, lossb = loss_fn(modelf, modelb, data, attention_mask, position_ids)
        loss.backward()

        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(modelf.parameters(), grad_norm)
            torch.nn.utils.clip_grad_norm_(modelb.parameters(), grad_norm)

        optimizerf.step()
        optimizerb.step()

        if lr_sched:
            schedulerf.step()
            schedulerb.step()

        # -------- EMA update --------
        emaf.update(modelf.parameters())
        emab.update(modelb.parameters())

        step += 1

        if torch.isnan(lossf + lossb).any():
            print("Loss is nan")
            return False

        if step % 10 == 0:
            tbar.set_description(f"F: {lossf:.2f} | B: {lossb:.2f}")
    return True


if __name__ == "__main__":
    success = train(step=0)
