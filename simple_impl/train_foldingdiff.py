import random
from pathlib import Path
import torch
from typing import Iterator
import numpy as np
from torch.utils.data import DataLoader, Sampler
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
# device = torch.device("cpu")
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

class SameLenSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        indices_by_seq_lens = {}
        for i in range(len(dataset)):
            item = dataset[i]
            seq_len = item['angles'].shape[0]
            indices = indices_by_seq_lens.get(seq_len, [])
            indices_by_seq_lens[seq_len] = indices + [i]
        self.indices_by_seq_lens = indices_by_seq_lens
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[list[int]]:        
        batches = []
        for seq_len in self.indices_by_seq_lens.keys():
            indices = torch.tensor(self.indices_by_seq_lens[seq_len])
            perm = torch.randperm(len(indices)) if self.shuffle else torch.arange(len(indices))
            indices = indices[perm]
            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
            for batch in torch.chunk(indices, num_batches):
                batches.append(batch.tolist())
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

exhaustive_t = False
# noised_ds_args = dict(
#     dset_key="angles",
#     timesteps=1000,
#     exhaustive_t=False,
#     beta_schedule="linear",
#     nonangular_variance=1.0,
#     angular_variance=np.pi,
# )
# train_noised_dataset = NoisedAnglesDataset(dset=train_dataset, **noised_ds_args)
# val_noised_dataset = NoisedAnglesDataset(dset=val_dataset, **noised_ds_args)


train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_sampler=SameLenSampler(train_dataset, batch_size=16, shuffle=True),
    num_workers=4,
    pin_memory=True,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_sampler=SameLenSampler(val_dataset, batch_size=16, shuffle=False),
    num_workers=4,
    pin_memory=True,
)

angles_per_residue = len(train_dataset.feature_names["angles"])  # 6 right now
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

mix = DiffusionMixture(
    beta_schedule,
    mix_type="log",
    drift_scale=1.0,
    pred=False,
    pred_scale=1.0,
    prior_type="unif",
)
loss_fn = get_mix_loss_fn(mix, num_steps=15, eps=0.001, weight_type="default")

def mean_ignoring_outliers_iqr(data_tensor):
    """
    Calculates the mean of a PyTorch tensor, ignoring outliers using the IQR method.
    """
    q1 = torch.quantile(data_tensor, 0.25)
    q3 = torch.quantile(data_tensor, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create a mask for inliers
    inlier_mask = (data_tensor >= lower_bound) & (data_tensor <= upper_bound)

    # Filter out outliers
    inliers = data_tensor[inlier_mask]

    # Calculate the mean of the inliers
    if inliers.numel() > 0:  # Check if there are any inliers left
        return torch.mean(inliers)
    else:
        return torch.tensor(float('nan')) # Return NaN if no inliers remain


def train():
    tbar = tqdm(
        range(0, steps),
        total=steps,
        bar_format="{desc}{bar}{r_bar}",
        mininterval=1,
    )

    for epoch in tbar:
        epoch_lossf = []
        epoch_lossb = []
        for i, batch in enumerate(train_dataloader):
            print(f"epoch {epoch + 1}, batch {i + 1}")
            data = batch['cossin'].to(device)
            manifold_dim = batch['angles'].shape[1] * angles_per_residue

            manifold = Torus(manifold_dim)

            optimizerf.zero_grad()
            optimizerb.zero_grad()

            loss, lossf, lossb = loss_fn(manifold, modelf, modelb, data)
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

            epoch_lossf.append(lossf)
            epoch_lossb.append(lossb)
            if torch.isnan(lossf + lossb).any():
                print("Loss is nan")
                return False

        
        torch.save(modelf.state_dict(), './forward_bert.pt')
        torch.save(modelb.state_dict(), './backward_bert.pt')

        epoch_lossf = mean_ignoring_outliers_iqr(torch.tensor(epoch_lossf))
        epoch_lossb = mean_ignoring_outliers_iqr(torch.tensor(epoch_lossb))
        tbar.set_description(f"F: {epoch_lossf:.2f} | B: {epoch_lossb:.2f}")
    return True

if __name__ == "__main__":
    success = train()
