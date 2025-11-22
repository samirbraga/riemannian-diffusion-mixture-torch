import random
import torch
from typing import Iterator

from torch.utils.data import DataLoader, Sampler
import wandb
from tqdm import tqdm
from transformers import BertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from foldingdiff.bert_for_diffusion import BertForDiffusion
from foldingdiff.datasets import CathCanonicalAnglesOnlyDataset
from geomstats.geometry.torus import Torus
from losses import get_mix_loss_fn
from schedule import LinearBetaSchedule
from sde_lib import DiffusionMixture
# from util.ema import ExponentialMovingAverage
from dotenv import load_dotenv

load_dotenv()

LEARNING_RATE = 5e-5

max_seq_len = 128
min_seq_len = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
steps = 100
grad_norm = 1.0
lr_sched = True

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

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_sampler=SameLenSampler(train_dataset, batch_size=32, shuffle=True),
    num_workers=4,
    pin_memory=True,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_sampler=SameLenSampler(val_dataset, batch_size=32, shuffle=False),
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

modelf = BertForDiffusion(config=cfg, ft_names=train_dataset.feature_names["angles"]).to(device)
modelb = BertForDiffusion(config=cfg, ft_names=train_dataset.feature_names["angles"]).to(device)

optimizerf = torch.optim.AdamW(modelf.parameters(), lr=LEARNING_RATE, weight_decay=0)
optimizerb = torch.optim.AdamW(modelb.parameters(), lr=LEARNING_RATE, weight_decay=0)

# emaf = ExponentialMovingAverage(modelf.parameters(), decay=0.9999)
# emab = ExponentialMovingAverage(modelb.parameters(), decay=0.9999)
schedulerf = get_linear_schedule_with_warmup(
    optimizer=optimizerf,
    num_warmup_steps=int(steps * 0.1),
    num_training_steps=steps
)
schedulerb = get_linear_schedule_with_warmup(
    optimizer=optimizerb,
    num_warmup_steps=int(steps * 0.1),
    num_training_steps=steps
)

beta_schedule = LinearBetaSchedule(beta_0=0.2, beta_f=0.001, t0=0, tf=1.0)

mix = DiffusionMixture(
    beta_schedule,
    mix_type="log",
    drift_scale=1.0,
    pred=False,
    pred_scale=1.0,
    prior_type="unif",
)
loss_fn = get_mix_loss_fn(mix, num_steps=100, eps=0.001, weight_type="default")

def mean_ignoring_outliers(data_tensor):
    # Create a mask for inliers
    inlier_mask = (data_tensor <= 10e8)

    # Filter out outliers
    inliers = data_tensor[inlier_mask]

    if inliers.numel() > 0:  # Check if there are any inliers left
        return torch.mean(inliers)
    else:
        return torch.tensor(float('nan')) # Return NaN if no inliers remain


def train():
    run = wandb.init(entity='rdem', project='Standard Metric - RiemannDiff')

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
            data = batch['cossin'].to(device)
            manifold_dim = batch['angles'].shape[1] * angles_per_residue

            manifold = Torus(manifold_dim)

            optimizerf.zero_grad()
            optimizerb.zero_grad()

            loss, lossf, lossb = loss_fn(manifold, modelf, modelb, data)
            loss.backward()

            # if grad_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(modelf.parameters(), grad_norm)
            #     torch.nn.utils.clip_grad_norm_(modelb.parameters(), grad_norm)

            optimizerf.step()
            optimizerb.step()

            if lr_sched:
                schedulerf.step()
                schedulerb.step()

            # -------- EMA update --------
            # emaf.update(modelf.parameters())
            # emab.update(modelb.parameters())

            epoch_lossf.append(lossf)
            epoch_lossb.append(lossb)
            if torch.isnan(lossf + lossb).any():
                print("Loss is nan")
                return False
        
        torch.save(modelf.state_dict(), './forward_bert.pt')
        torch.save(modelb.state_dict(), './backward_bert.pt')

        epoch_lossf = mean_ignoring_outliers(torch.tensor(epoch_lossf))
        epoch_lossb = mean_ignoring_outliers(torch.tensor(epoch_lossb))

        run.log({"lossf": epoch_lossf, "lossb": epoch_lossb}, step=epoch)
        tbar.set_description(f"F: {epoch_lossf:.2f} | B: {epoch_lossb:.2f}")
    return True

if __name__ == "__main__":
    success = train()
