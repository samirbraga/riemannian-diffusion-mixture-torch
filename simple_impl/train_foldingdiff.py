import random
import torch
from typing import Iterator

from accelerate import Accelerator
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import BertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from foldingdiff.bert_for_diffusion import BertForDiffusion
from foldingdiff.datasets import CathCanonicalAnglesOnlyDataset
from geomstats.geometry.torus import Torus
from losses import get_mix_loss_fn
from schedule import LinearBetaSchedule
from sde_lib import DiffusionMixture
from dotenv import load_dotenv

load_dotenv()

LEARNING_RATE = 5e-5

max_seq_len = 128
min_seq_len = 40
steps = 100
grad_norm = 1.0
lr_sched = True

ds_args = dict(
    pad=max_seq_len,
    min_length=min_seq_len,
    trim_strategy="leftalign",
    zero_center=False,
    toy=None,
    pdbs="cath",
)
train_dataset = CathCanonicalAnglesOnlyDataset(split="train", **ds_args)
val_dataset = CathCanonicalAnglesOnlyDataset(split="validation", **ds_args)

class SameLenSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, shuffle_lengths: bool = True, shuffle_items: bool = True):
        indices_by_seq_lens = {}
        for i in range(len(dataset)):
            item = dataset[i]
            seq_len = item['angles'].shape[0]
            indices = indices_by_seq_lens.get(seq_len, [])
            indices_by_seq_lens[seq_len] = indices + [i]
        self.indices_by_seq_lens = indices_by_seq_lens
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle_lengths = shuffle_lengths
        self.shuffle_items = shuffle_items

    def __len__(self) -> int:
        total_batches = 0
        for seq_len in self.indices_by_seq_lens.keys():
            indices = self.indices_by_seq_lens[seq_len]
            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
            total_batches += num_batches
        return total_batches
    
    def __iter__(self) -> Iterator[list[int]]:        
        batches = []
        for seq_len in sorted(self.indices_by_seq_lens.keys()):
            indices = torch.tensor(self.indices_by_seq_lens[seq_len])
            perm = torch.randperm(len(indices)) if self.shuffle_items else torch.arange(len(indices))
            indices = indices[perm]
            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
            for batch in torch.chunk(indices, num_batches):
                batches.append(batch.tolist())
        if self.shuffle_lengths:
            random.shuffle(batches)
        yield from batches

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_sampler=SameLenSampler(train_dataset, batch_size=32, shuffle_lengths=False, shuffle_items=True),
    num_workers=4,
    pin_memory=True,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_sampler=SameLenSampler(val_dataset, batch_size=32, shuffle_lengths=False, shuffle_items=False),
    num_workers=4,
    pin_memory=True,
)

angles_per_residue = len(train_dataset.feature_names["angles"])  # 6 right now
cfg = BertConfig(
    max_position_embeddings=max_seq_len,
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

modelf = BertForDiffusion(config=cfg, ft_names=train_dataset.feature_names["angles"])
modelb = BertForDiffusion(config=cfg, ft_names=train_dataset.feature_names["angles"])

optimizerf = torch.optim.AdamW(modelf.parameters(), lr=LEARNING_RATE, weight_decay=0)
optimizerb = torch.optim.AdamW(modelb.parameters(), lr=LEARNING_RATE, weight_decay=0)

num_training_steps = steps * len(train_dataloader)
schedulerf = get_linear_schedule_with_warmup(
    optimizer=optimizerf, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps
)
schedulerb = get_linear_schedule_with_warmup(
    optimizer=optimizerb, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps
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
loss_fn = get_mix_loss_fn(mix, reduce_mean=True, num_steps=50, eps=0.001, weight_type="default")

def mean_ignoring_outliers(data_tensor):
    inlier_mask = (data_tensor <= 10e8)
    inliers = data_tensor[inlier_mask]

    if inliers.numel() > 0:  # Check if there are any inliers left
        return torch.mean(inliers)
    else:
        return torch.tensor(float('nan')) # Return NaN if no inliers remain


def train():
    accelerator = Accelerator(mixed_precision="fp16", split_batches=True, log_with="wandb")
    accelerator.init_trackers(
        project_name="Standard Metric - RiemannDiff",
        init_kwargs={"wandb": {"entity": "rdem"}}
    )

    device = accelerator.device

    accelerator.wait_for_everyone()

    modelf_prep, modelb_prep, optimizerf_prep, optimizerb_prep, schedulerf_prep, schedulerb_prep, train_dl = accelerator.prepare(
        modelf, modelb, optimizerf, optimizerb, schedulerf, schedulerb, train_dataloader
    )

    tbar = tqdm(
        range(0, steps),
        total=steps,
        bar_format="{desc}{bar}{r_bar}",
        mininterval=1,
    ) if accelerator.is_main_process else range(0, steps)

    torus_map = {
        i: Torus((i - 1) * angles_per_residue)
        for i in range(min_seq_len, max_seq_len + 1)
    }

    min_lossb = 1e8

    for epoch in tbar:
        epoch_lossf = []
        epoch_lossb = []
        for i, batch in enumerate(train_dl):
            data = batch['cossin'].to(device)
            seq_len = batch['angles'].shape[1]
            manifold = torus_map[seq_len]

            optimizerf_prep.zero_grad(set_to_none=True)
            optimizerb_prep.zero_grad(set_to_none=True)

            with accelerator.autocast():
                loss, lossf, lossb = loss_fn(manifold, modelf_prep, modelb_prep, data)

            accelerator.backward(loss)
            if grad_norm > 0:
                all_params = list(modelf_prep.parameters()) + list(modelb_prep.parameters())
                accelerator.clip_grad_norm_(all_params, grad_norm)
            optimizerf_prep.step()
            optimizerb_prep.step()

            if lr_sched:
                schedulerf_prep.step()
                schedulerb_prep.step()

            lossf_reduced = accelerator.gather_for_metrics(lossf.detach()).mean()
            lossb_reduced = accelerator.gather_for_metrics(lossb.detach()).mean()

            if accelerator.is_main_process:
                epoch_lossf.append(lossf_reduced.cpu())
                epoch_lossb.append(lossb_reduced.cpu())

            if torch.isnan(lossf_reduced + lossb_reduced).any():
                accelerator.print("Loss is nan")
                return False

        if accelerator.is_main_process:
            epoch_lossf_tensor = torch.stack(epoch_lossf) if len(epoch_lossf) > 0 else torch.tensor([])
            epoch_lossb_tensor = torch.stack(epoch_lossb) if len(epoch_lossb) > 0 else torch.tensor([])

            epoch_lossf_mean = mean_ignoring_outliers(epoch_lossf_tensor)
            epoch_lossb_mean = mean_ignoring_outliers(epoch_lossb_tensor)

            if epoch_lossb_mean < min_lossb:
                min_lossb = epoch_lossb_mean
                torch.save(accelerator.unwrap_model(modelf_prep).state_dict(), './forward_bert.pt')
                torch.save(accelerator.unwrap_model(modelb_prep).state_dict(), './backward_bert.pt')

            accelerator.log({"lossf": epoch_lossf_mean, "lossb": epoch_lossb_mean}, step=epoch)
            tbar.set_description(f"F: {epoch_lossf_mean:.2f} | B: {epoch_lossb_mean:.2f}")
    
    accelerator.end_training()
    return True

if __name__ == "__main__":
    success = train()
