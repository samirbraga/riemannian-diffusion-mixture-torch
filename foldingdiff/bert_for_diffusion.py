import time
import torch
import logging
import functools
import numpy as np
from typing import Any, Dict
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup

from foldingdiff.losses import radian_smooth_l1_loss
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


class BertForDiffusion(BertForDiffusionBase, pl.LightningModule):
    def __init__(
        self,
        lr: float = 5e-5,
        l2: float = 0.0,
        l1: float = 0.0,
        epochs: int = 1,
        steps_per_epoch: int = 250,  # Dummy value
        **kwargs,
    ):
        """Feed args to BertForDiffusionBase and then feed the rest into"""
        BertForDiffusionBase.__init__(self, **kwargs)
        # Store information about leraning rates and loss
        self.learning_rate = lr

        self.loss_func = functools.partial(
            radian_smooth_l1_loss, beta=torch.pi / 10
        )

        # pl.utilities.rank_zero_info(f"Using loss: {self.loss_func}")
        if isinstance(self.loss_func, (tuple, list)):
            assert (
                len(self.loss_func) == self.n_inputs
            ), f"Got {len(self.loss_func)} loss functions, expected {self.n_inputs}"

        self.l1_lambda = l1
        self.l2_lambda = l2
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def _get_loss_terms(self, batch) -> torch.Tensor:
        """
        Returns the loss terms for the model. Length of the returned list
        is equivalent to the number of features we are fitting to.
        """
        known_noise = batch["known_noise"]
        predicted_noise = self.forward(
            batch["corrupted"],
            batch["t"],
            attention_mask=batch["attn_mask"],
            position_ids=batch["position_ids"],
        )
        assert (
            known_noise.shape == predicted_noise.shape
        ), f"{known_noise.shape} != {predicted_noise.shape}"

        # Indexes into batch then indices along sequence length
        # attn_mask has shape (batch, seq_len) --> where gives back
        # two lists of values, one for each dimension
        # known_noise has shape (batch, seq_len, num_fts)
        unmask_idx = torch.where(batch["attn_mask"])
        assert len(unmask_idx) == 2
        loss_terms = []
        for i in range(known_noise.shape[-1]):
            loss_fn = self.loss_func
            logging.debug(f"Using loss function {loss_fn}")
            l = loss_fn(
                predicted_noise[unmask_idx[0], unmask_idx[1], i],
                known_noise[unmask_idx[0], unmask_idx[1], i],
            )
            loss_terms.append(l)

        return torch.stack(loss_terms)

    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
        loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss_terms)

        # L1 loss implementation
        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            avg_loss += self.l1_lambda * l1_penalty

        pseudo_ft_names = self.ft_names
        assert len(loss_terms) == len(pseudo_ft_names)
        loss_dict = {
            f"train_loss_{val_name}": val
            for val_name, val in zip(pseudo_ft_names, loss_terms)
        }
        loss_dict["train_loss"] = avg_loss
        self.log_dict(loss_dict)  # Don't seem to need rank zero or sync dist

        self.training_step_outputs.append(avg_loss)
        return avg_loss

    def on_train_epoch_end(self) -> None:
        """Log the average training loss over the epoch"""
        mean_loss = torch.mean(torch.stack(self.training_step_outputs))
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)"
        )
        # Increment counter and timers
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            loss_terms = self._get_loss_terms(batch)

        avg_loss = torch.mean(loss_terms)

        # Log each of the loss terms
        pseudo_ft_names = self.ft_names
        assert len(loss_terms) == len(pseudo_ft_names)
        loss_dict = {
            f"val_loss_{val_name}": self.all_gather(val)
            for val_name, val in zip(pseudo_ft_names, loss_terms)
        }
        loss_dict["val_loss"] = avg_loss
        # with rank zero it seems that we don't need to use sync_dist
        self.log_dict(loss_dict, rank_zero_only=True)

        self.validation_step_outputs.append(avg_loss)
        return {"val_loss": avg_loss}

    def on_validation_epoch_end(self) -> None:
        """Log the average validation loss over the epoch"""
        mean_loss = torch.mean(torch.stack(self.validation_step_outputs))
        pl.utilities.rank_zero_info(
            f"Valid loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f}"
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval: Dict[str, Any] = {"optimizer": optim}

        # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
        # Transformers typically do well with linear warmup
        warmup_steps = int(self.epochs * 0.1)
        pl.utilities.rank_zero_info(
            f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
        )
        retval["lr_scheduler"] = {
            "scheduler": get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.epochs,
            ),
            "frequency": 1,
            "interval": "epoch",  # Call after 1 epoch
        }
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval

    def forward(self, inputs: torch.Tensor, timestep: torch.Tensor):
        inputs = inputs.view(inputs.shape[0], -1, 2)
        attn_mask, position_ids, masked_inputs = get_bert_attn_tensors(128 * 6, inputs)
        outputs = super().forward(masked_inputs, timestep, attn_mask, position_ids)
        outputs = outputs[:, :inputs.shape[1], :]
        return outputs.reshape(inputs.shape[0], -1)