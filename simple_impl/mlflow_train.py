import logging
import gc

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from timeit import default_timer as timer


import mlflow
import mlflow.pytorch


from data.protein import CATHDataset
from data.tensordataset import DataLoader
from likelihood import Likelihood
from losses import get_mix_loss_fn
from models.networks import ScoreNetwork
from schedule import LinearBetaSchedule
from sde_lib import DiffusionMixture
from util.ema import ExponentialMovingAverage
from util.loggers_pl import CSVLogger

log = logging.getLogger(__name__)
logger = CSVLogger('logs', flush_logs_every_n_steps=1000)

# PARÂMETROS GERAIS DO TREINAMENTO
grad_norm = 1.0
steps = 200000
train_val = True
val_freq = 1000
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_batch_size = 8192
batch_size = 8192
patience = 20000
best_val = False
lr_sched = False
learning_rate = 0.0002
hid_dim = 512
num_layers = 6
window_size = 50

# MODIFICAÇÃO 2: Carregar os dados do CATH
dataset = CATHDataset(tsv_path="./data/cath_s40_L64.tsv", window_size=window_size)

N = len(dataset)
N_val = N_test = N // 10
N_train = N - N_val - N_test
train_ds, eval_ds, test_ds = torch.utils.data.random_split(
    dataset,
    [N_train, N_val, N_test],
    generator=torch.Generator().manual_seed(seed),
)

train_ds, eval_ds, test_ds = (
    DataLoader(train_ds, batch_dims=batch_size, shuffle=True),
    DataLoader(eval_ds, batch_dims=eval_batch_size),
    DataLoader(test_ds, batch_dims=eval_batch_size),
)

beta_schedule = LinearBetaSchedule(beta_0=0.2, beta_f=0.001, t0=0, tf=1.0)
manifold = dataset.manifold

mix = DiffusionMixture(
    manifold,
    beta_schedule,
    mix_type='log',
    drift_scale=1.0,
    pred=False,
    pred_scale=1.0,
    prior_type='unif'
)
loss_fn = get_mix_loss_fn(mix, num_steps=15, eps=0.001, weight_type='default')
likelihood = Likelihood(mix, rtol=1e-5, atol=1e-5)

torus_dim = dataset.torus_dim
out_dim = torus_dim * 2

model_params = dict(
    num_layers=num_layers,
    hid_dim=hid_dim,
    act='swish',
    in_dim=out_dim + 1,
    out_dim=out_dim,
    manifold=manifold
)
modelf = ScoreNetwork(**model_params).to(device)
modelb = ScoreNetwork(**model_params).to(device)

emaf = ExponentialMovingAverage(parameters=modelf.parameters(), decay=0.9999)
emab = ExponentialMovingAverage(parameters=modelb.parameters(), decay=0.9999)

optimizerf = torch.optim.Adam(modelf.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
optimizerb = torch.optim.Adam(modelb.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

schedulerf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerf, T_max=steps)
schedulerb = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerb, T_max=steps)

def evaluate(stage, step, **kwargs):
    try:
        dataset = eval_ds if stage == "val" else test_ds
        emaf.copy_to(modelf.parameters())
        emab.copy_to(modelb.parameters())
        likelihood_fn = likelihood.get_log_prob(modelf, modelb)

        logp, nfe, N = 0.0, 0.0, 0
        for batch in dataset:
            if len(batch) > 0:
                logp_step, nfe_step = likelihood_fn(batch.to(device))
                logp += logp_step.sum()
                nfe += nfe_step
                N += logp_step.shape[0]
        nfe /= len(dataset)
        logp /= N

        logger.log_metrics({f"{stage}/logp": logp}, step)
        logger.log_metrics({f"{stage}/nfe": nfe}, step)
        
        
        mlflow.log_metric(f"{stage}_logp", logp.item(), step=step)
        mlflow.log_metric(f"{stage}_nfe", nfe, step=step)

        with logging_redirect_tqdm():
            log.info(f"[Epoch {step:06d}] {stage} logp: {logp:.3f} | nfe: {nfe:.1f}")
        logger.save()
        return logp
    except:
        return -10000

def train(step=0):
    tbar = tqdm(
        range(step, steps),
        total=steps - step,
        bar_format="{desc}{bar}{r_bar}",
        mininterval=1,
    )
    train_time = timer()
    total_train_time = 0

    for _ in tbar:
        batch = next(train_ds).to(device)
        optimizerf.zero_grad()
        optimizerb.zero_grad()
        loss, lossf, lossb = loss_fn(modelf, modelb, batch)
        loss.backward()

        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(modelf.parameters(), grad_norm)
            torch.nn.utils.clip_grad_norm_(modelb.parameters(), grad_norm)

        optimizerf.step()
        optimizerb.step()

        if lr_sched:
            schedulerf.step()
            schedulerb.step()

        emaf.update(modelf.parameters())
        emab.update(modelb.parameters())
        step += 1

        if torch.isnan(loss).any():
            log.warning("Loss is nan")
            return False

        if step % 10 == 0:
            loss_val, lossf_val, lossb_val = loss.item(), lossf.item(), lossb.item()
            logger.log_metrics({"train/loss_f": lossf_val}, step)
            logger.log_metrics({"train/loss_b": lossb_val}, step)
            logger.log_metrics({"train/loss_total": loss_val}, step)

           
            mlflow.log_metric("train_loss_total", loss_val, step=step)
            mlflow.log_metric("train_loss_f", lossf_val, step=step)
            mlflow.log_metric("train_loss_b", lossb_val, step=step)
            
            tbar.set_description(f"Total: {loss_val:.2f} | F: {lossf_val:.2f} | B: {lossb_val:.2f}")

        if step % val_freq == 0:
            if train_val:
                evaluate("val", step)
                gc.collect()
            train_time = timer()

    return True

if __name__ == "__main__":
    with mlflow.start_run():
        print("MLflow Run Started...")
        
        
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("total_steps", steps)
        mlflow.log_param("hidden_dim", hid_dim)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("gradient_clipping", grad_norm)
        mlflow.log_param("seed", seed)

        success = train(step=0)

        if success:
            print("Training finished successfully. Saving model to MLflow...")
            emaf.copy_to(modelf.parameters())
            emab.copy_to(modelb.parameters())

            
            mlflow.pytorch.log_model(modelf, "model_f")
            mlflow.pytorch.log_model(modelb, "model_b")
            
            print("Model weights saved as MLflow artifacts.")
        else:
            print("Training failed.")
    
    logger.save()
    logger.finalize("success" if success else "failure")
