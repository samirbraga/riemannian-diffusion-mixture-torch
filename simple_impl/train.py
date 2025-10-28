# train.py (versão modificada)

import logging
import gc

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from timeit import default_timer as timer

#  MODIFICAÇÃO 1: Importar a nova classe de treino CATHDataset 

import data.protein
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

# PARÂMETROS GERAIS DO TREINAMENTO (sem alteração) 
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



#  MODIFICAÇÃO 2: Carregar os dados do CATH 


# 2.1. Define o tamanho da janela.
# ps: valor padrão ao longo dos outros scripts
window_size = 50

# 2.2. Instancia o CATHDataset, apontando para o arquivo .tsv gerado.
dataset = CATHDataset(tsv_path="./data/cath_s40_L64.tsv", window_size=window_size)
# ----------------------------------------------------


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

#  MODIFICAÇÃO 3: Obter a geometria (manifold) dinamicamente
# Em vez de ser fixo, o manifold agora é pego diretamente do objeto dataset.
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

# MODIFICAÇÃO 4: Obter a dimensão do Torus dinamicamente 

torus_dim = dataset.torus_dim
out_dim = torus_dim * 2 # O dobro, pois usamos coordenadas (cos, sin)

hid_dim = 512
num_layers = 6


#  Veja que o restante do script segue a lógica original, sem modificações na definição


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

optimizerf = torch.optim.Adam(modelf.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
optimizerb = torch.optim.Adam(modelb.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

schedulerf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerf, T_max=steps)
schedulerb = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerb, T_max=steps)


def evaluate(stage, step, **kwargs):
    try:
        dataset = eval_ds if stage == "val" else test_ds

        emaf.copy_to(modelf.parameters())
        emab.copy_to(modelb.parameters())

        likelihood_fn = likelihood.get_log_prob(modelf, modelb)

        logp, nfe, N = 0.0, 0.0, 0
        tot = 0
        if hasattr(dataset, "__len__"):
            for batch in dataset:
                if len(batch) > 0:
                    logp_step, nfe_step = likelihood_fn(batch.to(device))
                    logp += logp_step.sum()
                    nfe += nfe_step
                    N += logp_step.shape[0]
            nfe /= len(dataset)
        else:
            dataset.batch_dims = eval_batch_size
            num_rounds = round(20_000 / eval_batch_size)
            for i in range(num_rounds):
                batch = next(dataset)
                logp_step, nfe_step = likelihood_fn(batch.to(device))
                logp += logp_step.sum()
                nfe += nfe_step
                N += logp_step.shape[0]
                tot += logp_step.shape[0]
            dataset.batch_dims = batch_size
            nfe /= num_rounds

        logp /= N

        logger.log_metrics({f"{stage}/logp": logp}, step)
        logger.log_metrics({f"{stage}/nfe": nfe}, step)

        with logging_redirect_tqdm():
            if stage == "test" and best_val:
                log.info(f">>> [Epoch {step:06d}] | Val logp={kwargs['best_logp']:.3f} | "
                         f"Test logp={logp:.3f} | nfe: {nfe:.1f}")
            else:
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
        batch = next(train_ds)
        batch = batch.to(device)

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

        # -------- EMA update --------
        emaf.update(modelf.parameters())
        emab.update(modelb.parameters())

        step += 1

        if torch.isnan(lossf + lossb).any():
            log.warning("Loss is nan")
            return False

        if step % 10 == 0:
            logger.log_metrics({"train/loss_f": lossf.item()}, step)
            logger.log_metrics({"train/loss_b": lossb.item()}, step)
            tbar.set_description(f"F: {lossf:.2f} | B: {lossb:.2f}")

        if step % val_freq == 0:
            logger.log_metrics(
                {"train/time_per_it": (timer() - train_time) / val_freq}, step
            )
            total_train_time += timer() - train_time
            eval_time = timer()

            if train_val:
                logp = evaluate("val", step)
                logger.log_metrics({"val/time_per_it": (timer() - eval_time)}, step)
                logger.log_metrics({"logp": logp}, step)

                gc.collect()

            train_time = timer()

    logger.log_metrics({"train/total_time": total_train_time}, step)
    return True

if __name__ == "__main__":
    success = train(step=0)

    #  MODIFICAÇÃO 5: Salvar o modelo ao final do treinamento num .pkl
    
    if success:
        print("Training finished successfully. Saving model weights...")
        
        
        emaf.copy_to(modelf.parameters())
        emab.copy_to(modelb.parameters())

       
        torch.save({
            'model_f_state_dict': modelf.state_dict(),
            'model_b_state_dict': modelb.state_dict(),
        }, 'trained_protein_model.pkl') 

        print("Model weights saved to 'trained_protein_model.pkl'")
    
    
    logger.save()
    logger.finalize("success" if success else "failure")