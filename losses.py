import torch
import numpy as np
from sde_lib import DiffusionMixture
from solver import get_twoway_sampler

def get_mix_loss_fn(mix: DiffusionMixture, reduce_mean=False, eps=1e-5, num_steps=10, 
                    weight_type='default', sampler_type='twoway', loss_type='smooth_l1', beta=0.1):

    reduce_op = torch.mean if reduce_mean else \
                lambda *args, **kwargs: torch.sum(*args, **kwargs)
    sampler = get_twoway_sampler(mix, num_steps)
    Z = mix.importance_cum_weight(mix.tf-eps, eps)

    def weight_fn(t):
        if weight_type=='default' :
            weight = 1./mix.beta_schedule.beta_t(t)
        elif 'const' in weight_type:
            weight = float(weight_type.split('_')[-1]) * torch.ones_like(t)
        elif weight_type=='importance':
            weight = torch.ones_like(t) * Z
        else:
            raise NotImplementedError(f'{weight_type} not implemented.')
        return weight

    # HELPER FUNCTION FOR MANIFOLD SMOOTH L1 LOSS
    def manifold_smooth_l1_loss(v, x, beta: float):
        """
        Calculates the smooth L1 loss for a tangent vector v at base point x.
        """
        norm_sq = mix.manifold.metric.squared_norm(v, x)
        norm = torch.sqrt(norm_sq + 1e-8)

        smooth_l1 = torch.where(
            norm < beta,
            0.5 * norm_sq / beta,
            norm - 0.5 * beta
        )
        return smooth_l1

    def loss_fn(modelf, modelb, x):
        shape = x.shape
        predf_fn = mix.get_drift_fn(modelf, train=True)
        predb_fn = mix.get_drift_fn(modelb, train=True)

        if 'importance' in weight_type:
            t = mix.sample_importance_weighted_time((x.shape[0],), eps, x.device)
        else:
            t = torch.rand(x.shape[0], device=x.device) * (mix.tf - eps) + eps

        x0 = mix.prior.sample(shape, x.device)

        if sampler_type == 'twoway':
            xt = sampler(x0, x, t)
        else:
            raise NotImplementedError(f'Sampler type: {sampler_type} not implemented.')

        weight = weight_fn(t)

        # FORWARD MODEL LOSS
        lossesf_vec = predf_fn(xt, t) - mix.bridge(x).drift(xt, t)
        
        if loss_type == 'smooth_l1':
            lossesf = manifold_smooth_l1_loss(lossesf_vec, xt, beta)
        elif loss_type == 'l2':
            lossesf = 0.5 * mix.manifold.metric.squared_norm(lossesf_vec, xt)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        lossesf = weight * lossesf
        lossesf = reduce_op(lossesf, dim=-1)

        # BACKWARD MODEL LOSS
        lossesb_vec = predb_fn(xt, mix.tf-t) - mix.rev().bridge(x0).drift(xt, mix.tf-t)
        
        if loss_type == 'smooth_l1':
            lossesb = manifold_smooth_l1_loss(lossesb_vec, xt, beta)
        elif loss_type == 'l2':
            lossesb = 0.5 * mix.manifold.metric.squared_norm(lossesb_vec, xt)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        lossesb = weight * lossesb
        lossesb = reduce_op(lossesb, dim=-1)

        lossf, lossb = torch.mean(lossesf), torch.mean(lossesb)

        return lossf+lossb, lossf, lossb
    return loss_fn


def get_loss_step_fn(loss_fn, clip_grad_norm=1.0, lr_sched=False):
    """Create a one-step training function.
    """

    def step_fn(state, batch):
        optimizerf, schedulerf, modelf, emaf = state.optimizerf, state.schedulerf, state.modelf, state.emaf
        optimizerb, schedulerb, modelb, emab = state.optimizerb, state.schedulerb, state.modelb, state.emab

        optimizerf.zero_grad()
        optimizerb.zero_grad()

        loss, lossf, lossb = loss_fn(modelf, modelb, batch)
        loss.backward()

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(modelf.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(modelb.parameters(), clip_grad_norm)

        optimizerf.step()
        optimizerb.step()

        if lr_sched:
            schedulerf.step()
            schedulerb.step()

        # -------- EMA update --------
        emaf.update(modelf.parameters())
        emab.update(modelb.parameters())

        step = state.step + 1
        new_train_state = state._replace(
            step=step,
            optimizerf=optimizerf,
            schedulerf=schedulerf,
            modelf=modelf,
            emaf=emaf,
            optimizerb=optimizerb,
            schedulerb=schedulerb,
            modelb=modelb,
            emab=emab,
        )

        return new_train_state, lossf, lossb
    return step_fn