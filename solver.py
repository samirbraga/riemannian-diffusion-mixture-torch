
import abc
import torch
import numpy as np
from tqdm import trange



class Predictor(abc.ABC):
    def __init__(self, sde):
        super().__init__()
        self.sde = sde
        self.manifold = sde.manifold
    @abc.abstractmethod
    def update_fn(self, x, t, dt):
        raise NotImplementedError()


class Corrector(abc.ABC):
    def __init__(self, sde, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps
    @abc.abstractmethod
    def update_fn(self, x0, x, t):
        raise NotImplementedError()


class EulerMaruyamaTwoWayPredictor:
    
    def __init__(self, mix):
        self.mix = mix
        self.manifold = mix.manifold

    def update_fn(self, x, t, dt, x0, xf, t_mask):
        shape = x.shape
        z = self.manifold.random_normal_tangent(base_point=x, n_samples=shape[0])
        
        fdrift, fdiff = self.mix.bridge(xf).coefficients(x, t)
        bdrift, bdiff = self.mix.rev().bridge(x0).coefficients(x, self.mix.tf - t)

        if fdiff.ndim > 1: fdiff = torch.diag(fdiff)
        if bdiff.ndim > 1: bdiff = torch.diag(bdiff)

        drift = torch.where(t_mask, fdrift, bdrift)
        diffusion = torch.where(t_mask.squeeze(), fdiff, bdiff)

        dt_reshaped = dt.view(-1, 1, 1)
        
        tangent_vector = drift * dt_reshaped
        
        #  MEMORY ALLOCATION FIX FOR THE BACKWARD PASS
        # Break down the operation to prevent memory overflow
        dt_sqrt = torch.sqrt(torch.abs(dt_reshaped))
        combined_diffusion = diffusion.view(-1, 1, 1) * dt_sqrt
        tangent_vector += combined_diffusion * z

        x = self.manifold.exp(tangent_vec=tangent_vector, base_point=x)
        return x, x


def get_twoway_sampler(mix, N=10):
    
    predictor = EulerMaruyamaTwoWayPredictor(mix)
    
    def sampler(x0, xf, t):
        with torch.no_grad():
            
            t_mask = (t < 0.5).view(-1, 1, 1)
            #  correct batch dimension for the loop
            x = torch.where(t_mask, x0, xf)
            
            # Squeeze t_mask for time-related calculations
            integration_time = torch.where(t_mask.squeeze(), t, 1.0 - t)
            dt = (integration_time - mix.t0) / N
            timesteps = torch.linspace(0, 1, N, device=x0.device) * integration_time.unsqueeze(1)

            for i in range(N):
                current_t = timesteps[:, i]
                # Pass the correctly shaped t_mask to the update function
                x, x_mean = predictor.update_fn(x, current_t, dt, x0, xf, t_mask)
                
        return x_mean
    return sampler