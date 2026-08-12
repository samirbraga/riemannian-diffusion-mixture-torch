
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PerResidueTorus:
    """Torus geometry applied per residue."""
    
    class TorusMetric:
        """Defines the metric for the torus, which is the standard Euclidean metric."""
        def squared_norm(self, v, x):
            return torch.sum(v * v, dim=(-1, -2))

    def __init__(self, torus_dim, max_len):
        self.torus_dim = torus_dim
        self.max_len = max_len
        self.metric = self.TorusMetric()

    def exp(self, base_point, tangent_vec):
        """Computes the exponential map by taking a step and projecting."""
        moved_point = base_point + tangent_vec
        return self.projection(moved_point)

    def log(self, base_point, point):
        """Computes the logarithmic map."""
        ambient_vector = point - base_point
        tangent_vector = self.to_tangent(v=ambient_vector, x=base_point)
        return tangent_vector
        
    def projection(self, point):
        """
        Projects a point from the ambient space back onto the torus manifold.
        """
        reshaped_point = point.view(
            -1, self.max_len, self.torus_dim, 2
        )
        norm = torch.linalg.norm(reshaped_point, dim=-1, keepdim=True)
        projected_reshaped = reshaped_point / (norm + 1e-8)
        projected_point = projected_reshaped.view(
            -1, self.max_len, 2 * self.torus_dim
        )
        return projected_point

    def to_tangent(self, v, x):
        B, L, D = v.shape
        out = v.clone()
        for i in range(self.torus_dim):
            cs = x[..., 2*i:2*i+2]
            vv = v[..., 2*i:2*i+2]
            rad = (vv * cs).sum(dim=-1, keepdim=True)
            out[..., 2*i:2*i+2] = vv - rad * cs
        return out

    def random_uniform(self, n_samples, device='cpu'):
        """Generates uniform random samples on the per-residue torus."""
        rand_angles = (torch.rand(n_samples, self.max_len, self.torus_dim, device=device) * 2 - 1) * np.pi
        cos_sin = torch.cat([torch.cos(rand_angles), torch.sin(rand_angles)], dim=-1)
        return cos_sin

    def random_normal_tangent(self, base_point, n_samples):
        """Generates a random tangent vector at the base_point."""
        noise = torch.randn_like(base_point)
        tangent_noise = self.to_tangent(v=noise, x=base_point)
        return tangent_noise

class CATHFullSequenceDataset(Dataset):
    """Loads torsion sequences from npy files."""
    def __init__(self, folder="./data/angles/",
                 max_len=512,
                 angles=("phi","psi","omega")):
        self.folder = folder
        self.files = sorted([os.path.join(folder, f)
                             for f in os.listdir(folder)
                             if f.endswith(".npy")])
        self.max_len = max_len
        self.angles = angles
        self.torus_dim = len(angles)
        self.manifold = PerResidueTorus(self.torus_dim, self.max_len)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        entry = np.load(self.files[idx], allow_pickle=True).item()
        
        seq = np.stack([entry[a] for a in self.angles], axis=-1)
        
        # The raw angle data in 'seq' contains NaN values.
        
        seq = np.nan_to_num(seq, nan=0.0)
        
         #Convert the cleaned angles to the (cos, sin) representation for the model
        cos_sin = np.concatenate([np.cos(seq), np.sin(seq)], axis=-1)
        
        
        pad = self.max_len - L
        if pad > 0:
            padded = np.pad(cos_sin, ((0,pad),(0,0)), constant_values=0.0)
            mask = np.zeros(self.max_len); mask[:L] = 1
        else:
            padded = cos_sin[:self.max_len]
            mask = np.ones(self.max_len)
            
        return {
            "x": torch.tensor(padded, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "length": L,
        }