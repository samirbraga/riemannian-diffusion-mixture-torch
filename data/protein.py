# protein.py (versão modificada)

import torch
import numpy as np
import pandas as pd

from data.tensordataset import TensorDataset
from geomstats.geometry.torus import Torus


class Top500(TensorDataset):
    """Classe original para o dataset Top500 (sem alterações)."""
    def __init__(self, data_dir="data/top500", amino="General"):
        self.manifold = Torus(2)
        data = pd.read_csv(
            f"{data_dir}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro"]
        assert amino in amino_types, f"amino type {amino} not implemented"

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")

        data = torch.tensor(data % 360 * np.pi / 180)
        #NOTA: coordenadas em vez de ângulos
        self.data = torch.stack([torch.cos(data[:,0]), torch.sin(data[:,0]), 
                        torch.cos(data[:,1]), torch.sin(data[:,1])], dim=1)
        self.amino = amino

        expand_factors = {'General': 1, 'Glycine': 10, 'Proline': 18, 'Pre-Pro': 20}
        self.expand_factor = expand_factors[amino]

        super().__init__(self.data)


class RNA(TensorDataset):
    """Classe original para o dataset RNA (sem alterações)."""
    def __init__(self, data_dir="data/rna"):
        self.manifold = Torus(7)
        data = pd.read_csv(
            f"{data_dir}/aggregated_angles.tsv",
            delimiter="\t",
            names=[
                "source",
                "base",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "chi",
            ],
        )

        data = data[
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        ].values.astype("float32")

        data = torch.tensor(data % 360 * np.pi / 180)
        #NOTA: coordenadas em vez de ângulos
        coords = []
        for i in range(data.shape[1]):
            coords.extend([torch.cos(data[:,i]), torch.sin(data[:,i])])
        self.data = torch.stack(coords, dim=1)
        
        self.expand_factor = 14

        super().__init__(self.data)


# protein.py (versão 2.0 com CATHDataset)

class CATHDataset(torch.utils.data.Dataset):
    """
    Classe Dataset personalizada para carregar os dados pré-processados do CATH.
    Lê o arquivo .tsv gerado e o prepara para o PyTorch, construindo corretamente
    os pontos na variedade Torus.
    """
    def __init__(self, tsv_path, window_size):
        
        self.data = pd.read_csv(tsv_path, sep='\t', header=None).values.astype(np.float32)
        
        
        num_angles_per_residue = 2
        self.torus_dim = window_size * num_angles_per_residue
        
        #
        self.manifold = Torus(dim=self.torus_dim)
        
        print(f"Dataset CATH carregado com {len(self.data)} amostras.")
        print(f"Dimensão do Torus (manifold): {self.torus_dim}")

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
       
        # 'angles' é um vetor achatado de n ângulos (φ1, ψ1, φ2, ψ2, ...)
        angles = torch.from_numpy(self.data[idx])
        
        # correção/tentativa de correção
        
       
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles) 
        
        
        coords = torch.stack([cos_angles, sin_angles], dim=1)
        
        
        point_on_torus = coords.flatten()

       
        
        return point_on_torus