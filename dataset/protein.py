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

 #protein.py (substitua a classe CATHDataset por esta versão)

class CATHDataset(TensorDataset):
    """
    Classe Dataset para o CATH, refatorada para seguir o padrão das classes
    Top500 e RNA. Todo o pré-processamento é feito uma vez no __init__
    para máxima eficiência e robustez.
    """
    def __init__(self, tsv_path, window_size):
        
        # 1. Movendo toda a lógica para __init__
        # A dimensão do toro é um hiperparâmetro definido pela 'window_size'
        num_angles_per_residue = 2
        self.torus_dim = window_size * num_angles_per_residue
        self.manifold = Torus(dim=self.torus_dim)

        # 2. Carrega todos os ângulos do arquivo para um tensor PyTorch
        print(f"Lendo dados de {tsv_path}...")
        all_angles = pd.read_csv(tsv_path, sep='\t', header=None).values.astype(np.float32)
        angles_tensor = torch.from_numpy(all_angles)

        print(f"Processando {len(angles_tensor)} amostras para um Torus de dimensão {self.torus_dim}...")

        # 3. A Conversão Vetorizada (A forma mais eficiente e "PyTorch-ic")
        # Esta operação é feita UMA VEZ para o dataset inteiro.
        # 'angles_tensor' tem shape [num_amostras, torus_dim]
        
        # Calcula todos os cossenos e senos de uma vez
        cos_angles = torch.cos(angles_tensor) # Shape: [N, torus_dim]
        sin_angles = torch.sin(angles_tensor) # Shape: [N, torus_dim]

        # Empilha para criar pares [cos, sin] para cada ângulo.
        # A dim=2 cria um shape [N, torus_dim, 2]
        coords = torch.stack([cos_angles, sin_angles], dim=2)

        # Achata a partir da dimensão 1 para obter a ordem intercalada correta.
        # [N, torus_dim, 2] -> [N, torus_dim * 2]
        # Resultando em [cos(a1), sin(a1), cos(a2), sin(a2), ...] para cada amostra.
        final_data = coords.flatten(start_dim=1)

        # 4. Chama o construtor da classe pai (TensorDataset) com os dados finais
        # self.data é criado automaticamente aqui.
        super().__init__(final_data)
        
        print(f"Dataset CATH carregado com sucesso. Tensor final com shape: {self.data.shape}")