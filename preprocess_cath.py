# preprocess_cath.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB import PPBuilder


PDB_DIRECTORY = "./dompdb"

OUTPUT_FILE = "./data/cath_s40_L64.tsv"


WINDOW_SIZE = 50


#INÍCIO DO SCRIPT 

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


parser = PDBParser(QUIET=True)
ppb = PPBuilder()

all_torsion_segments = []



try:
    pdb_files = [f for f in os.listdir(PDB_DIRECTORY) if f.endswith('.pdb') or '.' not in f]
except FileNotFoundError:
    print(f"ERRO: O diretório '{PDB_DIRECTORY}' não foi encontrado. Verifique o caminho e se os dados foram descompactados corretamente.")
    exit()

print(f"Encontrados {len(pdb_files)} arquivos PDB. Processando...")

#
for filename in tqdm(pdb_files):
    filepath = os.path.join(PDB_DIRECTORY, filename)
    try:
        
        structure = parser.get_structure("protein", filepath)

        
        for model in structure:
            
            for chain in model:
                
                for polypeptide in ppb.build_peptides(chain):
                    
                   
                    phi_psi_list = polypeptide.get_phi_psi_list()

                    
                    
                    angles_rad = [
                        (phi, psi) for phi, psi in phi_psi_list if phi is not None and psi is not None
                    ]

                   
                    if len(angles_rad) >= WINDOW_SIZE:
                        
                        for i in range(len(angles_rad) - WINDOW_SIZE + 1):
                            
                            segment = angles_rad[i : i + WINDOW_SIZE]
                            
                            
                            flat_segment = np.array(segment, dtype=np.float32).flatten()
                            
                           
                            if not np.isnan(flat_segment).any():
                                all_torsion_segments.append(flat_segment)

    except Exception as e:
        
        print(f"\nAviso: erro ao processar o arquivo {filename}: {e}. Pulando.")


print(f"\nCriando arquivo.")
df = pd.DataFrame(all_torsion_segments)


df.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)

print(f"\nProcessamento concluído!")
print(f"{len(all_torsion_segments)} segmentos de {WINDOW_SIZE} resíduos foram salvos em {OUTPUT_FILE}")