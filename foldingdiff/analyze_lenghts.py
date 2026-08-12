# analyze_lengths.py (VERSÃO ROBUSTA E SILENCIOSA)

import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
from Bio.PDB import PDBParser
from Bio.PDB import PPBuilder
import warnings

PDB_DIRECTORY = "./dompdb"

print("Iniciando a análise dos comprimentos das cadeias polipeptídicas...")

parser = PDBParser(QUIET=True)
ppb = PPBuilder()

all_lengths = []
rejected_files_count = 0

try:
    pdb_files = [f for f in os.listdir(PDB_DIRECTORY) if f.endswith('.pdb') or '.' not in f]
except FileNotFoundError:
    print(f"ERRO: O diretório '{PDB_DIRECTORY}' não foi encontrado.")
    exit()

print(f"Analisando {len(pdb_files)} arquivos PDB...")

for filename in tqdm(pdb_files):
    filepath = os.path.join(PDB_DIRECTORY, filename)
    file_is_problematic = False

    try:
        structure = parser.get_structure("protein", filepath)
        
        # --- BLOCO PARA IGNORAR WARNINGS ---
        # Vamos tratar os RuntimeWarnings como erros dentro deste bloco para poder capturá-los
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                for model in structure:
                    for chain in model:
                        for polypeptide in ppb.build_peptides(chain):
                            phi_psi_list = polypeptide.get_phi_psi_list()
                            
                            valid_residues = [
                                (phi, psi) for phi, psi in phi_psi_list if phi is not None and psi is not None
                            ]
                            
                            if valid_residues:
                                all_lengths.append(len(valid_residues))

            # Se um RuntimeWarning foi capturado no bloco 'try' acima, este 'except' será acionado
            except RuntimeWarning:
                file_is_problematic = True
                rejected_files_count += 1
        # ------------------------------------

    except Exception as e:
        # Captura outros erros (ex: erro de parsing do arquivo PDB)
        rejected_files_count += 1


# Usando collections.Counter para encontrar os comprimentos mais comuns
if not all_lengths:
    print("\nNenhuma cadeia polipeptídica válida foi encontrada. Verifique seus arquivos PDB.")
else:
    length_counts = Counter(all_lengths)
    print("\n--- Análise de Comprimento Concluída ---")
    if rejected_files_count > 0:
        print(f"Aviso: {rejected_files_count} arquivos PDB problemáticos ou corrompidos foram ignorados durante a análise.")
        
    print("\nOs 10 comprimentos de cadeia mais comuns (baseados em dados limpos) são:")
    for length, count in length_counts.most_common(10):
        print(f"Comprimento: {length} resíduos  | Ocorrências: {count}")
        
    # Salvar todos os comprimentos para uma análise mais detalhada (opcional)
    os.makedirs("./data", exist_ok=True) # Garante que o diretório 'data' exista
    pd.DataFrame(all_lengths, columns=['length']).to_csv('./data/chain_lengths.csv', index=False)
    print("\nUm arquivo detalhado com todos os comprimentos foi salvo em ./data/chain_lengths.csv")