# preprocess_cath.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB import PPBuilder
import warnings

# --- PARÂMETROS DE CONFIGURAÇÃO ---
# Altere estes valores para seus experimentos

# 1. Diretório onde seus arquivos .pdb estão localizados
PDB_DIRECTORY = "./dompdb"

# 2. Caminho para o arquivo de saída. É bom usar um nome descritivo.
#    Exemplo: "cath, comprimentos 15-30, janela de 10"
OUTPUT_FILE = "./data/cath_s40_L15-30_W10.tsv" 

# 3. O tamanho da janela deslizante para criar os segmentos de ângulos
WINDOW_SIZE = 3

# 4. A faixa de comprimentos de cadeia a serem incluídas no dataset.
#    Isto é crucial para criar um dataset homogêneo e estável.
#    Coloque ambos como None para desativar o filtro e processar tudo.
MIN_LENGTH = 3
MAX_LENGTH = 15
# ----------------------------------

# --- INÍCIO DO SCRIPT ---

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

parser = PDBParser(QUIET=True)
ppb = PPBuilder()

all_torsion_segments = []
rejected_files_count = 0

try:
    pdb_files = [f for f in os.listdir(PDB_DIRECTORY) if f.endswith('.pdb') or '.' not in f]
except FileNotFoundError:
    print(f"ERRO: O diretório '{PDB_DIRECTORY}' não foi encontrado. Verifique o caminho e se os dados foram descompactados corretamente.")
    exit()

print(f"Encontrados {len(pdb_files)} arquivos PDB. Processando...")
if MIN_LENGTH is not None and MAX_LENGTH is not None:
    print(f"MODO FILTRO ATIVADO: Processando apenas cadeias com comprimento entre {MIN_LENGTH} e {MAX_LENGTH} resíduos.")

for filename in tqdm(pdb_files):
    filepath = os.path.join(PDB_DIRECTORY, filename)
    
    # Lista temporária para os segmentos deste arquivo específico
    segments_from_this_file = []
    file_is_problematic = False

    try:
        structure = parser.get_structure("protein", filepath)
        
        # Bloco para capturar RuntimeWarnings e tratar arquivos corrompidos
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                for model in structure:
                    for chain in model:
                        for polypeptide in ppb.build_peptides(chain):
                            # Extrai a lista de ângulos (phi, psi) em radianos
                            phi_psi_list = polypeptide.get_phi_psi_list()
                            
                            # Filtra resíduos sem ângulos (geralmente o primeiro e último)
                            angles_rad = [(phi, psi) for phi, psi in phi_psi_list if phi is not None and psi is not None]
                            
                            # LÓGICA DO FILTRO POR FAIXA DE COMPRIMENTO
                            current_length = len(angles_rad)
                            if MIN_LENGTH is not None and current_length < MIN_LENGTH:
                                continue # Pula se a cadeia for muito curta
                            if MAX_LENGTH is not None and current_length > MAX_LENGTH:
                                continue # Pula se a cadeia for muito longa

                            # LÓGICA DA JANELA DESLIZANTE
                            if len(angles_rad) >= WINDOW_SIZE:
                                for i in range(len(angles_rad) - WINDOW_SIZE + 1):
                                    segment = angles_rad[i : i + WINDOW_SIZE]
                                    flat_segment = np.array(segment, dtype=np.float32).flatten()
                                    
                                    # Verificação final de NaN (boa prática)
                                    if not np.isnan(flat_segment).any():
                                        segments_from_this_file.append(flat_segment)
            
            # Se um RuntimeWarning foi capturado, marca o arquivo para descarte
            except RuntimeWarning:
                file_is_problematic = True
                rejected_files_count += 1

        # Apenas adicione os segmentos se o arquivo foi processado sem problemas
        if not file_is_problematic:
            all_torsion_segments.extend(segments_from_this_file)

    except Exception as e:
        # Captura outros erros (ex: erro de parsing do arquivo)
        rejected_files_count += 1


print("\n--- Processamento Concluído ---")
if rejected_files_count > 0:
    print(f"{rejected_files_count} arquivos PDB problemáticos ou fora da faixa foram ignorados.")

if not all_torsion_segments:
    print("Nenhum segmento foi gerado. Verifique seus parâmetros de filtro e arquivos PDB.")
else:
    print(f"Criando arquivo de saída...")
    df = pd.DataFrame(all_torsion_segments)
    df.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)
    print(f"\nSucesso! {len(all_torsion_segments)} segmentos de {WINDOW_SIZE} resíduos foram salvos em: {OUTPUT_FILE}")