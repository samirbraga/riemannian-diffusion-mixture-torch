import os
import numpy as np
from main_angles import extract_torsions

INPUT = "./data/dompdb/"
OUTPUT = "./data/angles/"

os.makedirs(OUTPUT, exist_ok=True)

files = sorted(os.listdir(INPUT))

for f in files:
    if not f.endswith(".pdb"):
        continue
    pdb_path = os.path.join(INPUT, f)
    tors = extract_torsions(pdb_path)
    if tors is None:
        continue

    out_path = os.path.join(OUTPUT, f.replace(".pdb", ".npy"))
    np.save(out_path, tors)

print("DONE: all PDBs converted to torsion .npy files")
