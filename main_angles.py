import os
import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

def extract_torsions(pdb_path):
    with open(pdb_path, "rt") as f:
        source = PDBFile.read(f)

    if source.get_model_count() > 1:
        return None

    source_struct = source.get_structure()[0]

    phi, psi, omega = struc.dihedral_backbone(source_struct)
    return {"phi": phi, "psi": psi, "omega": omega}

if __name__ == "__main__":
    pdb_path = "./data/dompdb/3tguW00"
    tors = extract_torsions(pdb_path)
    np.save("3tguW00.npy", tors)
    print(tors)
