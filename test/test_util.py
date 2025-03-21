import os
import pandas as pd

from rna_motif_library.util import ResidueTypeAssigner, get_non_redundant_sets
from rna_motif_library.settings import DATA_PATH


def test_residue_type_assigner():
    rta = ResidueTypeAssigner()
    assert rta.get_residue_type("A-A-1-", "XXXX") == "RNA"
    assert rta.get_residue_type("A-C-1-", "XXXX") == "RNA"
    assert rta.get_residue_type("A-G-1-", "XXXX") == "RNA"
    assert rta.get_residue_type("A-U-1-", "XXXX") == "RNA"
    assert rta.get_residue_type("A-ALA-1-", "XXXX") == "PROTEIN"
    assert rta.get_residue_type("A-DA-1-", "XXXX") == "DNA"
    assert rta.get_residue_type("A-0DG-1-", "5VY7") == "OTHER-POLYMER"
    assert rta.get_residue_type("A-0DG-1-", "5VY7") == "OTHER-POLYMER"
    assert rta.get_residue_type("A-0DG-2-", "5VY7") == "NON-CANONICAL NA"
    assert rta.get_residue_type("A-UNK-2-", "5VY7") == "UNKNOWN"


def test_get_non_redundant_sets():
    sets = get_non_redundant_sets(
        os.path.join(DATA_PATH, "csvs", "nrlist_3.369_3.5A.csv")
    )
    for k, v in sets.items():
        if len(v) < 5:
            continue
        print(k, v)
