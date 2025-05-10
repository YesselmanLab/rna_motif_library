import os
import pandas as pd

from rna_motif_library.util import (
    ResidueTypeAssigner,
    NonRedundantSetParser,
    parse_motif_name,
)
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
    path = os.path.join(DATA_PATH, "csvs", "nrlist_3.262_3.5A.csv")
    p = NonRedundantSetParser()
    sets = p.parse(path)


def test_parse_motif_name():
    assert parse_motif_name("HAIRPIN-1-CGG-7PWO-1") == ("HAIRPIN", "1", "CGG", "7PWO")
    assert parse_motif_name("HELIX-3-GCC-GGC-7PWO-4") == (
        "HELIX",
        "3",
        "GCC-GGC",
        "7PWO",
    )
    assert parse_motif_name("TWOWAY-11-0-CUUUCUGCCAAAG-UG-9C6I-1") == (
        "TWOWAY",
        "11-0",
        "CUUUCUGCCAAAG-UG",
        "9C6I",
    )
    assert parse_motif_name(
        "NWAY-13-10-2-2-2-0-GCUCAACGGAUAAAA-UCAUAGUGAUCC-AGCA-UUUA-UUUG-CU-7OTC-1"
    ) == (
        "NWAY",
        "13-10-2-2-2-0",
        "GCUCAACGGAUAAAA-UCAUAGUGAUCC-AGCA-UUUA-UUUG-CU",
        "7OTC",
    )
