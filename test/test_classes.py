import pandas as pd
import os
import json
from typing import Dict

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.classes import (
    ResidueNew,
    sanitize_x3dna_atom_name,
    X3DNAResidueFactory,
    get_x3dna_res_id,
)


def get_residues_from_json(pdb_code: str) -> Dict[str, ResidueNew]:
    df_atoms = pd.read_parquet(
        os.path.join(DATA_PATH, "pdbs_dfs", f"{pdb_code}.parquet")
    )
    residues = {}
    for i, g in df_atoms.groupby(
        ["auth_asym_id", "auth_seq_id", "auth_comp_id", "pdbx_PDB_ins_code"]
    ):
        coords = g[["Cartn_x", "Cartn_y", "Cartn_z"]].values
        atom_names = g["auth_atom_id"].tolist()
        atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
        chain_id, res_num, res_name, ins_code = i
        x3dna_res_id = get_x3dna_res_id(chain_id, res_num, res_name, ins_code)
        x3dna_res = X3DNAResidueFactory.create_from_string(x3dna_res_id)
        residues[x3dna_res_id] = ResidueNew.from_x3dna_residue(
            x3dna_res, atom_names, coords
        )
    return residues


def test_residue_json():
    residues = get_residues_from_json("1A9N")
    key = list(residues.keys())[0]
    res = residues[key]
    res_json = res.to_dict()
    new_res = ResidueNew.from_dict(res_json)
    assert res == new_res
