import pandas as pd
import glob
import json
import os
import numpy as np
from typing import Dict, List

from biopandas.pdb import PandasPdb

from rna_motif_library.classes import (
    Residue,
    get_residues_from_json,
    Basepair,
    X3DNAResidueFactory,
)
from rna_motif_library.interactions import get_basepairs_from_json
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import sanitize_x3dna_atom_name, get_x3dna_res_id


class ResidueManager:
    def __init__(self):
        self.residues = {}

    def get_residue(self, x3dna_res_code: str, pdb_code: str) -> Residue:
        if pdb_code not in self.residues:
            self.residues[pdb_code] = get_residues_from_json(
                os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_code}.json")
            )
        if x3dna_res_code not in self.residues[pdb_code]:
            return None
        else:
            return self.residues[pdb_code][x3dna_res_code]


class BasepairManager:
    def __init__(self):
        self.basepairs = {}

    def get_basepair(
        self, x3dna_res_1: str, x3dna_res_2: str, pdb_code: str
    ) -> Basepair:
        if pdb_code not in self.basepairs:
            self.basepairs[pdb_code] = get_basepairs_from_json(
                os.path.join(DATA_PATH, "jsons", "basepairs", f"{pdb_code}.json")
            )
        for bp in self.basepairs[pdb_code]:
            if bp.res_1.get_str() == x3dna_res_1 and bp.res_2.get_str() == x3dna_res_2:
                return bp
            if bp.res_1.get_str() == x3dna_res_2 and bp.res_2.get_str() == x3dna_res_1:
                return bp
        return None


def load_ideal_basepairs() -> Dict[str, List[Residue]]:
    pdbs = glob.glob(
        os.path.join("rna_motif_library", "resources", "ideal_basepairs", "*.pdb")
    )
    basepairs = {}
    for pdb in pdbs:
        ppdb = PandasPdb().read_pdb(pdb)
        df_atom = ppdb.df["ATOM"]
        residues = []
        for i, g in df_atom.groupby(
            ["chain_id", "residue_number", "residue_name", "insertion"]
        ):
            coords = g[["x_coord", "y_coord", "z_coord"]].values
            atom_names = g["atom_name"].tolist()
            atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
            chain_id, res_num, res_name, ins_code = i
            x3dna_res_id = get_x3dna_res_id(res_name, res_num, chain_id, ins_code)
            x3dna_res = X3DNAResidueFactory.create_from_string(x3dna_res_id)
            residues.append(Residue.from_x3dna_residue(x3dna_res, atom_names, coords))
        residues = sorted(residues, key=lambda x: x.res_id)
        bp_name = f"{residues[0].res_id}{residues[1].res_id}"
        basepairs[bp_name] = residues
    return basepairs


def load_ideal_bases():
    pdbs = glob.glob(
        os.path.join("rna_motif_library", "resources", "ideal_bases", "*.cif")
    )
    bases = {}
    for pdb in pdbs:
        ppdb = PandasPdb().read_pdb(pdb)
        df_atom = ppdb.df["ATOM"]
        row = df_atom.iloc[0]
        coords = df_atom[["x_coord", "y_coord", "z_coord"]].values
        atom_names = df_atom["atom_name"].tolist()
        atom_names = [sanitize_x3dna_atom_name(name) for name in atom_names]
        x3dna_res_id = get_x3dna_res_id(
            row["residue_name"],
            row["residue_number"],
            row["chain_id"],
            row["insertion"],
        )
        x3dna_res = X3DNAResidueFactory.create_from_string(x3dna_res_id)
        bases[row["residue_name"]] = Residue.from_x3dna_residue(
            x3dna_res, atom_names, coords
        )
    return bases
