import pandas as pd
import json
import os
import numpy as np

from rna_motif_library.classes import Residue, get_residues_from_json
from rna_motif_library.cli import get_pdb_codes
from rna_motif_library.interactions import get_hbonds_and_basepairs
from rna_motif_library.tranforms import (
    get_transformed_residue,
    create_nucleotide_frame,
    create_basepair_frame,
    align_basepair_to_identity,
)
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import get_cif_header_str


def get_non_canonical_basepairs():
    pdb_codes = get_pdb_codes()
    bp_data = []
    for pdb_code in pdb_codes:
        _, basepairs = get_hbonds_and_basepairs(pdb_code)
        for bp in basepairs:
            if bp.bp_type == "WC":
                continue
            res1, res2 = bp.res_1, bp.res_2
            if (
                res2.res_id < res1.res_id
            ):  # Swap if res2 comes before res1 alphabetically
                res1, res2 = res2, res1
            bp_data.append(
                {
                    "pdb_code": pdb_code,
                    "bp_name": bp.bp_name,
                    "bp_type": bp.bp_type,
                    "res_1": res1.get_str(),
                    "res_2": res2.get_str(),
                    "res_id_1": res1.res_id,
                    "res_id_2": res2.res_id,
                }
            )
    df = pd.DataFrame(bp_data)
    df.to_csv("non_canonical_basepairs.csv", index=False)


def get_canonical_basepairs():
    pdb_codes = get_pdb_codes()
    bp_data = []
    for pdb_code in pdb_codes:
        _, basepairs = get_hbonds_and_basepairs(pdb_code)
        for bp in basepairs:
            if bp.bp_type == "WC":
                continue
            res1, res2 = bp.res_1, bp.res_2
            if (
                res2.res_id < res1.res_id
            ):  # Swap if res2 comes before res1 alphabetically
                res1, res2 = res2, res1
            bp_id = f"{res1.res_id}{res2.res_id}"
            bp_data.append(
                {
                    "pdb_code": pdb_code,
                    "bp_name": bp.bp_name,
                    "bp_type": bp.bp_type,
                    "bp_id": bp_id,
                    "res_1": res1.get_str(),
                    "res_2": res2.get_str(),
                }
            )
    df = pd.DataFrame(bp_data)
    df.to_csv("canonical_basepairs.csv", index=False)


def basepair_to_cif(res1: Residue, res2: Residue, path: str):
    f = open(path, "w")
    f.write(get_cif_header_str())
    acount = 1
    for res in [res1, res2]:
        res_str, acount = res.to_cif_str(acount)
        f.write(res_str)
    f.close()


class ResidueManager:
    def __init__(self):
        self.residues = {}

    def get_residue(self, x3dna_res_code: str, pdb_code: str) -> Residue:
        if pdb_code not in self.residues:
            self.residues[pdb_code] = get_residues_from_json(
                os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_code}.json")
            )
        return self.residues[pdb_code][x3dna_res_code]


def main():
    get_canonical_basepairs()
    exit()
    rm = ResidueManager()
    df = pd.read_csv("non_canonical_basepairs.csv")
    res = ["A", "U", "G", "C"]
    df = df[df["res_id_1"].isin(res) & df["res_id_2"].isin(res)]
    df_sub = df.query("res_id_1 == 'A' and res_id_2 == 'A' and bp_name =='cSS'").copy()
    print(df_sub)
    count = 0
    for i, row in df_sub.iterrows():
        res1 = rm.get_residue(row["res_1"], row["pdb_code"])
        res2 = rm.get_residue(row["res_2"], row["pdb_code"])
        _, res1, res2 = align_basepair_to_identity(res1, res2)
        basepair_to_cif(res1, res2, f"bp_{count}.cif")
        count += 1
        if count > 10:
            break


if __name__ == "__main__":
    main()
