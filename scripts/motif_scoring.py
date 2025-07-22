import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.basepair import BasepairFactory


def get_basepairs_that_contain_residues(df, residue_ids):
    """Filter basepairs dataframe to only include rows where both residues are in the given list.

    Args:
        df: DataFrame containing basepair information with res_1 and res_2 columns
        residue_ids: List of residue identifiers to filter by

    Returns:
        DataFrame containing only rows where both res_1 and res_2 are in residue_ids
    """
    return df[(df["res_1"].isin(residue_ids)) & (df["res_2"].isin(residue_ids))]


def main():
    """
    main function for script
    """
    pdb_id = "6V3A"
    motifs = get_cached_motifs(pdb_id)
    motifs_by_name = {m.name: m for m in motifs}
    m_name = "HAIRPIN-72-UUGUGUAGGAUAGGUGGGAGGCUUUGAAGCUGGAACGCUAGUUCCAGUGGAGCCGUCCUUGAAAUACCACCCUG-6V3A-1"
    m = motifs_by_name[m_name]
    res_ids = [r.get_str() for r in m.get_residues()]
    df_bps = pd.read_json(
        os.path.join("data", "dataframes", "basepairs", f"{pdb_id}.json")
    )
    df_motif_bps = get_basepairs_that_contain_residues(df_bps, res_ids)
    # for i, row in df_motif_bps.iterrows():
    #    print(row["res_1"], row["res_2"], row["hbond_score"])
    base_centers = {}
    for r in m.get_residues():
        base_centers[r.get_str()] = np.array(r.get_base_atom_coords()).mean(axis=0)
    neighor_residue = {}
    for i, c in enumerate(m.strands):
        for j, res in enumerate(c):
            res_id = res.get_str()
            neighbors = []
            if j > 0:  # has previous residue
                neighbors.append(c[j - 1].get_str())
            if j < len(c) - 1:  # has next residue
                neighbors.append(c[j + 1].get_str())
            neighor_residue[res_id] = neighbors
    for i, r in enumerate(m.get_residues()):
        res_id = r.get_str()
        neighbors = neighor_residue[res_id]
        best_distance = 1000
        best_res_id = None
        for j, r_other in enumerate(m.get_residues()):
            if i >= j:
                continue
            r_other_id = r_other.get_str()
            if r_other_id in neighbors:
                continue
            dist = np.linalg.norm(base_centers[res_id] - base_centers[r_other_id])
            if dist < best_distance:
                best_distance = dist
                best_res_id = r_other_id
        print(res_id, best_res_id, best_distance)

    # plt.scatter(df_motif_bps["hbond_score"], df_motif_bps["stretch"])
    # plt.show()


if __name__ == "__main__":
    main()
