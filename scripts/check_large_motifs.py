import click
import pandas as pd

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.motif_factory import (
    HelixFinder,
    get_pdb_structure_data,
    get_pdb_structure_data_for_residues,
    get_cww_basepairs,
)


def find_pdbs_with_large_motifs():
    df = pd.read_csv("large_motifs.csv")
    df_res_counts = pd.read_csv("data/csvs/rna_residue_counts.csv")
    df_motif_counts = pd.read_csv("motif_counts.csv")
    # Find the smallest structure by RNA residue count
    # Create a list of (pdb_id, residue_count) tuples
    pdb_res_counts = []
    for pdb_id, g in df.groupby("pdb_id"):

        if len(g) < 2:
            continue
        res_count = df_res_counts[df_res_counts["pdb_id"] == pdb_id]["count"].values[0]
        motif_count = df_motif_counts[df_motif_counts["pdb_id"] == pdb_id][
            "count"
        ].values[0]
        pdb_res_counts.append((pdb_id, res_count, motif_count))

    # Sort by residue count
    pdb_res_counts.sort(key=lambda x: x[1])

    # Print the sorted list
    for pdb_id, res_count, motif_count in pdb_res_counts:
        print(f"{pdb_id}: {res_count} RNA residues, {motif_count} motifs")


def main():
    df = pd.read_csv("large_motifs.csv")
    df["num_helices"] = 0
    count = 0
    for pdb_id, g in df.groupby("pdb_id"):
        if pdb_id != "7C79":
            continue
        motifs = get_cached_motifs(pdb_id)
        pdb_data = get_pdb_structure_data(pdb_id)
        cww_basepairs = get_cww_basepairs(
            pdb_data, min_two_hbond_score=0.5, min_three_hbond_score=0.5
        )
        motif_by_name = {m.name: m for m in motifs}
        for i, row in g.iterrows():
            m = motif_by_name[row["motif_name"]]
            m.to_cif()
            res = m.get_residues()
            pdb_data_for_residues = get_pdb_structure_data_for_residues(pdb_data, res)
            hf = HelixFinder(pdb_data_for_residues, cww_basepairs, [])
            helices = hf.get_helices()
            df.loc[i, "num_helices"] = len(helices)
            if len(helices) > 1:
                print(pdb_id, row["motif_name"], len(helices), count)
                count += 1


if __name__ == "__main__":
    main()
