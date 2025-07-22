import click
import pandas as pd
import os

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.motif_factory import (
    HelixFinder,
    get_pdb_structure_data,
    get_pdb_structure_data_for_residues,
    get_cww_basepairs,
)
from rna_motif_library.util import add_motif_indentifier_columns


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
    df = pd.read_json("data/summaries/motifs/non_redundant_motifs_summary.json")
    df["is_large_motif"] = (
        (df["num_residues"] > 50)
        & (df["motif_type"] != "NWAY")
        & (df["motif_type"] != "HELIX")
    )
    df = df[df["is_large_motif"]]
    df_copy = df.copy()
    df_copy["count"] = [i for i in range(len(df_copy))]
    df_copy = df_copy[["count", "pdb_id", "motif_id"]]
    df_copy["exclude"] = 0
    df_copy.to_csv("large_motifs.csv", index=False)
    count = 0
    os.makedirs("large_motifs", exist_ok=True)
    all_motifs = {}
    df = add_motif_indentifier_columns(df, "motif_id")
    for i, row in df.iterrows():
        if row["pdb_id"] not in all_motifs:
            all_motifs[row["pdb_id"]] = {
                m.name: m for m in get_cached_motifs(row["pdb_id"])
            }
        motif = all_motifs[row["pdb_id"]][row["motif_id"]]
        motif.to_cif(f"large_motifs/{count}.cif")
        count += 1


if __name__ == "__main__":
    main()
