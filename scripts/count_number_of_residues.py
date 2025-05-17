import pandas as pd

from rna_motif_library.util import get_pdb_ids, run_w_processes
from rna_motif_library.residue import get_cached_residues


def process_pdb(pdb_id):
    rna_ids = ["A", "C", "G", "U"]
    residues = get_cached_residues(pdb_id)
    count = 0
    for key, res in residues.items():
        if res.res_id in rna_ids:
            count += 1
    return {"pdb_id": pdb_id, "count": count}


def main():
    df = pd.read_csv("data/csvs/rna_structures.csv")
    pdb_ids = df["pdb_id"].unique().tolist()

    # Run with multiple processes
    results = run_w_processes(process_pdb, pdb_ids, processes=15)
    # Create dataframe from results
    df_results = pd.DataFrame(results)
    df_results.to_csv("data/csvs/rna_residue_counts.csv", index=False)


if __name__ == "__main__":
    main()
