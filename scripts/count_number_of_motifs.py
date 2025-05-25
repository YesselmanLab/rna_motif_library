import pandas as pd

from rna_motif_library.util import get_pdb_ids, run_w_processes
from rna_motif_library.motif import get_cached_motifs


def process_pdb(pdb_id):
    motifs = get_cached_motifs(pdb_id)
    return {"pdb_id": pdb_id, "count": len(motifs)}


def main():
    df = pd.read_csv("data/csvs/rna_structures.csv")
    pdb_ids = df["pdb_id"].unique().tolist()

    # Run with multiple processes
    results = run_w_processes(process_pdb, pdb_ids, processes=8)
    # Create dataframe from results
    df_results = pd.DataFrame(results)
    df_results.to_csv("motif_counts.csv", index=False)


if __name__ == "__main__":
    main()
