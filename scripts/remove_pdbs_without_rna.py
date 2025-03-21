import os
import pandas as pd

from rna_motif_library.settings import DATA_PATH

def remove_files(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


# The problem is apparently you can only request structures with nucleic acid residues
# thus there are tons that do not include RNA
def main():
    df = pd.read_csv("data/csvs/rna_residue_counts.csv")
    df = df[df["count"] < 2]
    for _, row in df.iterrows():
        print(row["pdb_id"])
        pdb_id = row["pdb_id"]
        remove_files(os.path.join(DATA_PATH, "pdbs", f"{pdb_id}.cif"))
        remove_files(os.path.join(DATA_PATH, "pdbs_dfs", f"{pdb_id}.parquet"))
        remove_files(os.path.join(DATA_PATH, "dssr_output", f"{pdb_id}.json"))
        remove_files(os.path.join(DATA_PATH, "snap_output", f"{pdb_id}.out"))
        remove_files(os.path.join(DATA_PATH, "jsons", "hbonds", f"{pdb_id}.json"))
        remove_files(os.path.join(DATA_PATH, "jsons", "motifs", f"{pdb_id}.json"))
        remove_files(os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_id}.json"))
        remove_files(os.path.join(DATA_PATH, "jsons", "chains", f"{pdb_id}.json"))
        remove_files(os.path.join(DATA_PATH, "jsons", "protein_chains", f"{pdb_id}.json"))
        remove_files(os.path.join(DATA_PATH, "dataframes", "basepairs", f"{pdb_id}.json"))


if __name__ == "__main__":
    main()
