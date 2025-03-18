import os
import shutil
import pandas as pd
from rna_motif_library.motif import get_cached_motifs, get_cached_path


def main():
    df = pd.read_csv("/Users/jyesselman2/Documents/new_pdb_list.txt")
    for i, row in df.iterrows():
        pdb_id = row["pdb_id"]
        path = "data/dataframes/hbonds/"+pdb_id+".csv"
        if not os.path.exists(path):
            continue
        shutil.copy(path, f"new_hbonds/{pdb_id}.csv")
        #path = get_cached_path(pdb_id, "motifs")
        #if not os.path.exists(path):
        #    continue
        #shutil.copy(path, f"new_motifs/{pdb_id}.json")


if __name__ == "__main__":
    main()
