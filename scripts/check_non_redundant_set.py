import pandas as pd
import os

from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import get_non_redundant_sets, get_cached_path


def main():
    name = "nrlist_3.369_3.5A.csv"
    path = os.path.join(DATA_PATH, "csvs", name)
    sets = get_non_redundant_sets(path)
    not_found = []
    for pdb_set in sets.values():
        for pdb_id in pdb_set:
            cached_path = get_cached_path(pdb_id, "motifs")
            if not os.path.exists(cached_path) and pdb_id not in not_found:
                not_found.append(pdb_id)
    f = open("not_found.csv", "w")
    f.write("pdb_id\n")
    for pdb_id in not_found:
        f.write(f"{pdb_id}\n")
    f.close()


if __name__ == "__main__":
    main()
