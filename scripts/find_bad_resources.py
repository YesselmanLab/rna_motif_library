import pandas as pd
import glob
import os
from rna_motif_library.settings import DATA_PATH


def main():
    pdb_dir = os.path.join(DATA_PATH, "pdbs")
    for pdb_file in glob.glob(os.path.join(pdb_dir, "*.cif")):
        basename = os.path.basename(pdb_file)
        if len(basename) != 8:
            print(basename)
        pdb_id = os.path.basename(pdb_file).split(".")[0]
        if len(pdb_id) != 4:
            print(pdb_id)

    dssr_dir = os.path.join(DATA_PATH, "dssr_output")
    for dssr_file in glob.glob(os.path.join(dssr_dir, "*.json")):
        basename = os.path.basename(dssr_file)
        if len(basename) != 9:
            print(basename)
        pdb_id = os.path.basename(dssr_file).split(".")[0]
        if len(pdb_id) != 4:
            print(pdb_id)

    snap_dir = os.path.join(DATA_PATH, "snap_output")
    for snap_file in glob.glob(os.path.join(snap_dir, "*.json")):
        basename = os.path.basename(snap_file)
        if len(basename) != 9:
            print(basename)
        pdb_id = os.path.basename(snap_file).split(".")[0]
        if len(pdb_id) != 4:
            print(pdb_id)


if __name__ == "__main__":
    main()
