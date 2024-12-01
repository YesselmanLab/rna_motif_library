"""
This test ensures we can parse all the X3DNA residue strings correctly and can regenerate them if necessary
"""

import glob
import os
import json

from rna_motif_library.classes import X3DNAResidueFactory

DATA_PATH = "data/dssr_output"


def get_all_nts():
    nts = set()
    json_files = glob.glob(os.path.join(DATA_PATH, "dssr_output/*.json"))
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        for nt in data["nts"]:
            nts.add(nt["nt_id"])
        print(len(nts))
    f = open("nts.txt", "w")
    for nt in nts:
        f.write(nt + "\n")
    f.close()


def main():
    nt_path = "intergration_tests/resources/nts.txt"
    if not os.path.exists(nt_path):
        get_all_nts()

    count = 0
    f = open(nt_path, "r")
    for line in f:
        res_str = line.rstrip()
        res = X3DNAResidueFactory.create_from_string(res_str)
        if res.get_str() != res_str:
            print(res_str, res.get_str())
            count += 1
    assert count == 0
    f.close()


if __name__ == "__main__":
    main()
