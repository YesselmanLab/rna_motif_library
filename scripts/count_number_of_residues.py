import json
import os
import pandas as pd

from rna_motif_library.classes import get_residues_from_json
from rna_motif_library.settings import DATA_PATH
from rna_motif_library.util import get_pdb_codes


def main():
    pdb_codes = get_pdb_codes()
    all_data = []
    rna_ids = ["A", "C", "G", "U"]
    for pdb_code in pdb_codes:
        json_path = os.path.join(DATA_PATH, "jsons", "residues", f"{pdb_code}.json")
        residues = get_residues_from_json(json_path)
        count = 0
        for key, res in residues.items():
            if res.res_id in rna_ids:
                count += 1
        all_data.append({"pdb_code": pdb_code, "count": count})
        print(pdb_code, count)
    df = pd.DataFrame(all_data)
    df.to_csv("rna_residue_counts.csv", index=False)


if __name__ == "__main__":
    main()
