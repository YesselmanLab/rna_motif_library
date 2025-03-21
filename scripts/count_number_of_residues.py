import pandas as pd

from rna_motif_library.util import get_pdb_ids
from rna_motif_library.residue import get_cached_residues


def main():
    df = pd.read_csv("data/csvs/rna_residue_counts.csv")
    count = 0
    for _, row in df.iterrows():
        if row["count"] > 2:
            count += 1
    print(count)
    exit()
    pdb_codes = get_pdb_ids()
    print(len(pdb_codes))
    all_data = []
    rna_ids = ["A", "C", "G", "U"]
    for pdb_code in pdb_codes:
        residues = get_cached_residues(pdb_code)
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
