import pandas as pd

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.util import add_motif_indentifier_columns


def get_residue_num(row):
    return len(row["msequence"].replace(" ", "").replace(".", ""))


def main():
    motifs = get_cached_motifs("8YOP")
    motifs_by_name = {m.name: m for m in motifs}
    df = pd.read_csv("data/dataframes/check_motifs/8YOP.csv")
    df = add_motif_indentifier_columns(df, "motif_name")
    df["residue_num"] = df.apply(get_residue_num, axis=1)
    df = df.query("has_missing_residues == 1")
    for i, row in df.iterrows():
        motif_name = row["motif_name"]
        motif = motifs_by_name[motif_name]
        motif.to_cif()


if __name__ == "__main__":
    main()
