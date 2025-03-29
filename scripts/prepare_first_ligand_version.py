import pandas as pd
import os

from rna_motif_library.ligand import LIGAND_DATA_PATH


def main():
    df_info = pd.read_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_info_complete_final.json")
    )
    df_info.drop(
        columns=["aromatic_rings", "h_acceptors", "h_donors"],
        inplace=True,
    )
    df_features = pd.read_csv(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_features.csv")
    )
    df_features = df_features[
        ["id", "rings", "aromatic_rings", "h_acceptors", "h_donors"]
    ]
    df_info = df_info.merge(df_features, on="id")
    df_info.to_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "versions", "v0", "ligand_info.json"),
        orient="records",
    )


if __name__ == "__main__":
    main()
