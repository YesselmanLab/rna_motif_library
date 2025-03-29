import pandas as pd
import os
from rna_motif_library.ligand import LIGAND_DATA_PATH
from rna_motif_library.settings import VERSION


def main():
    df_info = pd.read_json(
        os.path.join(
            LIGAND_DATA_PATH, "summary", "versions", f"v{VERSION}", "ligand_info.json"
        )
    )
    df_new_info = pd.read_json(
        os.path.join(LIGAND_DATA_PATH, "summary", "ligand_info.json")
    )
    df_new_info = df_new_info[~df_new_info["id"].isin(df_info["id"])]
    print(df_new_info)


if __name__ == "__main__":
    main()
