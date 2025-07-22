import pandas as pd
import os

from rna_motif_library.motif import get_cached_motifs
from rna_motif_library.motif_analysis import parse_motif_indentifier


def add_residue_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column counting non-gap residues in each sequence.

    Args:
        df: DataFrame containing motif data with 'msequence' column

    Returns:
        DataFrame with additional 'residue_num' column counting non-gap characters
    """
    df["motif_1_length"] = df["m_sequence_1"].str.replace("-", "").str.len()
    df["motif_2_length"] = df["m_sequence_2"].str.replace("-", "").str.len()
    return df


def main():
    df = pd.read_json(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "summaries",
            "tertiary_contacts",
            "unique_tertiary_contacts.json",
        )
    )
    df = pd.read_json(
        "lora_compared.json"
    )
    df = df[df["in_their_db"] == False]
    print(len(df))
    exit()
    df = df[df["hbond_num"] > 10]
    # df = df[df["is_isolatable"] == True]
    # df.sort_values("num_hbonds", ascending=False, inplace=True)
    all_motifs = {}
    os.makedirs("tertiary_contacts", exist_ok=True)
    count = 0
    for i, row in df.iterrows():
        _, _, _, pdb_id = parse_motif_indentifier(row["motif_1_id"])
        if pdb_id not in all_motifs:
            all_motifs[pdb_id] = {
                m.name: m for m in get_cached_motifs(pdb_id)
            }
        try:
            motif_1 = all_motifs[pdb_id][row["motif_1_id"]]
            motif_2 = all_motifs[pdb_id][row["motif_2_id"]]
            motif_1.to_cif(f"tertiary_contacts/{count}-{motif_1.name}.cif")
            motif_2.to_cif(f"tertiary_contacts/{count}-{motif_2.name}.cif")
        except Exception as e:
            print(e)
            print(row["motif_1_id"], row["motif_2_id"])
            exit()
        count += 1
        if count > 100:
            break


if __name__ == "__main__":
    main()
